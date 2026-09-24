//! Checkpoint identity: the digest the head derives for a local checkpoint
//! directory and every rank re-derives for the checkpoint it loads.
//!
//! This is the head's side: when the model is a local directory, no identity
//! was supplied, and the launch is not a stub,
//! `WorkerProcessArgs::derive_checkpoint_identity` calls
//! [`checkpoint_identity`] once before any rank starts, and every launch
//! descriptor carries the result. Each rank derives the identity of the
//! checkpoint it resolves with the Python implementation in
//! `uniserve_models.loading` and refuses a mismatch before loading weights;
//! the head then compares the ranks' reports in `refuse_checkpoint_mismatch`.
//!
//! The rule is shared with the Python loader, which computes the same value,
//! so the two must agree byte for byte. One SHA-256 is fed, per file in
//! lexicographic byte order of its relative POSIX path: the path bytes, NUL,
//! the decimal file size, NUL, and for a sidecar whose contents the identity
//! covers, the lowercase hex SHA-256 of those contents followed by NUL. Weight
//! shards contribute only path and size, so the identity of a checkpoint of
//! hundreds of gigabytes costs a directory walk and a few small reads.

use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use sha2::{Digest as _, Sha256};

/// Suffixes of the sidecars whose contents enter the identity.
const CONTENT_SUFFIXES: [&str; 4] = ["json", "jinja", "model", "txt"];

/// Top-level directories holding training state rather than the served
/// checkpoint; the Python loader excludes the same ones from its sidecars.
const EXCLUDED_DIRECTORIES: [&str; 2] = ["optimizer", "original"];

/// One checkpoint file as the identity sees it.
struct Entry {
    /// Path relative to the checkpoint root, POSIX separated.
    relative: Vec<u8>,
    /// Size in bytes after following a file symlink.
    size: u64,
    /// Where to read the contents when the suffix asks for them.
    path: PathBuf,
}

/// Returns whether one path component is hidden.
fn hidden(name: &std::ffi::OsStr) -> bool {
    name.as_bytes().first() == Some(&b'.')
}

/// Collects every checkpoint file under `root` with its size.
///
/// Hidden entries and the excluded top-level directories are not entered. A
/// symbolic link to a regular file counts as that file, which is how a Hub
/// cache snapshot references its blobs; a link to a directory is not
/// followed, so a checkpoint cannot alias itself into its own identity.
fn walk(root: &Path) -> anyhow::Result<Vec<Entry>> {
    let mut entries = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let listing = std::fs::read_dir(&directory)
            .with_context(|| format!("reading checkpoint directory {}", directory.display()))?;
        for item in listing {
            let item = item
                .with_context(|| format!("reading checkpoint directory {}", directory.display()))?;
            let name = item.file_name();
            if hidden(&name) {
                continue;
            }
            let path = item.path();
            let symlink = item.file_type()?.is_symlink();

            // `metadata` follows links, which is what decides whether a
            // linked entry is a file this rank would read.
            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) => metadata,
                // A dangling link is not a file the checkpoint contains.
                Err(error) if symlink && error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("reading checkpoint entry {}", path.display()));
                }
            };

            if metadata.is_dir() {
                let excluded = directory == root
                    && EXCLUDED_DIRECTORIES
                        .iter()
                        .any(|excluded| name.as_bytes() == excluded.as_bytes());
                if !symlink && !excluded {
                    pending.push(path);
                }
            } else if metadata.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .with_context(|| format!("walked path {} left root", path.display()))?
                    .components()
                    .map(|component| component.as_os_str().as_bytes())
                    .collect::<Vec<_>>()
                    .join(&b'/');
                entries.push(Entry {
                    relative,
                    size: metadata.len(),
                    path,
                });
            }
        }
    }
    Ok(entries)
}

/// Derives the identity of the checkpoint stored in a local directory.
///
/// Returns the lowercase hex SHA-256 the identity rule defines. Fails when
/// the directory or a covered sidecar cannot be read.
pub(crate) fn checkpoint_identity(root: &Path) -> anyhow::Result<String> {
    let mut entries = walk(root)?;
    // Raw byte order of the whole relative path, which is the order the
    // Python side sorts by (`os.fsencode` of the path).
    entries.sort_by(|left, right| left.relative.cmp(&right.relative));

    let mut digest = Sha256::new();
    for entry in &entries {
        digest.update(&entry.relative);
        digest.update(b"\0");
        digest.update(entry.size.to_string().as_bytes());
        digest.update(b"\0");

        let covered = entry
            .path
            .extension()
            .is_some_and(|extension| CONTENT_SUFFIXES.iter().any(|suffix| extension == *suffix));
        if covered {
            let contents = std::fs::read(&entry.path)
                .with_context(|| format!("reading checkpoint sidecar {}", entry.path.display()))?;
            digest.update(hex(&Sha256::digest(&contents)).as_bytes());
            digest.update(b"\0");
        }
    }
    Ok(hex(&digest.finalize()))
}

/// Formats a digest as lowercase hex.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::checkpoint_identity;
    use std::path::Path;

    /// The identity of `fixture`, computed once from the identity rule and
    /// asserted by the Python loader's test over the same fixture.
    const GOLDEN: &str = "0977a85cc96b143616b62095d362701b1694cc5259cfec897efae003d06c89bb";

    /// Builds the shared fixture: sidecars, weight shards, a nested directory,
    /// a hidden file, excluded training state, and symlinked entries.
    fn fixture(root: &Path) {
        std::fs::write(root.join("config.json"), b"{\"architectures\": [\"A\"]}\n").unwrap();
        std::fs::write(root.join("tokenizer.model"), b"spm").unwrap();
        std::fs::write(root.join("chat_template.jinja"), b"{{ messages }}").unwrap();
        std::fs::write(root.join("merges.txt"), b"a b\n").unwrap();
        std::fs::write(
            root.join("model-00001-of-00002.safetensors"),
            vec![1u8; 300],
        )
        .unwrap();
        std::fs::write(
            root.join("model-00002-of-00002.safetensors"),
            vec![2u8; 200],
        )
        .unwrap();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            b"{\"weight_map\": {}}",
        )
        .unwrap();
        std::fs::create_dir(root.join("vae")).unwrap();
        std::fs::write(root.join("vae").join("config.json"), b"{\"z\": 4}").unwrap();
        std::fs::write(root.join("vae").join("weights.pt"), vec![3u8; 50]).unwrap();
        std::fs::write(root.join(".gitattributes"), b"* filter=lfs").unwrap();
        std::fs::create_dir(root.join("optimizer")).unwrap();
        std::fs::write(root.join("optimizer").join("state.pt"), vec![4u8; 10]).unwrap();
        std::fs::create_dir(root.join("original")).unwrap();
        std::fs::write(root.join("original").join("params.json"), b"{}").unwrap();
        // A snapshot references its blobs through links: one sidecar and
        // one shard link to files outside the walked tree.
        std::fs::create_dir(root.join(".blobs")).unwrap();
        std::fs::write(root.join(".blobs").join("gen"), b"{\"eos\": 1}").unwrap();
        std::fs::write(root.join(".blobs").join("shard"), vec![5u8; 120]).unwrap();
        std::os::unix::fs::symlink(
            root.join(".blobs").join("gen"),
            root.join("generation_config.json"),
        )
        .unwrap();
        std::os::unix::fs::symlink(root.join(".blobs").join("shard"), root.join("extra.bin"))
            .unwrap();
        // A linked directory is not entered.
        std::os::unix::fs::symlink(root.join("vae"), root.join("vae_link")).unwrap();
    }

    #[test]
    fn the_fixture_digests_to_the_shared_golden_value() {
        let directory = tempfile::tempdir().unwrap();
        fixture(directory.path());
        assert_eq!(checkpoint_identity(directory.path()).unwrap(), GOLDEN);
    }

    #[test]
    fn sidecar_contents_and_shard_sizes_change_the_identity() {
        let directory = tempfile::tempdir().unwrap();
        fixture(directory.path());
        let original = checkpoint_identity(directory.path()).unwrap();

        // A one-byte change in a config sidecar is a different checkpoint.
        std::fs::write(
            directory.path().join("config.json"),
            b"{\"architectures\": [\"B\"]}\n",
        )
        .unwrap();
        let sidecar_changed = checkpoint_identity(directory.path()).unwrap();
        assert_ne!(sidecar_changed, original);

        // A shard's contents are not read, but its size is part of the
        // identity, so a truncated or padded shard is detected.
        std::fs::write(
            directory.path().join("config.json"),
            b"{\"architectures\": [\"A\"]}\n",
        )
        .unwrap();
        assert_eq!(checkpoint_identity(directory.path()).unwrap(), original);
        std::fs::write(
            directory.path().join("model-00002-of-00002.safetensors"),
            vec![2u8; 201],
        )
        .unwrap();
        assert_ne!(checkpoint_identity(directory.path()).unwrap(), original);
    }
}
