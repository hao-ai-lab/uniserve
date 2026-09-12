use std::io::Write;
use std::sync::Arc;
use uniserve_core::SharedMedia;

#[test]
fn claimed_media_remains_readable_after_its_name_and_first_consumer_are_gone() {
    let mut file = tempfile::NamedTempFile::new_in("/dev/shm").unwrap();
    file.write_all(b"complete media bytes").unwrap();
    let name = file.path().file_name().unwrap().to_str().unwrap();
    // SAFETY: the fixture has completed its only write and never resizes this file.
    let media = Arc::new(unsafe { SharedMedia::open(name, 20) }.unwrap());
    assert_eq!(media.len(), 20);
    assert!(!media.is_empty());
    // SAFETY: the publication remains immutable; its name has already been claimed.
    assert!(unsafe { SharedMedia::open(name, 20) }.is_err());

    let download = Arc::clone(&media);
    drop(media);
    drop(file);
    assert_eq!(download.as_bytes(), b"complete media bytes");
}

#[test]
fn a_truncated_publication_is_rejected_and_releases_its_named_storage() {
    let mut file = tempfile::NamedTempFile::new_in("/dev/shm").unwrap();
    file.write_all(b"short").unwrap();
    let name = file.path().file_name().unwrap().to_str().unwrap();
    // SAFETY: the fixture does not write or resize after publication.
    assert!(unsafe { SharedMedia::open(name, 6) }.is_err());
    assert!(unsafe { SharedMedia::open(name, 5) }.is_err());
}
