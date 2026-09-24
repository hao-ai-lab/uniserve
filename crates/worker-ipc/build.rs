//! Build-time generation of Rust bindings for the worker FlatBuffers schema.
//!
//! The generated code lands in `${OUT_DIR}/flatbuffers/mod.rs`, which the
//! crate's `schema` module includes and the `codec` module maps to the owned
//! protocol types. `flatbuffers-build` emits `cargo:rerun-if-changed` for the
//! schema file, so editing it regenerates the bindings. The `_uniserve_ipc`
//! Python extension (`worker-ipc-py`) links this crate, so an installed
//! extension keeps the schema it was built with until it is rebuilt.

/// Compiles the worker schema for borrowed tables and direct builders.
///
/// The object API is not generated. The compiler is the `flatc` that the
/// `flatc-fork` build dependency builds from vendored sources, not a system
/// `flatc`; `flatbuffers-build` refuses a compiler whose version differs from
/// the one it supports.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let flatc = flatc_fork::flatc();
    flatbuffers_build::BuilderOptions::new_with_files(["schema/worker.fbs"])
        .set_compiler(flatc.to_string_lossy())
        .compile()?;
    Ok(())
}
