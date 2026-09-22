//! Build-time generation of Rust bindings for the worker FlatBuffers schema.

/// Compiles the worker schema for borrowed tables and direct builders.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let flatc = flatc_fork::flatc();
    flatbuffers_build::BuilderOptions::new_with_files(["schema/worker.fbs"])
        .set_compiler(flatc.to_string_lossy())
        .compile()?;
    Ok(())
}
