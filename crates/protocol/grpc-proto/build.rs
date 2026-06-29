fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let proto_dir = format!("{manifest_dir}/../../proto");
    let proto_file = format!("{proto_dir}/uniserve_grpc.proto");

 // Fail fast with an actionable message if the proto layout has moved,
 // rather than surfacing an opaque protoc error.
    if !std::path::Path::new(&proto_file).exists() {
        return Err(format!(
            "proto file not found at {proto_file} (proto_dir={proto_dir}); \
             expected layout <repo>/proto/uniserve_grpc.proto relative to {manifest_dir}"
        )
        .into());
    }

 // Re-run the build script when the proto source or its directory changes.
    println!("cargo:rerun-if-changed={proto_file}");
    println!("cargo:rerun-if-changed={proto_dir}");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[proto_file], &[proto_dir])?;

    Ok(())
}
