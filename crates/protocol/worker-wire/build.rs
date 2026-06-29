fn main() {
    let flatc = flatc_fork::flatc();
    flatbuffers_build::BuilderOptions::new_with_files(["../worker-ipc-core/schema/worker.fbs"])
        .set_compiler(flatc.to_string_lossy())
        .gen_object_api()
        .compile()
        .expect("worker FlatBuffers schema generation failed");
}
