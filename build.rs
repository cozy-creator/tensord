fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost = tonic_prost_build::Config::new();
    prost.protoc_executable(protoc);
    tonic_prost_build::configure()
        .generate_default_stubs(true)
        .compile_with_config(
            prost,
            &[
                "vendor/worker-protocol/proto/cozy/worker/v1/worker.proto",
                "proto/cozy/machine/v1/machine.proto",
            ],
            &["vendor/worker-protocol/proto", "proto"],
        )?;
    println!("cargo:rerun-if-changed=vendor/worker-protocol/proto/cozy/worker/v1/worker.proto");
    println!("cargo:rerun-if-changed=proto/cozy/machine/v1/machine.proto");
    Ok(())
}
