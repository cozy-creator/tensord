fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost = tonic_prost_build::Config::new();
    prost.protoc_executable(protoc);
    tonic_prost_build::configure()
        .compile_with_config(
            prost,
            &[
                "proto/cozy/machine/v1/machine.proto",
            ],
            &["proto"],
        )?;
    println!("cargo:rerun-if-changed=proto/cozy/machine/v1/machine.proto");
    client_wheel()
}

/// Packs python/cozy_machine_client as a pure wheel the binary embeds (`src/machine/client.rs`):
/// package environments and the installer helper take it from the machine, not the image.
/// Deterministic: sorted members, fixed timestamps.
fn client_wheel() -> Result<(), Box<dyn std::error::Error>> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use sha2::{Digest, Sha256};
    use std::io::Write;
    println!("cargo:rerun-if-changed=python/cozy_machine_client");
    println!("cargo:rerun-if-changed=pyproject.toml");
    let project = std::fs::read_to_string("pyproject.toml")?;
    let version = project
        .lines()
        .find_map(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .ok_or("pyproject.toml names no version")?;
    let info = format!("cozy_machine_client-{version}.dist-info");
    let mut members: Vec<(String, Vec<u8>)> = vec![];
    let mut sources: Vec<_> = std::fs::read_dir("python/cozy_machine_client")?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    sources.sort();
    for path in sources
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "py"))
    {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        members.push((format!("cozy_machine_client/{name}"), std::fs::read(&path)?));
    }
    members.push((format!("{info}/METADATA"), format!("Metadata-Version: 2.1\nName: cozy-machine-client\nVersion: {version}\nRequires-Python: >=3.11\nRequires-Dist: msgspec<1,>=0.18\nProvides-Extra: installer\nRequires-Dist: packaging<27,>=24; extra == \"installer\"\n").into_bytes()));
    members.push((format!("{info}/WHEEL"), b"Wheel-Version: 1.0\nGenerator: tensord build.rs\nRoot-Is-Purelib: true\nTag: py3-none-any\n".to_vec()));
    let mut record = String::new();
    for (name, body) in &members {
        let digest = URL_SAFE_NO_PAD.encode(Sha256::digest(body));
        record.push_str(&format!("{name},sha256={digest},{}\n", body.len()));
    }
    record.push_str(&format!("{info}/RECORD,,\n"));
    members.push((format!("{info}/RECORD"), record.into_bytes()));
    let name = format!("cozy_machine_client-{version}-py3-none-any.whl");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("client.whl");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&out)?);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .last_modified_time(zip::DateTime::default());
    for (member, body) in &members {
        zip.start_file(member.as_str(), options)?;
        zip.write_all(body)?;
    }
    zip.finish()?;
    println!("cargo:rustc-env=COZY_CLIENT_WHEEL_NAME={name}");
    Ok(())
}
