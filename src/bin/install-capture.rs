//! Owned CPU installation component gate, not a public machine intake route.
use cozy_machine::api::{
    auth::VerifiedActor,
    install::{prepare_uploaded, InstallerConfig},
    domain,
    workspaces::WorkspaceUploads,
};
use std::{fs, io::Read, path::PathBuf, sync::Arc};
use tensorfs_core::store::Store;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = None;
    let mut output = None;
    let mut helper = None;
    let mut client = None;
    let mut package = None;
    let mut release = None;
    let mut python = "3.12".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("each option requires a value")?;
        match flag.as_str() {
            "--archive" => archive = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--helper-python" => helper = Some(PathBuf::from(value)),
            "--client-wheel" => client = Some(PathBuf::from(value)),
            "--package" => package = Some(value),
            "--release" => release = Some(value),
            "--python" => python = value,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let output = output.ok_or("--output is required")?;
    fs::create_dir_all(&output)?;
    let archive = archive.ok_or("--archive is required")?;
    let length = fs::metadata(&archive)?.len();
    let store = Arc::new(Store::ensure(&output.join("store"))?);
    let uploads = WorkspaceUploads::open(&output.join("uploads"), store)?;
    let actor = VerifiedActor {
        public_key: [19; 32],
    };
    let file = domain::LocalPackageFileRef {
        filename: "source.tar".into(),
        length,
        ..Default::default()
    };
    let header = domain::LocalPackageUploadHeader {
        operation_id: "installer-component-gate".into(),
        file: Some(file.clone()),
    };
    let mut session = uploads.begin(actor, &header)?;
    if !session.verified() {
        let mut input = fs::File::open(archive)?;
        let mut buffer = vec![0; 1 << 20];
        loop {
            let count = input.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            let offset = session.received();
            session.append(domain::LocalPackageUploadChunk {
                offset,
                data: buffer[..count].to_vec(),
            })?;
        }
    }
    if !session.verified() {
        return Err("incomplete archive".into());
    }
    drop(session);
    let selected = domain::DesiredLocalPackageSet {
        operation_id: header.operation_id.clone(),
        source_archive: "source.tar".into(),
        files: vec![file],
        package: Some(domain::DevelopmentPackage {
            package: package.ok_or("--package is required")?,
            release: release.ok_or("--release is required")?,
            installation_id: "installation-component-gate".into(),
        }),
        ..Default::default()
    };
    let uploaded = uploads
        .package(actor, &selected)?
        .ok_or("uploaded source absent")?;
    let prepared = prepare_uploaded(
        &InstallerConfig {
            helper_python: helper.ok_or("--helper-python is required")?,
            python,
            generations: output.join("generations"),
            client_wheel: client.ok_or("--client-wheel is required")?,
            staging_root: output.join("staging"),
            sdk: vec![],
            uv: "uv".into(),
            seed_cache: None,
        },
        &uploaded,
    )?;
    fs::write(
        output.join("prepared-generation.json"),
        serde_json::to_vec_pretty(&prepared.record)?,
    )?;
    fs::write(
        output.join("package-interface.json"),
        &prepared.interface_bytes,
    )?;
    println!(
        "PREPARED {} {}",
        prepared.record.identity,
        prepared.record.python.display()
    );
    Ok(())
}
