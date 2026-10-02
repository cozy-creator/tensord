//! Isolated CPU contract fixture, never registered as the user's owned machine.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use cozy_machine::api::{self, auth::Authority, pb, MachineBackend, MachineIdentity};
use ed25519_dalek::VerifyingKey;
use std::{path::PathBuf, sync::Arc};
use tonic::Status;

struct InspectionBackend {
    authority: Authority,
}
impl MachineBackend for InspectionBackend {
    fn workspace(
        &self,
        query: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status> {
        if query.describe.is_some() {
            return Err(Status::unimplemented(
                "package description is not implemented by the inspection fixture",
            ));
        }
        Ok(pb::MachineExecutionWorkspace {
            worker_id: self.authority.worker_id.clone(),
            worker_boot_id: self.authority.boot_id.clone(),
            accelerator_backend: "none".into(),
            ..Default::default()
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut keys = vec![];
    let mut receipt_key = None;
    let mut worker_id = None;
    let mut ready_file = None;
    let mut listen = "127.0.0.1:0".to_string();
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("every option requires a value")?;
        match arg.as_str() {
            "--listen" => listen = value,
            "--worker-id" => worker_id = Some(value),
            "--owner-key" => {
                let raw: [u8; 32] = URL_SAFE_NO_PAD
                    .decode(value)?
                    .try_into()
                    .map_err(|_| "--owner-key must encode 32 Ed25519 bytes")?;
                keys.push(VerifyingKey::from_bytes(&raw)?);
            }
            "--receipt-key-file" => receipt_key = Some(std::fs::read(value)?),
            "--ready-file" => ready_file = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let identity = MachineIdentity::ephemeral(
        worker_id.ok_or("--worker-id is required")?,
        keys,
        receipt_key.ok_or("--receipt-key-file is required")?,
    )?;
    let backend = Arc::new(InspectionBackend {
        authority: identity.authority.clone(),
    });
    let listener = tokio::net::TcpListener::bind(listen).await?;
    #[derive(serde::Serialize)]
    struct Ready<'a> {
        address: String,
        worker_id: &'a str,
        boot_id: &'a str,
        cert_pem: &'a str,
    }
    let bytes = serde_json::to_vec(&Ready {
        address: listener.local_addr()?.to_string(),
        worker_id: &identity.authority.worker_id,
        boot_id: &identity.authority.boot_id,
        cert_pem: &identity.cert_pem,
    })?;
    if let Some(path) = ready_file {
        std::fs::write(path, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes)?);
    api::serve(listener, identity, backend).await
}
