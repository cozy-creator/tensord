//! One-shot CPU qualification server, not another machine daemon or scheduler.
use cozy_machine::{
    device_executor::{read_frame, write_answer, Kind},
    model_source_driver,
    model_sources::{ModelSources, SelectedManifest},
    protocol::{recv_fd, send_fd},
    shared_host_plane::{HostConfig, HostScope, SharedHostPlane},
};
use serde::Deserialize;
use std::{
    fs::File,
    io::{self, Write},
    os::unix::net::UnixListener,
    path::PathBuf,
};
#[derive(Deserialize)]
struct Config {
    store: PathBuf,
    selection: SelectedManifest,
    scope: HostScope,
    cache: HostConfig,
    socket: PathBuf,
}
fn main() -> io::Result<()> {
    let config_path = std::env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected CPU qualification config path",
        )
    })?;
    let config: Config = serde_json::from_reader(File::open(config_path)?)?;
    let mut sources = ModelSources::open(&config.store, std::slice::from_ref(&config.selection))?;
    let host = SharedHostPlane::new(&config.store, config.cache)?;
    host.authorize(
        config.scope.clone(),
        config.selection.manifest.clone(),
        sources.authorized_header(&config.selection.manifest)?,
        config.selection.components.clone(),
    )?;
    let listener = UnixListener::bind(&config.socket)?;
    println!("ready");
    io::stdout().flush()?;
    let (mut stream, _) = listener.accept()?;
    let peer = host.register_socket(config.scope, &stream)?;
    while let Some(frame) = read_frame(&mut stream)? {
        let descriptor = if frame.descriptor {
            Some(File::from(recv_fd(&stream)?))
        } else {
            None
        };
        let (mut answer, descriptor) =
            if matches!(frame.kind, Kind::HostTier | Kind::HostTierPrepare) {
                host.request(&peer, &frame, descriptor)
                    .expect("matching host kind")?
            } else if frame.kind == Kind::ModelSourceRead {
                let (answer, file) = model_source_driver::answer(&mut sources, &frame)?;
                (answer, Some(file))
            } else {
                drop(descriptor);
                (
                    cozy_machine::device_executor::Answer::unavailable(frame.seq),
                    None,
                )
            };
        answer.seq = frame.seq;
        answer.descriptor = descriptor.is_some();
        write_answer(&mut stream, &answer)?;
        if let Some(file) = descriptor {
            send_fd(&stream, &file)?;
        }
    }
    // EOF does not prove receiver exit or authorize eviction. SharedHostPlane native claims
    // preserve recipient mappings on owner loss; production pool retains owner until pidfd exit.
    eprintln!("{}", serde_json::to_string(&host.stats()?)?);
    Ok(())
}
