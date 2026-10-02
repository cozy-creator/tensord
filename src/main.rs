mod os;
mod owner;
mod protocol;
use owner::{Owner, Shared};
use protocol::{Body, Command, Reply, CAPS};
use std::{
    collections::HashSet,
    io::{self, Write},
    os::unix::{
        fs::PermissionsExt,
        net::{UnixListener, UnixStream},
    },
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("cozy-machine: {error}");
        std::process::exit(1);
    }
}
fn run() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("version") => {
            #[derive(serde::Serialize)]
            struct Version { name: &'static str, version: &'static str, tensorfs: &'static str, capabilities: &'static [&'static str] }
            let record = Version { name: "cozy-machine", version: env!("CARGO_PKG_VERSION"), tensorfs: tensorfs_core::VERSION, capabilities: CAPS };
            println!("{}", serde_json::to_string(&record)?); Ok(())
        }
        Some("serve") => {
            let mut root = None; let mut budget = 16 * 1024 * 1024; let mut ttl = 300;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--state" => root = args.next().map(PathBuf::from),
                    "--host-bytes" => budget = args.next().ok_or_else(|| io::Error::other("--host-bytes requires bytes"))?.parse().map_err(io::Error::other)?,
                    "--cache-ttl-seconds" => ttl = args.next().ok_or_else(|| io::Error::other("--cache-ttl-seconds requires seconds"))?.parse().map_err(io::Error::other)?,
                    _ => return Err(io::Error::other(format!("unknown argument {arg}"))),
                }
            }
            serve(Owner::new(&root.ok_or_else(|| io::Error::other("--state is required for this experimental service"))?,
                budget, Duration::from_secs(ttl))?)
        }
        _ => Err(io::Error::other("usage: cozy-machine version --json | serve --state PATH [--host-bytes N] [--cache-ttl-seconds N]")),
    }
}
fn serve(owner: Shared) -> io::Result<()> {
    let path = owner.lock().unwrap().socket.clone();
    // Only the singleton owner can remove a stale socket; no active peer store is touched.
    match std::fs::remove_file(&path) {
        Ok(()) => (),
        Err(e) if e.kind() == io::ErrorKind::NotFound => (),
        Err(e) => return Err(e),
    }
    let listener = UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    let stopped = Arc::new(AtomicBool::new(false));
    println!("READY {}", path.display());
    io::stdout().flush()?;
    for stream in listener.incoming() {
        let stream = stream?;
        if stopped.load(Ordering::Acquire) {
            break;
        }
        let peer = match owner.lock().unwrap().connect(&stream) {
            Ok(peer) => peer,
            Err(error) => {
                let mut stream = stream;
                let _ = protocol::write(
                    &mut stream,
                    &Reply {
                        seq: 0,
                        body: Body::Error {
                            code: "connection_unavailable",
                            detail: error.to_string(),
                        },
                    },
                );
                continue;
            }
        };
        let (owner, stopped, path) = (owner.clone(), stopped.clone(), path.clone());
        std::thread::spawn(move || {
            let result = client(stream, peer, &owner, &stopped, &path);
            owner.lock().unwrap().disconnect(peer);
            if let Err(error) = result {
                eprintln!("peer {peer}: {error}");
            }
        });
    }
    std::fs::remove_file(path)?;
    Ok(())
}
fn client(
    mut stream: UnixStream,
    peer: u64,
    owner: &Shared,
    stopped: &AtomicBool,
    path: &PathBuf,
) -> io::Result<()> {
    let mut capabilities = HashSet::new();
    loop {
        let request = match protocol::read(&mut stream) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(error) if error.kind() != io::ErrorKind::InvalidData => {
                // A complete malformed JSON request is operation-local; truncated transport ends this peer.
                if !error.to_string().contains("JSON") && error.kind() != io::ErrorKind::Other {
                    return Err(error);
                }
                protocol::write(
                    &mut stream,
                    &Reply {
                        seq: 0,
                        body: Body::Error {
                            code: "invalid_request",
                            detail: error.to_string(),
                        },
                    },
                )?;
                continue;
            }
            Err(error) => return Err(error),
        };
        let seq = request.seq;
        if let Command::Hello {
            runtime,
            capabilities: offered,
        } = request.command
        {
            let _ = runtime; // Version is provenance, never an admission condition.
            capabilities = offered
                .into_iter()
                .filter(|c| CAPS.contains(&c.as_str()))
                .collect();
            let mut selected: Vec<_> = capabilities.iter().cloned().collect();
            selected.sort();
            let incarnation = owner.lock().unwrap().incarnation.clone();
            protocol::write(
                &mut stream,
                &Reply {
                    seq,
                    body: Body::Hello {
                        version: env!("CARGO_PKG_VERSION"),
                        tensorfs: tensorfs_core::VERSION,
                        incarnation,
                        capabilities: selected,
                    },
                },
            )?;
            continue;
        }
        let required = match &request.command {
            Command::Import { .. } => Some("objects.put-fd/1"),
            Command::Attach { .. } => Some("weights.hosted/1"),
            _ => None,
        };
        if let Some(cap) = required {
            if !capabilities.contains(cap) {
                if matches!(request.command, Command::Import { .. }) {
                    drop(protocol::recv_fd(&stream)?);
                }
                protocol::write(
                    &mut stream,
                    &Reply {
                        seq,
                        body: Body::Error {
                            code: "capability_missing",
                            detail: cap.into(),
                        },
                    },
                )?;
                continue;
            }
        }
        let result: io::Result<(Body, Option<std::fs::File>)> = match request.command {
            Command::Import { object } => {
                let fd = protocol::recv_fd(&stream)?;
                owner
                    .lock()
                    .unwrap()
                    .import(object, fd.into())
                    .map(|body| (body, None))
            }
            Command::Attach { object } => owner
                .lock()
                .unwrap()
                .attach(peer, object)
                .map(|(body, fd)| (body, Some(fd))),
            Command::Release { lease, incarnation } => owner
                .lock()
                .unwrap()
                .release(peer, lease, &incarnation)
                .map(|body| (body, None)),
            Command::Stats => Ok((owner.lock().unwrap().stats(), None)),
            Command::Shutdown => {
                if !owner.lock().unwrap().stop() {
                    Err(io::Error::other("active leases prevent shutdown"))
                } else {
                    let acknowledgement = protocol::write(
                        &mut stream,
                        &Reply {
                            seq,
                            body: Body::Shutdown,
                        },
                    );
                    stopped.store(true, Ordering::Release);
                    let _ = UnixStream::connect(path);
                    return acknowledgement;
                }
            }
            _ => Ok((
                Body::Error {
                    code: "operation_unsupported",
                    detail: "this operation is not implemented".into(),
                },
                None,
            )),
        };
        let (body, fd) = result.unwrap_or_else(|e| {
            (
                Body::Error {
                    code: "operation_failed",
                    detail: e.to_string(),
                },
                None,
            )
        });
        protocol::write(&mut stream, &Reply { seq, body })?;
        if let Some(fd) = fd {
            protocol::send_fd(&stream, &fd)?;
        }
    }
}
