//! Development tool: Read throughput from a far machine. `seed` (on the machine's host) journals
//! one finished run with a large output (MiB); `measure` (on the client) reads it N times through
//! `cozy.machine.v1` Read and prints MB/s. Not proof of
//! anything but the transport.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use cozy_machine::{
    api::{
        capability::{mint, Grant, MACHINE},
        domain, v1,
    },
    journal::{Invocation, Journal, Outcome, ProcessBirth, SubmissionContext},
};
use ed25519_dalek::SigningKey;
use std::{
    io::{self, Read},
    path::PathBuf,
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

/// Bytes from a counter-mode xorshift: incompressible, reproducible, no temp file.
struct Noise {
    left: u64,
    state: u64,
}
impl Read for Noise {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let n = (buffer.len() as u64).min(self.left) as usize & !7;
        for chunk in buffer[..n].chunks_exact_mut(8) {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            chunk.copy_from_slice(&self.state.to_le_bytes());
        }
        self.left -= n as u64;
        Ok(n)
    }
}

fn key(seed: &str) -> SigningKey {
    SigningKey::from_bytes(&tensorfs_core::sha256::digest(seed.as_bytes()))
}

fn seed(state: PathBuf, mib: u64, key_seed: &str) -> io::Result<()> {
    let length = mib << 20;
    let store =
        tensorfs_core::store::Store::ensure(&state.join("tensorfs")).map_err(io::Error::other)?;
    let mut hash = sha2_digest(Noise {
        left: length,
        state: 0x9e37_79b9_7f4a_7c15,
    })?;
    let object = tensorfs_core::ids::ObjectRef {
        sha256: tensorfs_core::sha256::hex(&hash),
        length,
    };
    store
        .put_stream(
            &mut Noise {
                left: length,
                state: 0x9e37_79b9_7f4a_7c15,
            },
            Some(&object),
            &Default::default(),
        )
        .map_err(io::Error::other)?;
    let actor = tensorfs_core::sha256::hex(key(key_seed).verifying_key().as_bytes());
    let mut journal = Journal::open(&state.join("execution"))?;
    let digest = format!("sha256:{}", "a".repeat(64));
    let record = journal.accept_public(
        SubmissionContext {
            actor,
            request_id: "bench".into(),
            submission_id: "bench".into(),
            expected_workspace_id: journal.workspace_id().into(),
            capture_digest: digest.clone(),
            invocation_digest: digest.clone(),
            payload_digest: digest,
            ..Default::default()
        },
        Invocation {
            package: "bench/blob".into(),
            generation: "b".repeat(32),
            module: "bench:app".into(),
            entrypoint: "blob".into(),
            input: serde_json::json!({}),
            attention_kernel: String::new(),
            inputs: Default::default(),
            ..Default::default()
        },
    )?;
    journal.claim(&record.id)?;
    let mut holder = Command::new("sleep").arg("60").spawn()?;
    let birth = cozy_machine::execution::process_birth(holder.id())?;
    journal.register_process(&record.id, ProcessBirth { ..birth })?;
    journal.running(&record.id, None)?;
    let product = domain::RunProduct {
        output: "blob".into(),
        op: domain::RunProductOp::Set as i32,
        content: Some(domain::Ref {
            digest: hash.to_vec(),
            length,
        }),
        media_type: "application/octet-stream".into(),
        ..Default::default()
    };
    journal.append_product(&record.id, None, &cozy_machine::products::encode(&product))?;
    journal.finish(
        &record.id,
        Outcome::Failed(cozy_machine::journal::Failure::machine(
            "bench_seeded",
            "bench output seeded",
        )),
    )?;
    let _ = holder.kill();
    let _ = holder.wait();
    hash.fill(0);
    println!(
        "{{\"run\":\"bench\",\"number\":{},\"bytes\":{length}}}",
        record.id
    );
    Ok(())
}

fn sha2_digest(mut source: impl Read) -> io::Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let n = source.read(&mut buffer)?;
        if n == 0 {
            return Ok(hash.finalize().into());
        }
        hash.update(&buffer[..n]);
    }
}

/// `measure HOST:PORT LEAF.pem WORKER KEYSEED RUN_NUMBER REPEAT [LIMIT_BYTES [SERVER_PID
/// [WINDOWS]]]`: per attempt, one Read per HTTP/2 window mode, then one HTTPS GET.
async fn measure(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (address, pem, worker, key_seed) = (
        &args[2],
        &std::fs::read_to_string(&args[3])?,
        &args[4],
        &args[5],
    );
    let repeat: u32 = args[7].parse()?;
    let limit: u64 = args.get(8).map_or(Ok(u64::MAX), |l| l.parse())?;
    let server: Option<u32> = args.get(9).map(|p| p.parse()).transpose()?;
    let window = args.get(10).map_or("adaptive", String::as_str);
    // CPU seconds from /proc stat: utime+stime (self or server).
    let ticks = |pid: &str, field: usize| -> f64 {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        let rest: Vec<&str> = stat
            .rsplit_once(") ")
            .map_or(vec![], |(_, r)| r.split(' ').collect());
        let get = |i: usize| {
            rest.get(i)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        (get(field) + get(field + 1)) / 100.0
    };
    let server_cpu = || server.map_or(0.0, |pid| ticks(&pid.to_string(), 11));
    let signer = key(key_seed);
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    let machine = mint(
        &signer,
        Grant {
            machine: worker.into(),
            action: MACHINE.into(),
            expires: now + 3600,
            ..Default::default()
        },
    );
    // Each transfer opens its own connection: adaptive (BDP), fixed (16 MiB
    // stream / 32 MiB connection) or default (hyper's own) HTTP/2 windows.
    let connect = |window: &str| {
        Endpoint::from_shared(format!("https://{address}")).map(|endpoint| {
            endpoint
                .http2_adaptive_window(window == "adaptive")
                .initial_stream_window_size((window == "fixed").then_some(16 << 20))
                .initial_connection_window_size((window == "fixed").then_some(32 << 20))
        })
    };
    for attempt in 1..=repeat {
        for window in window.split(',') {
            // Timed from the connection, as curl's total is.
            let (cpu, served) = (ticks("self", 11), server_cpu());
            let started = Instant::now();
            let channel = connect(window)?
                .tls_config(
                    ClientTlsConfig::new()
                        .ca_certificate(Certificate::from_pem(pem))
                        .domain_name("localhost"),
                )?
                .connect()
                .await?;
            let mut client =
                v1::machine_client::MachineClient::new(channel).max_decoding_message_size(16 << 20);
            let mut request = tonic::Request::new(v1::ReadRequest {
                target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                    run: "bench".into(),
                    output: "blob".into(),
                    index: 0,
                    ..Default::default()
                })),
                ..Default::default()
            });
            request
                .metadata_mut()
                .insert("authorization", format!("Cozy-Cap {machine}").parse()?);
            let mut stream = client.read(request).await?.into_inner();
            let mut bytes = 0u64;
            while let Some(frame) = stream.message().await? {
                bytes += frame.data.len() as u64;
                if bytes >= limit {
                    break;
                }
            }
            let seconds = started.elapsed().as_secs_f64();
            let (cpu, served) = (ticks("self", 11) - cpu, server_cpu() - served);
            println!("{{\"path\":\"grpc-read-{}\",\"attempt\":{attempt},\"bytes\":{bytes},\"seconds\":{seconds:.3},\"mb_per_s\":{:.1},\"client_cpu_s\":{cpu:.2},\"server_cpu_s\":{served:.2}}}", window, bytes as f64 / seconds / 1e6);
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("seed") => seed(PathBuf::from(&args[2]), args[3].parse()?, &args[4])?,
        Some("key") => println!("{}", URL_SAFE_NO_PAD.encode(key(&args[2]).verifying_key().as_bytes())),
        Some("measure") => measure(&args).await?,
        _ => eprintln!("usage: read-bench seed STATE MIB KEYSEED | key KEYSEED | measure HOST:PORT LEAF.pem WORKER KEYSEED RUN_NUMBER REPEAT [LIMIT_BYTES [SERVER_PID [WINDOWS: adaptive,fixed,default]]]"),
    }
    Ok(())
}
