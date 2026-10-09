use cozy_machine::{
    owner::{Owner, Shared},
    protocol::{self, Body, Command, Reply, CAPS},
};
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
    // Whatever this process inherited (a parent's readiness, key or credential pipe) reaches
    // nothing it starts; descriptors it hands on are handed explicitly.
    cozy_machine::process::cloexec_from(3);
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        // The image entrypoint and the CLI's machine launcher: no arguments, a grant in the
        // environment (read before any thread starts; the one-shot key leaves the environment).
        // An activated Runtime update's service, started by the machine's stable parent.
        Some("service") if args.len() == 0 => match (provider_self(), cozy_machine::machine::grant::from_process()?) {
            (provider, Some(mut grant)) => {
                grant.provider = provider;
                // The parent's key pipe is read and closed whether or not the grant has a key.
                let inherited = cozy_machine::machine::supervise::inherited_key();
                grant.receipt_key = grant.receipt_key.take().or(inherited);
                run_machine(grant, cozy_machine::machine::supervise::inherited())
            }
            (_, None) => Err(io::Error::other("this process has no machine grant")),
        },
        None | Some("run") if args.len() == 0 => match (provider_self(), cozy_machine::machine::grant::from_process()?) {
            (provider, Some(mut grant)) => {
                grant.provider = provider;
                cozy_machine::machine::identity::hold(&grant.layout)?;
                if let Some(key) = grant.developer_key.as_deref().filter(|_| nix::unistd::geteuid().is_root()) {
                    cozy_machine::machine::ssh::start(key)?;
                }
                let paths = cozy_machine::machine::update::Paths::new(&grant.layout.engine(), &grant.layout.root);
                let ready = cozy_machine::machine::supervise::supervise(&paths, grant.receipt_key.as_deref(), grant.provider.as_ref())?;
                run_machine(grant, ready)
            }
            (_, None) => Err(io::Error::other("this process has no machine grant (COZY_WORKER_ID and the rest of a pod's environment)")),
        },
        Some("version") => {
            #[derive(serde::Serialize)]
            struct Version { name: &'static str, implementation: &'static str, version: &'static str, #[serde(skip_serializing_if = "Option::is_none")] commit: Option<&'static str>,
                api: [&'static str; 1], capabilities: Vec<&'static str> }
            // Machine contracts beside the private socket's: clients choose by capability.
            let capabilities = cozy_machine::api::CAPABILITIES.iter().chain(CAPS).chain(&[cozy_machine::machine::provider::CREDENTIAL_CAPABILITY]).copied().collect();
            // `api` names the client API it serves (G/API.md): a controller chooses its machine by it.
            // `commit` is the source a release build was made from (`task release`). The installed
            // Runtime/TensorFS pair is Status's to report; the linked crate's label named neither.
            let record = Version { name: "cozy-machine", implementation: "rust", version: env!("CARGO_PKG_VERSION"), commit: option_env!("COZY_MACHINE_COMMIT"), api: ["cozy.machine.v1"],
                capabilities };
            println!("{}", serde_json::to_string(&record)?); Ok(())
        }
        Some("host-memory") => {
            println!("{}", serde_json::to_string(&cozy_machine::host_memory::read())?);
            Ok(())
        }
        Some("serve") => {
            let mut root = None; let mut generations=None; let mut parallelism=1;
            let mut machine_config=None; let mut listen=None; let mut gpu_config=None;
            let mut installer_python=None; let mut client_wheel=None; let mut package_python="3.12".to_string();
            let mut budget=16*1024*1024; let mut ttl=300;
            let mut sdk=cozy_machine::published::PackageSdk{uv:"uv".into(),..Default::default()};
            while let Some(arg)=args.next() {
                match arg.as_str() {
                    "--state"=>root=args.next().map(PathBuf::from),
                    "--generations"=>generations=args.next().map(PathBuf::from),
                    "--machine-config"=>machine_config=args.next().map(PathBuf::from),
                    "--listen"=>listen=args.next(),
                    "--gpu-config"=>gpu_config=args.next().map(PathBuf::from),
                    "--installer-python"=>installer_python=args.next().map(PathBuf::from),
                    "--client-wheel"=>client_wheel=args.next().map(PathBuf::from),
                    "--package-python"=>package_python=args.next().ok_or_else(||io::Error::other("--package-python requires a Python version"))?,
                    "--uv"=>sdk.uv=args.next().map(PathBuf::from).ok_or_else(||io::Error::other("--uv requires a path"))?,
                    "--package-sdk"=>sdk.requirements.push(args.next().ok_or_else(||io::Error::other("--package-sdk requires a requirement or wheel"))?),
                    "--package-find-links"=>sdk.find_links=args.next().map(PathBuf::from),
                    "--cpu-parallelism"=>parallelism=args.next().ok_or_else(||io::Error::other("--cpu-parallelism requires a count"))?.parse().map_err(io::Error::other)?,
                    "--host-bytes"=>budget=args.next().ok_or_else(||io::Error::other("--host-bytes requires bytes"))?.parse().map_err(io::Error::other)?,
                    "--cache-ttl-seconds"=>ttl=args.next().ok_or_else(||io::Error::other("--cache-ttl-seconds requires seconds"))?.parse().map_err(io::Error::other)?,
                    _=>return Err(io::Error::other(format!("unknown argument {arg}"))),
                }
            }
            let root=root.ok_or_else(||io::Error::other("--state is required"))?;
            let generations=generations.unwrap_or_else(||root.join("generations"));
            let owner=Owner::new(&root,&root.join("tensorfs"),budget,Duration::from_secs(ttl))?;
            let service=cozy_machine::service::Service::open(&root,&generations,parallelism)?;
            if let Some(config)=gpu_config {
                let gpu=cozy_machine::gpu_service::GpuPool::new(&root.join("gpu"),cozy_machine::gpu_service::GpuConfig::load(&config)?,owner.lock().unwrap().store())?;
                service.configure_gpu(gpu)?;
            }
            let listener=bind_control(&owner)?;
            match (machine_config,listen) {
                (Some(config),Some(listen))=>{
                    let identity=cozy_machine::api::MachineIdentity::retained(&cozy_machine::api::identity::MachineConfig::load(&config)?)?;
                    sdk.python=package_python.clone();
                    start_api(&root,&generations,identity,std::net::TcpListener::bind(listen)?,&owner,&service,installer_python,client_wheel,package_python,sdk,None)?
                }
                (None,None)=>{ cozy_machine::jobs::Jobs::configure(&service,owner.lock().unwrap().store(),None)?; }
                _=>return Err(io::Error::other("--machine-config and --listen are required together")),
            }
            serve(owner,service,listener)
        }
        _=>Err(io::Error::other("usage: cozy-machine version --json | host-memory | serve --state PATH [--generations PATH] [--cpu-parallelism N] [--host-bytes N]")),
    }
}
/// The pod's provider credential, kept from everything this machine starts. On a pod's first
/// start this re-executes the machine without the key in its environment: it runs first, before
/// the grant takes the one-shot readiness key out of the environment and before any thread.
fn provider_self() -> Option<cozy_machine::machine::provider::ProviderSelf> {
    if !std::env::vars().any(|(name, _)| name == "COZY_WORKER_ID") {
        return None;
    }
    let root = std::env::var_os("COZY_MACHINE_ROOT").unwrap_or_else(|| "/".into());
    cozy_machine::machine::provider::ProviderSelf::take(std::path::Path::new(&root))
}

/// Runs the machine from its grant: identity and readiness under the machine root, the engine
/// under `var/lib/cozy/rust-machine`, the API on the granted port.
fn run_machine(
    mut grant: cozy_machine::machine::grant::Grant,
    mut ready: cozy_machine::machine::supervise::Ready,
) -> io::Result<()> {
    use cozy_machine::machine::{grant::Lifetime, identity, receipt};
    // An older stable parent may have selected a partially published candidate. Restore the
    // links before opening services, then let that parent select the restored executable.
    let paths = cozy_machine::machine::update::Paths::new(&grant.layout.engine(), &grant.layout.root);
    if cozy_machine::machine::update::recover_activation(&paths)? {
        std::process::exit(cozy_machine::machine::update::REPLACE_EXIT);
    }
    for name in &grant.ignored {
        eprintln!("cozy-machine: ignoring {name}, which this machine does not read");
    }
    let layout = grant.layout.clone();
    let lifetime = identity::open(&layout)?;
    let rental = grant.lifetime == Lifetime::Rental;
    let readiness = receipt::Readiness::open(
        Some(layout.bootstrap.join("readiness-envelope.json")),
        grant.receipt_key.take(),
        rental,
    )?;
    if let Some(attested) = readiness.attested() {
        if receipt::boot_id(&attested).as_deref() != Some(lifetime.boot_id.as_str()) {
            return Err(io::Error::other(
                "the retained readiness envelope names another boot than this machine root",
            ));
        }
    }
    let fresh = readiness.attested().is_none();
    let keys = cozy_machine::api::auth::Keys::fixed(grant.authorized.clone());
    let mut identity = cozy_machine::api::MachineIdentity::machine(
        grant.worker_id.clone(),
        keys.clone(),
        lifetime,
        readiness,
    )?;
    let lifecycle = cozy_machine::machine::lifecycle::Lifecycle::open(
        layout.state.join("idle.json"),
        rental,
        fresh,
    )?;
    identity.lifecycle = Some(lifecycle.clone());
    let engine = layout.engine();
    identity.store = Some(layout.store.clone());
    let generations = engine.join("generations");
    let owner = Owner::new(&engine, &layout.store, 16 * 1024 * 1024, Duration::from_secs(300))?;
    let service = cozy_machine::service::Service::open(&engine, &generations, 1)?;
    let paths = cozy_machine::machine::update::Paths::new(&engine, &layout.root);
    let updates = {
        let (service, admitted) = (service.clone(), lifecycle.clone());
        let idle = move || service.idle().unwrap_or(false) && admitted.admitted() == 0;
        cozy_machine::machine::update::Updates::open(
            paths.clone(),
            Box::new(idle),
            Some(lifecycle.clone()),
        )?
    };
    identity.updates = Some(updates.clone());
    let readiness = identity.readiness.clone();
    std::thread::Builder::new()
        .name("readiness-report".into())
        .spawn(move || {
            readiness.wait_proved();
            // A Runtime update this process started on is now committed.
            if let Err(error) = updates.commit() {
                eprintln!("cozy-machine: Runtime update commit: {error}");
            }
            ready.report();
        })?;
    // A machine with GPUs serves published GPU packages on all of them (the envelope, sorted
    // by index; a degree-K plan uses the first K), with the executor environment the pod
    // proofs used.
    let gpus = receipt::gpus()?;
    if !gpus.is_empty() {
        let devices: Vec<String> = gpus.iter().map(|g| g.device_index.to_string()).collect();
        let home = engine.join("executor-home");
        std::fs::create_dir_all(&home)?;
        let mut config = serde_json::json!({
            "devices": devices.join(","),
            "authorized_device_limit_bytes": null,
            "pinned_budget_bytes": 4i64 << 30,
            "environment": {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home, "COZY_HOME": home},
            // A rented pod is the machine's alone (its grant says so); anywhere else it is a guest.
            "host": {"mode": if rental { "dedicated" } else { "shared" }},
            "image_kernels": image_kernels(&layout.root, &paths.sdk()),
        });
        // The operator's GPU settings, when the machine root has them, over these defaults.
        if let Ok(file) = std::fs::File::open(layout.state.join("gpu-config.json")) {
            let settings: serde_json::Map<String, serde_json::Value> =
                serde_json::from_reader(file).map_err(io::Error::other)?;
            config.as_object_mut().expect("an object").extend(settings);
        }
        let config: cozy_machine::gpu_service::GpuConfig =
            serde_json::from_value(config).map_err(io::Error::other)?;
        let store = owner.lock().unwrap().store();
        service.configure_gpu(cozy_machine::gpu_service::GpuPool::new(
            &engine.join("gpu"),
            config,
            store,
        )?)?;
    }
    if let Some(hub) = grant.hub.clone().filter(|_| rental) {
        identity.hubs = vec![(hub.origin.clone(), hub.worker_id.clone())];
        let hub = Arc::new(cozy_machine::machine::hub::Hub::new(hub)?);
        let service = service.clone();
        let provider = grant.provider.clone();
        std::thread::Builder::new()
            .name("rental-lifecycle".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a lifecycle runtime");
                runtime.block_on(async {
                    tokio::spawn(cozy_machine::machine::lifecycle::keep_authority(
                        hub.clone(),
                        keys,
                    ));
                    let busy = move || !service.idle().unwrap_or(false);
                    cozy_machine::machine::lifecycle::release_when_idle(lifecycle, hub, provider, busy).await;
                });
                // The Hub accepted the release, or the rental ended itself: this process ends.
                std::process::exit(0);
            })?;
    }
    let control = bind_control(&owner)?;
    let api = std::net::TcpListener::bind((grant.listen_host, grant.worker_port))?;
    identity.webrtc = cozy_machine::machine::player::listen(&layout.state, grant.webrtc_port, !rental)?;
    identity.player = grant.webrtc_reach.clone();
    // The client this binary embeds, and the installer helper over it (made once per client).
    let uv = cozy_machine::machine::client::uv(&layout.root);
    let wheel = cozy_machine::machine::client::wheel(&engine)?;
    let helper = match cozy_machine::machine::client::helper(&engine, &uv, &wheel) {
        Ok(python) => Some(python),
        Err(error) => {
            eprintln!(
                "cozy-machine: no installer helper, so local packages cannot install: {error}"
            );
            None
        }
    };
    start_api(
        &engine,
        &generations,
        identity,
        api,
        &owner,
        &service,
        helper,
        Some(wheel),
        "3.12".into(),
        image_sdk(&paths.sdk(), &layout.root, uv),
        // A run naming no Hub reads this rental's own Hub as the pod (Go agent parity).
        grant.hub.clone().filter(|_| rental).map(|hub| {
            cozy_machine::hub::Source::pod(
                &hub.origin,
                &hub.worker_id,
                &hub.worker_token,
                hub.ca_der,
                hub.object_hosts,
            )
        }),
    )?;
    serve(owner, service, control)
}

/// Executors use the machine's own Runtime/TensorFS pair (the wheels its root ships), so they
/// speak its protocol; without a pair, each release's locked SDK.
/// The image's Runtime interpreter and the executors' Runtime wheel, when both are here: the
/// machine compiles its kernels with them as it starts.
fn image_kernels(root: &std::path::Path, sdk: &std::path::Path) -> Option<serde_json::Value> {
    let python = root.join("opt/cozy/python/bin/python");
    let wheel = std::fs::read_dir(sdk).ok()?.flatten().map(|entry| entry.path()).find(|path| {
        path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name.starts_with("cozy_runtime-") && name.ends_with(".whl")
        })
    })?;
    python.is_file().then(|| serde_json::json!({"python": python, "runtime_wheel": wheel}))
}

fn image_sdk(
    wheels: &std::path::Path,
    root: &std::path::Path,
    uv: PathBuf,
) -> cozy_machine::published::PackageSdk {
    // An activation replaces sdk/current, not the immutable candidate directory.
    // Freeze that selection once: both cache identity and the helper's actual wheel
    // inputs must name the same build even when filenames/version labels repeat.
    let wheels = wheels.canonicalize().unwrap_or_else(|_| wheels.to_path_buf());
    let mut pair: Vec<String> = std::fs::read_dir(&wheels)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == "whl"))
        .map(|path| path.display().to_string())
        .collect();
    pair.sort();
    let complete = pair.iter().any(|w| w.contains("/cozy_runtime-"))
        && pair.iter().any(|w| w.contains("/tensorfs-"));
    let seed = root.join("opt/cozy/dependency-seed/uv-cache");
    cozy_machine::published::PackageSdk {
        uv,
        python: "3.12".into(),
        find_links: complete.then_some(wheels),
        requirements: if complete { pair } else { vec![] },
        client_wheel: None,
        seed_cache: seed.is_dir().then_some(seed),
    }
}

#[cfg(test)]
mod sdk_selection_tests;

#[allow(clippy::too_many_arguments)]
fn start_api(
    root: &std::path::Path,
    generations: &std::path::Path,
    identity: cozy_machine::api::MachineIdentity,
    listener: std::net::TcpListener,
    owner: &Shared,
    service: &Arc<cozy_machine::service::Service>,
    helper: Option<PathBuf>,
    wheel: Option<PathBuf>,
    python: String,
    mut sdk: cozy_machine::published::PackageSdk,
    own_hub: Option<cozy_machine::hub::Source>,
) -> io::Result<()> {
    use cozy_machine::{api, machine_api::NativeBackend};
    sdk.client_wheel = wheel.clone();
    let (pair, uv): (Vec<PathBuf>, _) = (
        sdk.requirements.iter().map(PathBuf::from).collect(),
        sdk.uv.clone(),
    );
    let store = owner.lock().unwrap().store();
    let mut backend = NativeBackend::new(service.clone(), identity.authority.clone(), store.clone());
    backend.own_hub = own_hub;
    let publisher =
        cozy_machine::published::Publisher::new(&root.join("published"), sdk, store.clone())?;
    service.configure_publisher(publisher.clone());
    backend.publisher = Some(publisher);
    // Local packages install only with the helper; package environments get the client either way.
    backend.installer = match (helper, wheel) {
        (Some(helper_python), Some(client_wheel)) => Some(api::install::InstallerConfig {
            helper_python,
            client_wheel,
            python,
            generations: generations.to_path_buf(),
            staging_root: root.join("package-staging"),
            sdk: pair,
            uv: uv.clone(),
        }),
        (Some(_), None) => {
            return Err(io::Error::other(
                "--installer-python requires --client-wheel",
            ))
        }
        _ => None,
    };
    let objects = Arc::new(cozy_machine::objects::Objects::new(
        &root.join("writes"),
        store.clone(),
        service.engine.clone(),
    )?);
    backend.runs = Some(Arc::new(cozy_machine::runs::Runs {
        service: service.clone(),
        objects: objects.clone(),
        publisher: backend.publisher.clone(),
        local: backend.installer.clone().map(|installer| {
            Arc::new(cozy_machine::local_source::LocalSources::new(
                objects,
                installer,
                store.clone(),
            ))
        }),
        own_hub: backend.own_hub.clone(),
        jobs: Default::default(),
    }));
    cozy_machine::jobs::Jobs::configure(service, store.clone(), backend.runs.as_ref())?;
    listener.set_nonblocking(true)?;
    #[derive(serde::Serialize)]
    struct Ready<'a> {
        address: String,
        worker_id: &'a str,
        boot_id: &'a str,
        cert_pem: &'a str,
    }
    let ready = serde_json::to_vec(&Ready {
        address: listener.local_addr()?.to_string(),
        worker_id: &identity.authority.worker_id,
        boot_id: &identity.authority.boot_id,
        cert_pem: &identity.cert_pem,
    })?;
    cozy_machine::machine::identity::write_atomic(&root.join("api-ready.json"), &ready, 0o600)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    std::thread::Builder::new()
        .name("machine-api".into())
        .spawn(move || {
            let result = runtime.block_on(async {
                api::serve(
                    tokio::net::TcpListener::from_std(listener)?,
                    identity,
                    Arc::new(backend),
                )
                .await
            });
            if let Err(error) = result {
                eprintln!("machine API stopped: {error}");
            }
        })?;
    Ok(())
}
fn bind_control(owner: &Shared) -> io::Result<(UnixListener, UnixListener)> {
    let (weights, admin) = {
        let owner = owner.lock().unwrap();
        (owner.socket.clone(), owner.admin.clone())
    };
    Ok((bind_private(&weights)?, bind_private(&admin)?))
}
fn bind_private(path: &std::path::Path) -> io::Result<UnixListener> {
    // Only the singleton owner can remove a stale socket; no active peer store is touched.
    match std::fs::remove_file(path) {
        Ok(()) => (),
        Err(e) if e.kind() == io::ErrorKind::NotFound => (),
        Err(e) => return Err(e),
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}
fn serve(
    owner: Shared,
    service: Arc<cozy_machine::service::Service>,
    (listener, admin): (UnixListener, UnixListener),
) -> io::Result<()> {
    let paths = {
        let owner = owner.lock().unwrap();
        Arc::new([owner.socket.clone(), owner.admin.clone()])
    };
    let stopped = Arc::new(AtomicBool::new(false));
    println!("READY {}", paths[0].display());
    println!("ADMIN {}", paths[1].display());
    io::stdout().flush()?;
    {
        let (owner, stopped, paths, service) = (
            owner.clone(),
            stopped.clone(),
            paths.clone(),
            service.clone(),
        );
        std::thread::Builder::new()
            .name("machine-admin".into())
            .spawn(move || {
                for stream in admin.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    if stopped.load(Ordering::Acquire) {
                        break;
                    }
                    if !matches!(os_peer_outside(&stream), Ok(true)) {
                        let _ = protocol::write(
                            &mut stream,
                            &Reply {
                                seq: 0,
                                body: Body::Error {
                                    code: "connection_unavailable",
                                    detail: "administration is refused to the machine's own \
                                             descendants"
                                        .into(),
                                },
                            },
                        );
                        continue;
                    }
                    let (owner, stopped, paths, service) = (
                        owner.clone(),
                        stopped.clone(),
                        paths.clone(),
                        service.clone(),
                    );
                    std::thread::spawn(move || {
                        if let Err(error) = client(stream, None, &owner, &stopped, &paths, &service)
                        {
                            eprintln!("admin peer: {error}");
                        }
                    });
                }
            })?;
    }
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
        let (owner, stopped, paths, service) = (
            owner.clone(),
            stopped.clone(),
            paths.clone(),
            service.clone(),
        );
        std::thread::spawn(move || {
            let result = client(stream, Some(peer), &owner, &stopped, &paths, &service);
            owner.lock().unwrap().disconnect(peer);
            if let Err(error) = result {
                eprintln!("peer {peer}: {error}");
            }
        });
    }
    for path in paths.iter() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}
fn os_peer_outside(stream: &UnixStream) -> io::Result<bool> {
    cozy_machine::os::peer_descends_from_machine(stream).map(|inside| !inside)
}
/// `peer` is a registered weight peer; `None` is the owner's admin connection.
fn client(
    mut stream: UnixStream,
    peer: Option<u64>,
    owner: &Shared,
    stopped: &AtomicBool,
    paths: &[PathBuf; 2],
    service: &Arc<cozy_machine::service::Service>,
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
            Command::Submit { .. }
            | Command::Execution { .. }
            | Command::Executions
            | Command::Cancel { .. }
            | Command::ReadResult { .. } => Some("execution.cpu/1"),
            _ => None,
        };
        let admin = peer.is_none();
        let wrong = match &request.command {
            Command::Import { .. } | Command::Attach { .. } | Command::Release { .. } => {
                admin.then_some("weight operations use the machine socket")
            }
            Command::Submit { .. }
            | Command::Execution { .. }
            | Command::Executions
            | Command::Cancel { .. }
            | Command::ReadResult { .. }
            | Command::Shutdown => (!admin).then_some("administration uses the admin socket"),
            _ => None,
        };
        if let Some(detail) = wrong {
            if matches!(request.command, Command::Import { .. }) {
                drop(protocol::recv_fd(&stream)?);
            }
            protocol::write(
                &mut stream,
                &Reply {
                    seq,
                    body: Body::Error {
                        code: "wrong_socket",
                        detail: detail.into(),
                    },
                },
            )?;
            continue;
        }
        let peer = peer.unwrap_or(0); // only weight operations, never reached by admin, use it
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
            Command::Submit {
                key,
                generation,
                entrypoint,
                input,
            } => service
                .submit(&key, &generation, &entrypoint, input)
                .map(|record| {
                    (
                        Body::Execution {
                            record: Box::new(record),
                        },
                        None,
                    )
                }),
            Command::Execution { id } => service.engine.get(&id).map(|record| {
                (
                    Body::Execution {
                        record: Box::new(record),
                    },
                    None,
                )
            }),
            Command::Executions => service
                .engine
                .list()
                .map(|records| (Body::Executions { records }, None)),
            Command::Cancel { id } => {
                service
                    .engine
                    .cancel(&id, "local-machine-owner")
                    .map(|record| {
                        (
                            Body::Execution {
                                record: Box::new(record),
                            },
                            None,
                        )
                    })
            }
            Command::ReadResult { id, index } => (|| {
                let record = service.engine.get(&id)?;
                let artifact = record
                    .result
                    .as_ref()
                    .and_then(|r| r.artifacts.get(index))
                    .cloned()
                    .ok_or_else(|| io::Error::other("result artifact absent"))?;
                let file = service.engine.open_result(&id, index)?;
                Ok((Body::ResultArtifact { id, artifact }, Some(file)))
            })(),
            Command::Shutdown => {
                // Hold the weight gate while quiescing dispatch: neither half may
                // stop after discovering the other still has live obligations.
                let mut weight_owner = owner.lock().unwrap();
                if !weight_owner.idle() || !service.stop()? {
                    Err(io::Error::other(
                        "accepted work or active leases prevent shutdown",
                    ))
                } else {
                    assert!(weight_owner.stop());
                    drop(weight_owner);
                    let acknowledgement = protocol::write(
                        &mut stream,
                        &Reply {
                            seq,
                            body: Body::Shutdown,
                        },
                    );
                    stopped.store(true, Ordering::Release);
                    for path in paths {
                        let _ = UnixStream::connect(path);
                    }
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
