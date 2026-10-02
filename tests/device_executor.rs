//! Actual released Runtime executors and installed classifier package, no handler doubles.
#[path = "../src/device_executor.rs"]
#[allow(dead_code)]
mod device_executor;
#[allow(dead_code)]
#[path = "../src/execution.rs"]
mod execution;
#[allow(dead_code)]
#[path = "../src/journal.rs"]
mod journal;
#[allow(dead_code)]
#[path = "../src/os.rs"]
mod os;
#[allow(dead_code)]
#[path = "../src/protocol.rs"]
mod protocol;

use device_executor::{
    postprocess, read_result, Baseline, Binding, Budgets, DeviceCommand, DeviceExecutor,
    ExecutorConfig, Frame, Services,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::PathBuf,
    sync::Arc,
};

#[derive(Deserialize)]
struct Generation {
    python: PathBuf,
    application: String,
    interface: Value,
}

const FIXTURES: &str =
    "/home/fidika/cozy_v2/outputs/cozy-machine-continued-20261002/session-package-fixtures";

fn prepared(version: &str, generation_path: &str) -> (DeviceExecutor, PathBuf) {
    let generation_path = PathBuf::from(FIXTURES).join(generation_path);
    let hold = File::open(generation_path.parent().unwrap().join(".hold")).unwrap();
    fs2::FileExt::lock_shared(&hold).unwrap();
    let generation: Generation =
        serde_json::from_slice(&fs::read(generation_path).unwrap()).unwrap();
    let root = PathBuf::from(format!(
        "/home/fidika/cozy_v2/outputs/cm-device-20261002/rust-{version}-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&root).unwrap();
    fs::create_dir(root.join("tmp")).unwrap();
    let interface = root.join("package-interface.json");
    fs::write(
        &interface,
        serde_json::to_vec(&generation.interface).unwrap(),
    )
    .unwrap();
    let config = ExecutorConfig {
        python: generation.python,
        root: root.clone(),
        socket: root.join("e.sock"),
        environment: BTreeMap::from([
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("LANG".into(), "C.UTF-8".into()),
            ("OMP_NUM_THREADS".into(), "1".into()),
            ("TMPDIR".into(), root.join("tmp").display().to_string()),
        ]),
        generation_hold: Some(Arc::new(hold)),
    };
    let mut executor = DeviceExecutor::spawn(config).unwrap();
    assert_eq!(executor.hello.runtime_version, version); // fixture provenance, not admission
    assert!(!executor.hello.torch_loaded);
    assert!(executor.hello.executor_protocol_revision > 0);
    assert!(!executor.hello.tensorfs_version.is_empty());
    let maps = fs::read_to_string(format!("/proc/{}/maps", executor.birth.pid)).unwrap();
    assert!(!maps.contains("libcuda.so"));
    assert!(!maps.contains("libnvidia-ml"));
    let devices = executor
        .hello
        .sealed
        .get("CUDA_VISIBLE_DEVICES")
        .cloned()
        .unwrap_or_default();
    let start = executor
        .command(
            &DeviceCommand::Start {
                devices: devices.clone(),
                application: generation.application.clone(),
                package_interface: interface.clone(),
                sequence_parallel_degree: 1,
                import_only: false,
            },
            &mut Baseline,
        )
        .unwrap();
    assert!(start.ok, "{start:?}");
    let binding = Binding {
        application: generation.application,
        package_interface: interface.display().to_string(),
        ..Binding::default()
    };
    let loaded = executor
        .command(
            &DeviceCommand::Load {
                construction: "classifier".into(),
                devices,
                sequence_parallel_degree: 1,
                binding: Box::new(binding),
                budgets: Budgets::default(),
                authorized_device_limit_bytes: None,
                attention_pin: String::new(),
                host_tier: false,
                stages: false,
                descriptor_sources: false,
            },
            &mut Baseline,
        )
        .unwrap();
    assert!(loaded.ok, "{loaded:?}");
    let active = executor
        .command(
            &DeviceCommand::Activate {
                construction: "classifier".into(),
            },
            &mut Baseline,
        )
        .unwrap();
    assert!(active.ok, "{active:?}");
    (executor, root)
}

fn invoke(
    executor: &mut DeviceExecutor,
    root: &std::path::Path,
    id: &str,
    iterations: u64,
    services: &mut impl Services,
) -> Frame {
    let spool = root.join(id);
    fs::create_dir(&spool).unwrap();
    let prepared=executor.command(&DeviceCommand::PrepareRequest{request_id:id.into(),construction:"classifier".into(),entrypoint:"classify".into(),payload:json!({"samples":[[5.1,3.5,1.4,0.2],[6.,2.7,5.1,1.6],[6.7,3.1,4.7,1.5]],"iterations":iterations,"seed":19})},&mut Baseline).unwrap();
    assert!(prepared.ok, "{prepared:?}");
    executor
        .command(
            &DeviceCommand::Invoke {
                request_id: id.into(),
                construction: "classifier".into(),
                entrypoint: "classify".into(),
                spool,
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: -1,
                stages: false,
            },
            services,
        )
        .unwrap()
}

#[test]
#[ignore = "actual SDK gate needs explicitly installed current/older generation fixtures"]
fn current_and_older_stock_executor_reuse_actual_classifier_and_spooled_result() {
    for (version, generation) in [
        (
            "0.18.99",
            "current-generations/72c4d77b275a474b8c828de950020e18/generation.json",
        ),
        (
            "0.18.89",
            "older-generations/1b1fd9d131b34e0faf1568f5e360b694/generation.json",
        ),
    ] {
        let (mut executor, root) = prepared(version, generation);
        for (index, id) in ["first", "second", "third"].iter().enumerate() {
            let reply = invoke(&mut executor, &root, id, 2, &mut Baseline);
            let result = read_result(&root.join(id), &reply).unwrap();
            assert_eq!(result["predictions"], json!([0, 2, 1]));
            assert_eq!(result["call_sequence"], index + 1);
            assert_eq!(reply.outputs.len(), 1);
            assert!(reply.frames.is_empty());
            let output = &reply.outputs[0];
            assert_eq!(output.output_id, "report");
            assert_eq!(output.kind, "file");
            assert_eq!(output.media_type, "application/json");
            assert_eq!(output.size_bytes, Some(273));
            assert!(output.digest.starts_with("blake2b:"));
            assert!(output.asset_ref.starts_with(&format!("attempt:{id}/")));
            let (retained, bindings) =
                postprocess(&executor.codec(), &root.join(id), &reply).unwrap();
            assert_eq!(retained, result);
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].asset_ref, output.asset_ref);
            assert_eq!(bindings[0].producer_digest, output.digest);
            let bytes = fs::read(root.join(id).join(&bindings[0].name)).unwrap();
            assert_eq!(
                bindings[0].sha256,
                tensorfs_core::sha256::hex_digest(&bytes)
            );
            let saved: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(saved["predictions"], json!([0, 2, 1]));
            executor.cancel(id).unwrap(); // stale matching old attempt cannot cancel the next one
            let maps = fs::read_to_string(format!("/proc/{}/maps", executor.birth.pid)).unwrap();
            assert!(!maps.contains("libcuda.so"));
            assert!(!maps.contains("libnvidia-ml"));
        }
        let probe = executor
            .command(&DeviceCommand::Probe { collect: false }, &mut Baseline)
            .unwrap();
        assert!(probe.ok);
        executor.shutdown().unwrap();
    }
}

#[test]
#[ignore = "actual SDK gate needs explicitly installed generation fixture"]
fn credential_refusal_and_result_mutation_are_operation_local() {
    let (mut executor, root) = prepared(
        "0.18.99",
        "current-generations/72c4d77b275a474b8c828de950020e18/generation.json",
    );
    let secret = executor
        .command(
            &DeviceCommand::PrepareRequest {
                request_id: "rejected".into(),
                construction: "classifier".into(),
                entrypoint: "classify".into(),
                payload: json!({"credential":"must-not-cross"}),
            },
            &mut Baseline,
        )
        .unwrap_err();
    assert_eq!(secret.kind(), std::io::ErrorKind::PermissionDenied);
    let reply = invoke(&mut executor, &root, "valid", 2, &mut Baseline);
    assert!(read_result(&root.join("valid"), &reply).is_ok());
    fs::write(root.join("valid/result.canonical"), b"mutated").unwrap();
    assert_eq!(
        read_result(&root.join("valid"), &reply).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    executor.shutdown().unwrap();
}

#[test]
#[ignore = "actual SDK image gate needs explicitly installed cpu_image generation"]
fn stock_executor_deferred_webp_reuses_sdk_encoder_and_exact_asset_binding() {
    let (mut executor,root)=prepared("0.18.99","/home/fidika/cozy_v2/outputs/cm-device-20261002/image-generations/43a2a90811c34ec09434644f3b977414/generation.json");
    let spool = root.join("image");
    fs::create_dir(&spool).unwrap();
    let prepared = executor
        .command(
            &DeviceCommand::PrepareRequest {
                request_id: "image".into(),
                construction: "classifier".into(),
                entrypoint: "render".into(),
                payload: json!({"samples":[[5.1,3.5,1.4,0.2],[6.,2.7,5.1,1.6],[6.7,3.1,4.7,1.5]]}),
            },
            &mut Baseline,
        )
        .unwrap();
    assert!(prepared.ok, "{prepared:?}");
    let reply = executor
        .command(
            &DeviceCommand::Invoke {
                request_id: "image".into(),
                construction: "classifier".into(),
                entrypoint: "render".into(),
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: -1,
                stages: false,
            },
            &mut Baseline,
        )
        .unwrap();
    assert_eq!(reply.frames.len(), 1);
    assert_eq!(reply.frames[0].codec, "webp");
    assert_eq!(reply.frames[0].raw_bytes, 32 * 32 * 3);
    let (result, bindings) = postprocess(&executor.codec(), &spool, &reply).unwrap();
    assert_eq!(result["predictions"], json!([0, 2, 1]));
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].output_id, "image");
    assert_eq!(bindings[0].media_type, "image/webp");
    assert_eq!(bindings[0].asset_ref, reply.outputs[0].asset_ref);
    let path = spool.join(&bindings[0].name);
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
    let decoded=std::process::Command::new(&executor.codec().python).args(["-I","-c","import json,sys;from PIL import Image;i=Image.open(sys.argv[1]).convert('RGB');print(json.dumps({'size':i.size,'pixels':[i.getpixel((x,16))for x in(2,15,29)]}))"]).arg(path).output().unwrap();
    assert!(decoded.status.success());
    let decoded: Value = serde_json::from_slice(&decoded.stdout).unwrap();
    assert_eq!(decoded["size"], json!([32, 32]));
    for (actual, wanted) in decoded["pixels"].as_array().unwrap().iter().zip([
        [230, 50, 50],
        [50, 50, 230],
        [50, 230, 50],
    ]) {
        for (actual, wanted) in actual.as_array().unwrap().iter().zip(wanted) {
            assert!((actual.as_i64().unwrap() - wanted).abs() <= 8);
        }
    }
    executor.shutdown().unwrap();
}
