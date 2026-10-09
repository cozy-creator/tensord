//! A run's capability against a real AuthKit (th-241): public reads go anonymous; the first
//! private operation trades the owner's device-key-signed capability, once, through the
//! JWT-bearer grant (an assertion the leaf signs, DPoP from the same key); every proof is
//! checked by AuthKit's resource verifier, and the Hub's narrowing, a revoked device key and
//! the capability's expiry each end the run's private work with a typed reason.
#[path = "common/hub_oauth.rs"]
mod hub_oauth;

use cozy_machine::hub::{reason, Capability, Catalog, Leaf, Op, Source};
use serde_json::json;
use std::{sync::Arc, thread::sleep, time::Duration};
use tensorfs_core::transport::{AccessToken, DpopCredential, DpopKey, Publication};

/// A key as the machine's identity holds it: a P-256 PKCS#8 PEM.
fn leaf() -> Arc<Leaf> {
    let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
        .unwrap()
        .serialize_pem();
    Arc::new(Leaf::from_pem(&pem).unwrap())
}

fn publish(
    catalog: &Catalog,
    store: &tensorfs_core::store::Store,
    manifest: &tensorfs_core::ids::ObjectRef,
    operation: &str,
) -> tensorfs_core::err::Result<()> {
    tensorfs_core::transport::publish(&Publication {
        store,
        hub: catalog.origin(),
        destination: "acme/tiny",
        manifest,
        operation,
        credential: catalog.credential(),
        policy: catalog.policy(),
        progress: &|_, _| (),
        streams: 2,
    })
    .map(drop)
}

const TINY: Op = Op::Publish { model: "acme/tiny" };

/// One JSON read at the catalog's Hub, presented as the catalog presents.
fn get(catalog: &Catalog, path: &str) -> Result<serde_json::Value, String> {
    let (body, _) = tensorfs_core::transport::api_get(
        &format!("{}{path}", catalog.origin()),
        catalog.policy(),
        catalog.credential(),
        1 << 20,
        tensorfs_core::transport::Deadline::after_seconds(Some(30.0)),
        &tensorfs_core::transport::Ledger::new(),
    )
    .map_err(|e| e.detail)?;
    serde_json::from_slice(&body).map_err(|e| e.to_string())
}

#[test]
#[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
fn a_run_capability_trades_once_for_exactly_what_it_names() {
    let (upstream, hub) = hub_oauth::test_hub();
    let authkit = hub_oauth::AuthKit::start(&upstream);
    let leaf = leaf();
    let jkt = leaf.thumbprint();
    let hosts = vec!["localhost".to_string()];

    // A public-only run: its reads carry nothing, and AuthKit hears of nothing.
    let public = Source::new(&authkit.hub, None, hosts.clone(), None).unwrap();
    let catalog = Catalog::new(&public).unwrap();
    assert_eq!(get(&catalog, "/v1/packages/acme/pkg").unwrap()["package"], "acme/pkg");
    assert_eq!(hub.lock().unwrap().reads, 1);
    assert_eq!(authkit.exchanges(), 0);
    assert!(authkit.verified().is_empty());

    // A run that publishes: the capability names acme/tiny's checkpoints and one read the
    // Hub narrows away. Nothing is traded until the first private operation.
    let narrowed = format!("sha256:{}", "ab".repeat(32));
    let ops = json!([
        {"type": "tensorhub_model_publish", "model": "acme/tiny"},
        {"type": "tensorhub_model_read", "model": "acme/narrowed", "manifest": narrowed},
    ]);
    let signed = authkit.capability(&jkt, &ops, 600);
    let source = Source::new(&authkit.hub, None, hosts.clone(), Some(Capability::new(&signed, &authkit.token_endpoint, leaf.clone()).unwrap())).unwrap();
    let store_root = std::env::temp_dir().join(format!("cm-oauth-store-{}", uuid::Uuid::new_v4()));
    let store = tensorfs_core::store::Store::ensure(&store_root).unwrap();
    let manifest = hub_oauth::checkpoint(&store, &[7; 8192]);
    assert_eq!(authkit.exchanges(), 0);
    let tiny = Catalog::for_op(&source, TINY).unwrap();
    publish(&tiny, &store, &manifest, "op-1").unwrap();
    publish(&Catalog::for_op(&source, TINY).unwrap(), &store, &manifest, "op-2").unwrap();
    {
        let hub = hub.lock().unwrap();
        assert_eq!(hub.finalized, [manifest.id(), manifest.id()]);
        assert_eq!(hub.credentialed_uploads, 0, "object hosts see no credential");
    }
    assert_eq!(authkit.exchanges(), 1, "one trade per run");
    let verified = authkit.verified();
    assert!(verified.iter().all(|s| s["owner"] == authkit.owner().as_str() && s["actor"] == jkt.as_str()), "{verified:?}");
    // Public reads of the same run stay anonymous.
    get(&Catalog::new(&source).unwrap(), "/v1/packages/acme/pkg").unwrap();
    assert_eq!(authkit.verified().len(), verified.len());

    // An operation the capability does not name never reaches the Hub; one the Hub narrowed
    // away is refused there.
    let other = Op::Publish { model: "acme/other" };
    assert_eq!(Catalog::for_op(&source, other).err().unwrap().0, reason::REQUIRED);
    let read = Catalog::for_op(&source, Op::Read { model: "acme/narrowed", manifest: &narrowed }).unwrap();
    let refused = get(&read, &format!("/v1/models/acme/narrowed/checkpoints/{narrowed}")).unwrap_err();
    assert_eq!(read.reason("catalog_read_failed", &refused).0, reason::EXCEEDED, "{refused:?}");

    // The token is bound to the leaf: presented with another key's proof, it reads nothing.
    struct Fixed(String);
    impl AccessToken for Fixed {
        fn current(&self) -> Option<String> {
            Some(self.0.clone())
        }
    }
    let thief = DpopCredential::new(
        Arc::new(DpopKey::generate().unwrap()),
        Fixed(authkit.last_token()),
        vec!["127.0.0.1".into()],
    )
    .proving_origin(&authkit.resource);
    let finalized = hub.lock().unwrap().finalized.len();
    let stolen = publish_with(&thief, &authkit.hub, tiny.policy(), &store, &manifest);
    assert!(stolen.is_err(), "a stolen token without the key is refused");
    assert_eq!(hub.lock().unwrap().finalized.len(), finalized);

    // Revoking the device key ends the run's private work at its next request, typed, with
    // no second trade.
    authkit.revoke_device_key();
    let tiny = Catalog::for_op(&source, TINY).unwrap();
    assert!(publish(&tiny, &store, &manifest, "op-3").is_err());
    assert_eq!(tiny.reason("weights_publication_failed", "").0, reason::REFUSED);
    assert_eq!(authkit.exchanges(), 1);

    // A capability that expires ends typed too, before the Hub hears of it.
    authkit.enroll_device_key();
    let brief = authkit.capability(&jkt, &ops, 2);
    let source = Source::new(&authkit.hub, None, hosts, Some(Capability::new(&brief, &authkit.token_endpoint, leaf).unwrap())).unwrap();
    sleep(Duration::from_secs(3));
    assert_eq!(Catalog::for_op(&source, TINY).err().unwrap().0, reason::EXPIRED);
    assert_eq!(authkit.exchanges(), 1);
    assert_eq!(hub.lock().unwrap().finalized.len(), finalized);
    let _ = std::fs::remove_dir_all(store_root);
}

/// One publication presented by `credential`, whatever it is.
fn publish_with(
    credential: &dyn tensorfs_core::transport::CredentialProvider,
    hub: &str,
    policy: &tensorfs_core::transport::SourcePolicy,
    store: &tensorfs_core::store::Store,
    manifest: &tensorfs_core::ids::ObjectRef,
) -> tensorfs_core::err::Result<()> {
    tensorfs_core::transport::publish(&Publication {
        store,
        hub,
        destination: "acme/tiny",
        manifest,
        operation: "op-stolen",
        credential,
        policy,
        progress: &|_, _| (),
        streams: 2,
    })
    .map(drop)
}
