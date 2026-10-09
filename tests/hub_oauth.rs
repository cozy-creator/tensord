//! The machine's Hub grants against a real AuthKit (th-238): codes redeemed with the machine's
//! leaf key, DPoP-bound tokens presented and refreshed by the hub module over TensorFS, and
//! every proof checked by AuthKit's resource verifier.
#[path = "common/hub_oauth.rs"]
mod hub_oauth;

use cozy_machine::hub::{self, Authorization, Catalog, Grants, Publishing, Source};
use serde_json::Value;
use std::{sync::Arc, thread::sleep, time::Duration};
use tensorfs_core::transport::{AccessToken, DpopKey, Publication};

fn code(issued: Value) -> Authorization {
    let field = |name: &str| issued[name].as_str().unwrap().to_string();
    Authorization {
        issuer: field("issuer"),
        code: field("code"),
        code_verifier: field("code_verifier"),
        redirect_uri: field("redirect_uri"),
        resource: field("resource"),
    }
}

/// A key as the machine's identity holds it: a P-256 PKCS#8 PEM.
fn leaf_key() -> Arc<DpopKey> {
    let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
        .unwrap()
        .serialize_pem();
    Arc::new(DpopKey::from_pem(&pem).unwrap())
}

fn publish(
    publishing: &Publishing,
    store: &tensorfs_core::store::Store,
    manifest: &tensorfs_core::ids::ObjectRef,
    operation: &str,
) {
    tensorfs_core::transport::publish(&Publication {
        store,
        hub: publishing.origin(),
        destination: "acme/tiny",
        manifest,
        operation,
        credential: publishing,
        policy: publishing.policy(),
        progress: &|_, _| (),
        streams: 2,
    })
    .unwrap();
}

#[test]
#[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
fn grants_redeem_present_refresh_and_end_as_authkit_decides() {
    let (upstream, hub) = hub_oauth::test_hub();
    let authkit = hub_oauth::AuthKit::start(&upstream);
    let key = leaf_key();
    let jkt = key.thumbprint();
    let grants = Grants::new(key);

    // Execution: the code redeems with the leaf's key, and a catalog read passes AuthKit's
    // verifier (its first proof is challenged for a nonce and retried with it).
    let execution = grants
        .redeem(
            &code(authkit.authorize("execution", &jkt)),
            hub::EXECUTION,
            None,
        )
        .unwrap();
    let source = Source::granted(
        &authkit.hub,
        execution.clone(),
        None,
        vec!["localhost".into()],
    );
    let catalog = Catalog::new(&source).unwrap();
    assert_eq!(
        catalog.json("/v1/packages/acme/pkg").unwrap()["package"],
        "acme/pkg"
    );
    assert_eq!(hub.lock().unwrap().reads, 1);
    let seen = authkit.seen();
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0]["type"], hub::EXECUTION);

    // A code redeems once, only with the key it names, and only as the grant it is.
    let spent = code(authkit.authorize("execution", &jkt));
    grants.redeem(&spent, hub::EXECUTION, None).unwrap();
    let again = grants.redeem(&spent, hub::EXECUTION, None).unwrap_err();
    assert!(again.0.contains("invalid_grant"), "{again:?}");
    let stolen = Grants::new(leaf_key()).redeem(
        &code(authkit.authorize("execution", &jkt)),
        hub::EXECUTION,
        None,
    );
    assert!(stolen.is_err(), "another key redeems nothing");
    let mixed = grants
        .redeem(
            &code(authkit.authorize("execution", &jkt)),
            hub::PUBLICATION,
            None,
        )
        .unwrap_err();
    assert!(
        mixed.0.contains("not tensorhub_machine_publication"),
        "{mixed:?}"
    );

    // Publication: an offline grant writes the destination; the object host sees no
    // credential, and an execution grant is refused on publication routes.
    let store_root = std::env::temp_dir().join(format!("cm-oauth-store-{}", uuid::Uuid::new_v4()));
    let store = tensorfs_core::store::Store::ensure(&store_root).unwrap();
    let manifest = hub_oauth::checkpoint(&store, &[7; 8192]);
    let publication = grants
        .redeem(
            &code(authkit.authorize("publication", &jkt)),
            hub::PUBLICATION,
            None,
        )
        .unwrap();
    publish(
        &Publishing::new(&source, &publication, None).unwrap(),
        &store,
        &manifest,
        "op-1",
    );
    {
        let hub = hub.lock().unwrap();
        assert_eq!(hub.finalized, [manifest.id()]);
        assert_eq!(hub.credentialed_uploads, 0);
        assert!(!hub.uploaded.is_empty());
    }
    let wrong = tensorfs_core::transport::publish(&Publication {
        store: &store,
        hub: &authkit.hub,
        destination: "acme/tiny",
        manifest: &manifest,
        operation: "op-wrong",
        credential: catalog.credential(),
        policy: catalog.policy(),
        progress: &|_, _| (),
        streams: 2,
    });
    assert!(wrong.is_err(), "an execution grant publishes nothing");

    // Access tokens live 4 s and renew at half life: a later read presents a fresh one.
    let before = authkit
        .seen()
        .iter()
        .rfind(|s| s["type"] == hub::EXECUTION)
        .unwrap()["token"]
        .clone();
    sleep(Duration::from_millis(2500));
    catalog.json("/v1/packages/acme/pkg").unwrap();
    let after = authkit.seen().last().unwrap()["token"].clone();
    assert_ne!(before, after, "the refreshed token is presented");

    // Signing out ends the execution grant at its next refresh, and its reads with it; the
    // offline publication grant outlives the sign-in.
    authkit.logout();
    sleep(Duration::from_millis(2500));
    assert!(catalog.json("/v1/packages/acme/pkg").is_err());
    let ended = execution.ended().expect("the execution grant ended");
    assert!(ended.contains("invalid_grant"), "{ended}");
    assert_eq!(hub.lock().unwrap().reads, 2);
    assert!(
        publication.current().is_some(),
        "the publication grant refreshes after sign-out"
    );
    publish(
        &Publishing::new(&source, &publication, None).unwrap(),
        &store,
        &manifest,
        "op-2",
    );
    assert_eq!(hub.lock().unwrap().finalized.len(), 2);
    let _ = std::fs::remove_dir_all(store_root);
}
