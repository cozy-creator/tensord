//! Compile the standalone native intake module without adding a second service owner.
use tensord::native_inputs;

#[test]
fn journal_boundary_records_are_typed_and_tolerate_added_members() {
    let bytes=br#"{"spec":{"actor":"key","request_id":"request","input_id":"file","manifest_sha256":"abc","manifest_length":1,"manifest_canonical_bytes":[123],"content_bytes":0,"retention_id":"id"},"released":false,"receipt":null,"future_observation":true}"#;
    let record: native_inputs::IntakeState = serde_json::from_slice(bytes).unwrap();
    assert_eq!(record.spec.actor, "key");
    assert!(!record.released);
}
