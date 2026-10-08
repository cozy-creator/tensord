//! Retained byte allowance from the actual installed callable, not a caller's description.
use serde_json::Value;
use std::io;

pub const DEFAULT_BYTES: u64 = 256 << 20;

/// Keep the ordinary working allowance, but honor larger declared fixed output slots.
/// Model manifests use their separate weights grant and contribute no asset capacity.
pub fn for_callable(interface: &Value, kind: &str, name: &str) -> io::Result<u64> {
    let row = interface[kind]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["name"] == name))
        .ok_or_else(|| invalid("installed output callable is absent"))?;
    Ok(capacity(&row["result"], None)?.max(DEFAULT_BYTES))
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail)
}

fn bound(field: &Value) -> io::Result<Option<u64>> {
    match field
        .get("asset_bound")
        .and_then(|value| value.get("max_bytes"))
    {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|n| *n > 0)
            .map(Some)
            .ok_or_else(|| invalid("installed asset max_bytes must be a positive integer")),
    }
}

fn sum(mut values: impl Iterator<Item = io::Result<u64>>) -> io::Result<u64> {
    values.try_fold(0u64, |total, value| {
        total
            .checked_add(value?)
            .ok_or_else(|| invalid("installed output byte capacity overflows"))
    })
}

fn capacity(schema: &Value, bytes: Option<u64>) -> io::Result<u64> {
    if schema.get("asset").is_some() || schema.get("input").and_then(Value::as_str) == Some("tree")
    {
        return Ok(bytes.unwrap_or(DEFAULT_BYTES));
    }
    if let Some(fields) = schema.get("fields").and_then(Value::as_array) {
        return sum(fields
            .iter()
            .map(|field| capacity(&field["type"], bound(field)?)));
    }
    if let Some(branches) = schema.get("union").and_then(Value::as_array) {
        return branches
            .iter()
            .map(|branch| capacity(branch, bytes))
            .try_fold(0, |most, n| Ok(most.max(n?)));
    }
    if let Some(items) = schema.get("tuple").and_then(Value::as_array) {
        return sum(items.iter().map(|item| capacity(item, None)));
    }
    // An unbounded collection has no fixed cardinality to sum. Preserve its ordinary
    // aggregate allowance instead of multiplying a per-element bound by an invented count.
    if let Some(item) = schema.get("list") {
        return Ok(if capacity(item, None)? > 0 {
            DEFAULT_BYTES
        } else {
            0
        });
    }
    if let Some(mapping) = schema.get("map") {
        return Ok(if capacity(&mapping["value"], None)? > 0 {
            DEFAULT_BYTES
        } else {
            0
        });
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn call(result: Value) -> Value {
        json!({"entrypoints":[{"name":"run","result":result}]})
    }
    #[test]
    fn actual_fixed_slots_sum_without_clamping_explicit_bounds() {
        let result = json!({"fields":[
            {"name":"trace","type":{"input":"tree"},"asset_bound":{"max_bytes":64u64<<30}},
            {"name":"video","type":{"asset":"video"}},
            {"name":"summary","type":{"asset":"file"},"asset_bound":{"max_bytes":64u64<<20}}
        ]});
        assert_eq!(
            for_callable(&call(result), "entrypoints", "run").unwrap(),
            (64u64 << 30) + (320 << 20)
        );
        assert!(for_callable(&call(json!({})), "entrypoints", "missing").is_err());
    }
    #[test]
    fn nested_optional_and_weights_use_their_actual_shapes() {
        let result = json!({"fields":[{"name":"nested","type":{"fields":[
            {"name":"tree","type":{"union":["null",{"input":"tree"}]},"asset_bound":{"max_bytes":2u64<<30}},
            {"name":"weights","type":{"input":"model"}}
        ]}}]});
        assert_eq!(
            for_callable(&call(result), "entrypoints", "run").unwrap(),
            2 << 30
        );
        assert_eq!(
            for_callable(&call(json!({"fields":[]})), "entrypoints", "run").unwrap(),
            DEFAULT_BYTES
        );
        assert_eq!(
            capacity(&json!({"list":{"asset":"image"}}), None).unwrap(),
            DEFAULT_BYTES
        );
    }
    #[test]
    fn malformed_and_overflowing_bounds_refuse_without_wrapping() {
        for bad in [json!(-1), json!(true), json!("100"), json!(0)] {
            assert!(
                capacity(
                    &json!({"fields":[{"type":{"asset":"file"},"asset_bound":{"max_bytes":bad}}]}),
                    None
                )
                .is_err()
            );
        }
        assert!(
            capacity(
                &json!({"fields":[
                    {"type":{"asset":"file"},"asset_bound":{"max_bytes":u64::MAX}},
                    {"type":{"asset":"file"},"asset_bound":{"max_bytes":1}}
                ]}),
                None
            )
            .is_err()
        );
    }
}
