//! Parse harmless JSON formatting variations without accepting ambiguous duplicate keys.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use std::fmt;

pub(crate) fn parse(bytes: &[u8]) -> Result<Value, serde_json::Error> {
    serde_json::from_slice::<Unique>(bytes).map(|value| value.0)
}

/// Stable application JSON for intent and results. Keep exact integer precision and number
/// kind: JCS converts every integer to f64 and can merge distinct valid seeds. Object key
/// order and spelling/whitespace are not semantics; arrays and typed numbers are.
pub(crate) fn intent_bytes(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    fn ordered(value: &Value) -> Value {
        match value {
            Value::Object(fields) => {
                let mut fields: Vec<_> = fields.iter().collect();
                fields.sort_by(|(a, _), (b, _)| a.cmp(b));
                Value::Object(
                    fields
                        .into_iter()
                        .map(|(key, value)| (key.clone(), ordered(value)))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(ordered).collect()),
            value => value.clone(),
        }
    }
    serde_json::to_vec(&ordered(value))
}

struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(Json).map(Unique)
    }
}
struct Json;
impl<'de> Visitor<'de> for Json {
    type Value = Value;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys and finite numbers")
    }
    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite JSON number"))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }
    fn visit_string<E: de::Error>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(Unique(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            let Unique(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_identity_tolerates_older_numeric_and_formatting_forms() {
        let old = parse(br#" { "samples": [[6.0, -0.0]], "seed": 19 } "#).unwrap();
        let current = parse(br#"{"seed":19,"samples":[[6,0]]}"#).unwrap();
        assert_eq!(
            serde_json_canonicalizer::to_vec(&old).unwrap(),
            serde_json_canonicalizer::to_vec(&current).unwrap()
        );
        assert!(parse(br#"{"samples":[{"seed":1,"seed":2}]}"#).is_err());
        assert!(parse(br#"{"seed":19} {}"#).is_err());
        assert!(parse(br#"{"seed":1e999}"#).is_err());
    }

    #[test]
    fn intent_preserves_large_integers_and_number_kind_while_ignoring_formatting() {
        let first = parse(br#" { "nested":{"z":2,"a":1}, "seed":9007199254740993 } "#).unwrap();
        let formatted = parse(br#"{"seed":9007199254740993,"nested":{"a":1,"z":2}}"#).unwrap();
        let different = parse(br#"{"seed":9007199254740992,"nested":{"a":1,"z":2}}"#).unwrap();
        assert_eq!(
            intent_bytes(&first).unwrap(),
            intent_bytes(&formatted).unwrap()
        );
        assert_ne!(
            intent_bytes(&first).unwrap(),
            intent_bytes(&different).unwrap()
        );
        assert_ne!(
            intent_bytes(&parse(b"1").unwrap()).unwrap(),
            intent_bytes(&parse(b"1.0").unwrap()).unwrap()
        );
        assert_eq!(
            intent_bytes(&parse(b"18446744073709551615").unwrap()).unwrap(),
            b"18446744073709551615"
        );
    }
}
