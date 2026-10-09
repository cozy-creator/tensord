//! The pod's public endpoint as its provider put it in the environment, told to the Hub once
//! the API listens. A provider reports a pod's ports 22-55 s after its machine listens
//! (2026-10-09); the Hub probes this report at once. The provider's view stays its fallback,
//! and a Hub that does not take reports answers 404.
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Endpoint {
    pub public_host: String,
    /// Private TCP port to its public port, as the provider mapped them.
    pub ports: BTreeMap<String, u16>,
}

/// RunPod (`RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_<port>`) or vast.ai (`PUBLIC_IPADDR`,
/// `VAST_TCP_PORT_<port>`); None anywhere else.
pub fn from_env(vars: impl IntoIterator<Item = (String, String)>) -> Option<Endpoint> {
    let vars: BTreeMap<String, String> = vars.into_iter().collect();
    [("RUNPOD_PUBLIC_IP", "RUNPOD_TCP_PORT_"), ("PUBLIC_IPADDR", "VAST_TCP_PORT_")]
        .into_iter()
        .find_map(|(host, prefix)| {
            let host = vars.get(host)?.trim();
            host.parse::<std::net::IpAddr>().ok()?;
            let ports: BTreeMap<String, u16> = vars
                .iter()
                .filter_map(|(name, value)| {
                    let private = name.strip_prefix(prefix)?.parse::<u16>().ok()?;
                    let public = value.trim().parse::<u16>().ok()?;
                    (private > 0 && public > 0).then(|| (private.to_string(), public))
                })
                .collect();
            (!ports.is_empty()).then(|| Endpoint { public_host: host.to_string(), ports })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn runpod_and_vast_name_their_endpoint() {
        // The environment of rental nanoha (RunPod EU-RO-1), 2026-10-09.
        let runpod = from_env(vars(&[
            ("RUNPOD_PUBLIC_IP", "213.173.105.223"),
            ("RUNPOD_TCP_PORT_22", "43566"),
            ("RUNPOD_TCP_PORT_8443", "43565"),
            ("RUNPOD_TCP_PORT_8445", "43567"),
            ("RUNPOD_POD_ID", "320hr6lbxdjnlq"),
        ]))
        .unwrap();
        assert_eq!(runpod.public_host, "213.173.105.223");
        assert_eq!(runpod.ports.get("8443"), Some(&43565));
        assert_eq!(runpod.ports.len(), 3);
        let vast = from_env(vars(&[("PUBLIC_IPADDR", "1.2.3.4"), ("VAST_TCP_PORT_8443", "40001")])).unwrap();
        assert_eq!(vast.ports.get("8443"), Some(&40001));
    }

    #[test]
    fn no_provider_or_no_ports_reports_nothing() {
        assert_eq!(from_env(vars(&[("HOME", "/root")])), None);
        assert_eq!(from_env(vars(&[("RUNPOD_PUBLIC_IP", "213.173.105.223")])), None);
        assert_eq!(from_env(vars(&[("RUNPOD_PUBLIC_IP", "not-an-ip"), ("RUNPOD_TCP_PORT_8443", "1")])), None);
    }
}
