//! The machine owns its media listener and reports direct reachable endpoints.
use crate::api::{v1, MachineIdentity};
use std::{
    collections::HashMap,
    fs, io,
    net::{IpAddr, SocketAddr, TcpListener},
    path::Path,
};

pub fn listen(
    state: &Path,
    configured: Option<u16>,
    persistent: bool,
) -> io::Result<Option<TcpListener>> {
    if configured.is_none() && !persistent {
        return Ok(None);
    }
    let saved = state.join("webrtc-port");
    let port = match configured {
        Some(port) => port,
        None => match fs::read_to_string(&saved) {
            Ok(value) => value
                .trim()
                .parse::<u16>()
                .map_err(|_| io::Error::other("the persisted WebRTC port is unreadable"))?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error),
        },
    };
    // Media grants carry DTLS-bound capabilities. Its listener is separate from the
    // personal control API, which remains on loopback.
    let listener = TcpListener::bind((std::net::Ipv4Addr::UNSPECIFIED, port))?;
    if persistent {
        super::identity::write_atomic(
            &saved,
            listener.local_addr()?.port().to_string().as_bytes(),
            0o600,
        )?;
    }
    Ok(Some(listener))
}

fn mapped(port: u16, env: &HashMap<String, String>) -> Option<String> {
    if let Some(address) = env.get("COZY_WEBRTC_PUBLIC_ADDRESS") {
        return address
            .parse::<SocketAddr>()
            .ok()
            .map(|address| address.to_string());
    }
    for (host, key) in [
        ("RUNPOD_PUBLIC_IP", format!("RUNPOD_TCP_PORT_{port}")),
        ("PUBLIC_IPADDR", format!("VAST_TCP_PORT_{port}")),
    ] {
        if let (Some(host), Some(port)) = (env.get(host), env.get(&key)) {
            if let (Ok(host), Ok(port)) = (host.parse::<IpAddr>(), port.parse::<u16>()) {
                if port > 0 {
                    return Some(SocketAddr::new(host, port).to_string());
                }
            }
        }
    }
    None
}
fn addresses(port: u16) -> Vec<String> {
    let env = std::env::vars().collect();
    if let Some(mapped) = mapped(port, &env) {
        return vec![mapped];
    }
    let mut addresses = vec![];
    if let Ok(interfaces) = nix::ifaddrs::getifaddrs() {
        for interface in interfaces {
            let Some(address) = interface.address else {
                continue;
            };
            let ip = if let Some(v4) = address.as_sockaddr_in() {
                Some(IpAddr::V4(v4.ip()))
            } else {
                address.as_sockaddr_in6().map(|v6| IpAddr::V6(v6.ip()))
            };
            if let Some(ip) = ip {
                if !ip.is_unspecified()
                    && !ip.is_multicast()
                    && !matches!(ip,IpAddr::V6(v6) if v6.is_unicast_link_local())
                {
                    addresses.push(SocketAddr::new(ip, port).to_string());
                }
            }
        }
    }
    addresses.sort_by_key(|address| {
        (
            address.parse::<SocketAddr>().unwrap().ip().is_loopback(),
            address.clone(),
        )
    });
    addresses.dedup();
    addresses
}

pub fn descriptor(identity: &MachineIdentity) -> Option<v1::WebRtc> {
    let attested = identity.readiness.attested()?;
    let value: serde_json::Value = serde_json::from_slice(&attested).ok()?;
    let port = u16::try_from(value["webrtc"]["port"].as_u64()?).ok()?;
    Some(v1::WebRtc {
        port: port as u32,
        addresses: addresses(port),
        fingerprint: tensorfs_core::sha256::hex_digest(&identity.cert_der),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persistent_machine_selects_and_reuses_its_media_port() {
        let root = std::env::temp_dir().join(format!("cm-media-port-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let first = listen(&root, None, true).unwrap().unwrap();
        let port = first.local_addr().unwrap().port();
        assert!(port > 0);
        drop(first);
        assert_eq!(
            listen(&root, None, true)
                .unwrap()
                .unwrap()
                .local_addr()
                .unwrap()
                .port(),
            port
        );
        assert!(listen(&root, None, false).unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn provider_mapping_is_config_not_an_inferred_internal_port() {
        let env = HashMap::from([
            ("PUBLIC_IPADDR".into(), "203.0.113.7".into()),
            ("VAST_TCP_PORT_8085".into(), "30001".into()),
        ]);
        assert_eq!(mapped(8085, &env).as_deref(), Some("203.0.113.7:30001"));
        assert!(mapped(8086, &env).is_none());
    }
}
