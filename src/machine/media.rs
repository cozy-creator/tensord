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

fn mapped(port: u16, env: &HashMap<String, String>) -> io::Result<Option<String>> {
    if let Some(address) = env.get("COZY_WEBRTC_PUBLIC_ADDRESS") {
        let address = address.parse::<SocketAddr>().map_err(|_| {
            io::Error::other("COZY_WEBRTC_PUBLIC_ADDRESS must be an IPv4 host:port")
        })?;
        if !address.is_ipv4() || address.port() == 0 {
            return Err(io::Error::other(
                "COZY_WEBRTC_PUBLIC_ADDRESS must name a nonzero IPv4 port",
            ));
        }
        return Ok(Some(address.to_string()));
    }
    for (host, key) in [
        ("RUNPOD_PUBLIC_IP", format!("RUNPOD_TCP_PORT_{port}")),
        ("PUBLIC_IPADDR", format!("VAST_TCP_PORT_{port}")),
    ] {
        let Some(external_port) = env.get(&key) else {
            continue;
        };
        let host = env
            .get(host)
            .ok_or_else(|| io::Error::other(format!("{key} requires its provider public IP")))?
            .parse::<IpAddr>()
            .map_err(|_| io::Error::other(format!("{key} requires a valid provider public IP")))?;
        let port = external_port
            .parse::<u16>()
            .map_err(|_| io::Error::other(format!("{key} must be a nonzero IPv4 port")))?;
        if !host.is_ipv4() || port == 0 {
            return Err(io::Error::other(format!(
                "{key} must name a nonzero port with an IPv4 provider public IP"
            )));
        }
        return Ok(Some(SocketAddr::new(host, port).to_string()));
    }
    Ok(None)
}
fn addresses(port: u16) -> io::Result<Vec<String>> {
    let env = std::env::vars().collect();
    if let Some(mapped) = mapped(port, &env)? {
        return Ok(vec![mapped]);
    }
    let mut addresses = vec![];
    if let Ok(interfaces) = nix::ifaddrs::getifaddrs() {
        for interface in interfaces {
            let Some(address) = interface.address else {
                continue;
            };
            let ip = address.as_sockaddr_in().map(|v4| IpAddr::V4(v4.ip()));
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
    Ok(addresses)
}

pub fn descriptor(identity: &MachineIdentity) -> Option<v1::WebRtc> {
    let attested = identity.readiness.payload()?;
    let value: serde_json::Value = serde_json::from_slice(&attested).ok()?;
    let port = u16::try_from(value["webrtc"]["port"].as_u64()?).ok()?;
    let (addresses, unavailable_reason) = match addresses(port) {
        Ok(addresses) => (addresses, String::new()),
        Err(error) => (vec![], error.to_string()),
    };
    Some(v1::WebRtc {
        port: port as u32,
        addresses,
        unavailable_reason,
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
        assert_eq!(
            mapped(8085, &env).unwrap().as_deref(),
            Some("203.0.113.7:30001")
        );
        assert!(mapped(8086, &env).unwrap().is_none());
        for (host, port) in [
            ("invalid", "30001"),
            ("203.0.113.7", "invalid"),
            ("203.0.113.7", "0"),
            ("::1", "30001"),
        ] {
            let invalid = HashMap::from([
                ("PUBLIC_IPADDR".into(), host.into()),
                ("VAST_TCP_PORT_8085".into(), port.into()),
            ]);
            assert!(mapped(8085, &invalid).is_err());
        }
        assert!(mapped(
            8085,
            &HashMap::from([("VAST_TCP_PORT_8085".into(), "30001".into())])
        )
        .is_err());
    }
    #[test]
    fn advertised_ipv4_addresses_accept_real_connections_and_bad_config_is_explicit() {
        let root = std::env::temp_dir().join(format!("cm-media-connect-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let listener = listen(&root, None, true).unwrap().unwrap();
        for address in addresses(listener.local_addr().unwrap().port()).unwrap() {
            let address = address.parse::<SocketAddr>().unwrap();
            assert!(address.is_ipv4());
            assert!(std::net::TcpStream::connect_timeout(
                &address,
                std::time::Duration::from_secs(2)
            )
            .is_ok());
        }
        let env = HashMap::from([("COZY_WEBRTC_PUBLIC_ADDRESS".into(), "invalid".into())]);
        assert!(mapped(8085, &env).is_err());
        let signer = ed25519_dalek::SigningKey::from_bytes(&[91; 32]);
        let identity = MachineIdentity::ephemeral(
            "descriptor".into(),
            vec![signer.verifying_key()],
            vec![7; 32],
        )
        .unwrap();
        identity.readiness.seal(serde_json::to_vec(&serde_json::json!({"boot_id":identity.authority.boot_id,"webrtc":{"port":listener.local_addr().unwrap().port()}})).unwrap()).unwrap();
        let descriptor = descriptor(&identity).unwrap();
        assert_eq!(
            descriptor.port,
            listener.local_addr().unwrap().port() as u32
        );
        assert_eq!(
            descriptor.fingerprint,
            tensorfs_core::sha256::hex_digest(&identity.cert_der)
        );
        fs::remove_dir_all(root).unwrap();
    }
}
