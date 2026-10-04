//! The machine's direct player endpoint: the WebRTC listener a browser reaches with its play
//! link, and the addresses Status reports for it. No relay: a viewer behind NAT is refused.
use crate::api::v1;
use std::{
    collections::HashMap,
    fs, io,
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener},
    path::Path,
};

/// Where a browser reaches the listener: this computer's own interfaces, a provider's (or the
/// launcher's) mapping of its port, or a mapping that could not be read, which stops playback
/// alone.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Reach {
    #[default]
    Interfaces,
    Mapped(SocketAddr),
    Invalid(String),
}

/// The listener on every interface: its capability is bound to the viewer's DTLS certificate,
/// while the control API stays where the grant puts it. A rental listens on its granted port;
/// this computer's machine keeps the port it first took, so a play link survives restarts.
pub fn listen(state: &Path, granted: Option<u16>, persistent: bool) -> io::Result<Option<TcpListener>> {
    let bind = |port| TcpListener::bind((Ipv4Addr::UNSPECIFIED, port));
    if let Some(port) = granted {
        return bind(port).map(Some);
    }
    if !persistent {
        return Ok(None);
    }
    let saved = state.join("webrtc-port");
    let kept = fs::read_to_string(&saved).ok().and_then(|text| text.trim().parse::<u16>().ok());
    let listener = match kept.map(bind) {
        Some(Ok(listener)) => listener,
        _ => bind(0)?,
    };
    if kept != Some(listener.local_addr()?.port()) {
        super::identity::write_atomic(&saved, listener.local_addr()?.port().to_string().as_bytes(), 0o600)?;
    }
    Ok(Some(listener))
}

/// The launch environment's mapping of a granted port: `COZY_WEBRTC_PUBLIC_ADDRESS`, else
/// RunPod's or vast.ai's public IP and external port for it.
pub fn reach(granted: Option<u16>, env: &HashMap<String, String>) -> Reach {
    let ipv4 = |address: SocketAddr, what: &str| match address.is_ipv4() && address.port() != 0 {
        true => Reach::Mapped(address),
        false => Reach::Invalid(format!("{what} must name an IPv4 address and a nonzero port")),
    };
    if let Some(address) = env.get("COZY_WEBRTC_PUBLIC_ADDRESS") {
        return match address.parse() {
            Ok(address) => ipv4(address, "COZY_WEBRTC_PUBLIC_ADDRESS"),
            Err(_) => Reach::Invalid("COZY_WEBRTC_PUBLIC_ADDRESS must be host:port".into()),
        };
    }
    let Some(port) = granted else { return Reach::Interfaces };
    for (host, mapped) in [("RUNPOD_PUBLIC_IP", "RUNPOD_TCP_PORT"), ("PUBLIC_IPADDR", "VAST_TCP_PORT")] {
        let mapped = format!("{mapped}_{port}");
        let Some(external) = env.get(&mapped) else { continue };
        return match (env.get(host).map(|ip| ip.parse::<IpAddr>()), external.parse::<u16>()) {
            (Some(Ok(ip)), Ok(external)) => ipv4(SocketAddr::new(ip, external), &mapped),
            _ => Reach::Invalid(format!("{mapped} needs a port and {host} an IP address")),
        };
    }
    Reach::Interfaces
}

/// Status's player endpoint: the listener's port, where it is reached, and the fingerprint of
/// the certificate its DTLS presents.
pub fn endpoint(port: u16, reach: &Reach, cert_der: &[u8]) -> v1::WebRtc {
    let (addresses, unavailable_reason) = match reach {
        Reach::Mapped(address) => (vec![address.to_string()], String::new()),
        Reach::Invalid(reason) => (vec![], reason.clone()),
        Reach::Interfaces => (interfaces(port), String::new()),
    };
    v1::WebRtc {
        port: port.into(),
        addresses,
        fingerprint: tensorfs_core::sha256::hex_digest(cert_der),
        unavailable_reason,
    }
}

/// This computer's IPv4 addresses (LAN, Tailscale; loopback last) at `port`.
fn interfaces(port: u16) -> Vec<String> {
    let mut found: Vec<Ipv4Addr> = nix::ifaddrs::getifaddrs()
        .into_iter()
        .flatten()
        .filter_map(|interface| interface.address?.as_sockaddr_in().map(|v4| v4.ip()))
        .filter(|ip| !ip.is_unspecified() && !ip.is_multicast())
        .collect();
    found.sort_by_key(|ip| (ip.is_loopback(), *ip));
    found.dedup();
    found.into_iter().map(|ip| SocketAddr::new(ip.into(), port).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This computer's machine keeps its port across restarts, and takes another if that one
    /// is now in use; every address it reports accepts a connection.
    #[test]
    fn a_persistent_machine_keeps_its_port_and_reports_reachable_addresses() {
        let root = std::env::temp_dir().join(format!("cm-player-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let first = listen(&root, None, true).unwrap().unwrap();
        let port = first.local_addr().unwrap().port();
        let taken = listen(&root, None, true).unwrap().unwrap();
        assert_ne!(taken.local_addr().unwrap().port(), port);
        drop((first, taken));
        let again = listen(&root, None, true).unwrap().unwrap();
        let port = again.local_addr().unwrap().port();
        assert_eq!(listen(&root, None, false).unwrap().map(|l| l.local_addr().unwrap()), None);
        let reported = endpoint(port, &Reach::Interfaces, b"leaf");
        assert!(reported.addresses.iter().any(|a| a == &format!("127.0.0.1:{port}")));
        for address in &reported.addresses {
            std::net::TcpStream::connect(address).unwrap();
        }
        assert_eq!(reported.fingerprint, tensorfs_core::sha256::hex_digest(b"leaf"));
        fs::remove_dir_all(root).unwrap();
    }

    /// A provider's mapping is read from its own names for the granted port; a malformed one
    /// is reported, never guessed around.
    #[test]
    fn a_rental_reports_its_provider_mapping() {
        let env = |pairs: &[(&str, &str)]| -> HashMap<String, String> {
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
        };
        let vast = env(&[("PUBLIC_IPADDR", "203.0.113.7"), ("VAST_TCP_PORT_8085", "30001")]);
        assert_eq!(reach(Some(8085), &vast), Reach::Mapped("203.0.113.7:30001".parse().unwrap()));
        assert_eq!(reach(Some(8086), &vast), Reach::Interfaces);
        let runpod = env(&[("RUNPOD_PUBLIC_IP", "198.51.100.2"), ("RUNPOD_TCP_PORT_8085", "40002")]);
        assert_eq!(reach(Some(8085), &runpod), Reach::Mapped("198.51.100.2:40002".parse().unwrap()));
        let set = env(&[("COZY_WEBRTC_PUBLIC_ADDRESS", "192.0.2.9:7000"), ("RUNPOD_TCP_PORT_8085", "1")]);
        assert_eq!(reach(None, &set), Reach::Mapped("192.0.2.9:7000".parse().unwrap()));
        for invalid in [
            env(&[("VAST_TCP_PORT_8085", "30001")]),
            env(&[("PUBLIC_IPADDR", "203.0.113.7"), ("VAST_TCP_PORT_8085", "0")]),
            env(&[("PUBLIC_IPADDR", "::1"), ("VAST_TCP_PORT_8085", "30001")]),
            env(&[("COZY_WEBRTC_PUBLIC_ADDRESS", "invalid")]),
        ] {
            let Reach::Invalid(reason) = reach(Some(8085), &invalid) else { panic!("{invalid:?}") };
            assert!(endpoint(8085, &Reach::Invalid(reason), b"leaf").addresses.is_empty());
        }
    }
}
