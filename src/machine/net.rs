//! Outbound TCP that never waits on an unroutable address family.
use std::{io, net::SocketAddr};
use tokio::{net::TcpStream, task::JoinSet};

/// Connects to host:port over whichever of its addresses answers first. A host whose IPv6
/// is unroutable otherwise stalls each connect for the kernel's SYN retries (about two
/// minutes) before trying IPv4.
pub async fn connect(host: &str, port: u16) -> io::Result<TcpStream> {
    connect_any(tokio::net::lookup_host((host, port)).await?).await
}

async fn connect_any(addresses: impl IntoIterator<Item = SocketAddr>) -> io::Result<TcpStream> {
    let mut attempts = JoinSet::new();
    for address in addresses {
        attempts.spawn(TcpStream::connect(address));
    }
    let mut last = io::Error::new(io::ErrorKind::NotFound, "the host has no address");
    while let Some(attempt) = attempts.join_next().await {
        match attempt.map_err(io::Error::other)? {
            Ok(stream) => return Ok(stream),
            Err(error) => last = error,
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_address_that_never_answers_does_not_hold_the_connect() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let open = listener.local_addr().unwrap();
        // TEST-NET-1 is never routed: a SYN to it is dropped, as on a host without IPv6.
        let silent: SocketAddr = format!("192.0.2.1:{}", open.port()).parse().unwrap();
        let stream = connect_any([silent, open]).await.unwrap();
        assert_eq!(stream.peer_addr().unwrap(), open);
    }
}
