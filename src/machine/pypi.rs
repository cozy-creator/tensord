//! A published Runtime or TensorFS wheel for this platform, fetched from PyPI over HTTPS and
//! checked against the index's own SHA-256.
use bytes::Bytes;
use http_body_util::{BodyExt, Empty, Limited};
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio_rustls::rustls::{self, pki_types::ServerName, RootCertStore};

const MAX_WHEEL_BYTES: usize = 256 << 20;

pub fn wheel(distribution: &str, version: &str, dir: &Path) -> io::Result<PathBuf> {
    let project = distribution.replace('_', "-");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let index = get(
            &format!("https://pypi.org/pypi/{project}/{version}/json"),
            32 << 20,
        )
        .await?;
        #[derive(serde::Deserialize)]
        struct File {
            filename: String,
            url: String,
            digests: Digests,
        }
        #[derive(serde::Deserialize)]
        struct Digests {
            sha256: String,
        }
        #[derive(serde::Deserialize)]
        struct Release {
            urls: Vec<File>,
        }
        let release: Release = serde_json::from_slice(&index).map_err(io::Error::other)?;
        let arch = std::env::consts::ARCH;
        let file = release
            .urls
            .into_iter()
            .find(|f| {
                f.filename.ends_with(".whl")
                    && f.filename.contains("manylinux")
                    && f.filename.contains(arch)
            })
            .ok_or_else(|| {
                io::Error::other(format!(
                    "PyPI has no linux {arch} wheel of {project} {version}"
                ))
            })?;
        let body = get(&file.url, MAX_WHEEL_BYTES).await?;
        let digest: String = Sha256::digest(&body)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if digest != file.digests.sha256 || file.filename.contains('/') {
            return Err(io::Error::other(format!(
                "{} does not match PyPI's digest",
                file.filename
            )));
        }
        fs::create_dir_all(dir)?;
        let path = dir.join(&file.filename);
        super::identity::write_atomic(&path, &body, 0o644)?;
        Ok(path)
    })
}

async fn get(url: &str, max: usize) -> io::Result<Bytes> {
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| io::Error::other("only HTTPS downloads"))?;
    let (host, path) = rest
        .split_once('/')
        .map(|(h, p)| (h, format!("/{p}")))
        .unwrap_or((rest, "/".into()));
    let roots = RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(io::Error::other)?
    .with_root_certificates(roots)
    .with_no_client_auth();
    let tcp = tokio::net::TcpStream::connect((host, 443)).await?;
    let name = ServerName::try_from(host.to_owned()).map_err(io::Error::other)?;
    let stream = tokio_rustls::TlsConnector::from(Arc::new(tls))
        .connect(name, tcp)
        .await?;
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(io::Error::other)?;
    tokio::spawn(connection);
    let request = Request::get(path)
        .header("host", host)
        .header("user-agent", "cozy-machine")
        .body(Empty::<Bytes>::new())
        .map_err(io::Error::other)?;
    let response = sender
        .send_request(request)
        .await
        .map_err(io::Error::other)?;
    if response.status() != StatusCode::OK {
        return Err(io::Error::other(format!(
            "{url} answered HTTP {}",
            response.status()
        )));
    }
    Ok(Limited::new(response.into_body(), max)
        .collect()
        .await
        .map_err(io::Error::other)?
        .to_bytes())
}
