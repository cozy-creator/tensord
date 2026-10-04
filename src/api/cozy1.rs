//! `cozy/1` (G/API.md): a run's outputs, live or finished, to a browser over WebRTC-direct. The
//! WebRTC port serves ICE-lite with passive ICE-TCP (RFC 4571 frames); DTLS presents the
//! machine's leaf, whose fingerprint the page pins; one data channel, `cozy` with protocol
//! `cozy/1`, carries JSON requests and binary `[u32 stream][u64 offset][payload]` frames. The
//! client chooses its ICE credential (`cozy+webrtc+v1/…`, ufrag = pwd) and synthesizes this
//! end's SDP, so nothing here parses SDP. str0m does ICE, DTLS and SCTP; this module frames TCP
//! and speaks cozy/1, on the same wire as the Go agent's, so one player reaches either.
use super::{
    auth::{Lapse, StreamAuthority, VerifiedActor},
    backend::{MachineBackend, OutputSnapshot},
    capability::{self, Grant},
    machine_v1::{query, stream_run},
    v1, MachineIdentity,
};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    future::Future,
    io,
    net::{IpAddr, SocketAddr},
    os::unix::fs::FileExt,
    pin::Pin,
    sync::{Arc, Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use str0m::{
    channel::ChannelId,
    config::{CryptoProvider, DtlsCert, Fingerprint},
    net::{Protocol, Receive, TcpType},
    Candidate, Event, IceCreds, Input, Output, Rtc,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{tcp::OwnedReadHalf, TcpListener, TcpStream},
    sync::mpsc,
};

/// The ICE credential every client chooses starts with this; its ufrag is its password.
pub const UFRAG_PREFIX: &str = "cozy+webrtc+v1/";
const HEADER: usize = 12;
const MAX_MESSAGE: usize = 64 << 10;
const MAX_HELLO: usize = 8 << 10;
const MAX_STREAMS: usize = 8;
// Only unauthenticated handshakes are bounded by capacity. Authenticated viewers have no
// viewer cap; each session still bounds its own buffered output and open requests.
const MAX_CONNS: usize = 64;
const MAX_CONNS_PER_IP: usize = 8;
/// Ended ufrags kept so a reconnect cannot resume a session no state remains for.
const ENDED_UFRAGS: usize = 4096;
const MIN_WINDOW: usize = 256 << 10;
const MAX_WINDOW: usize = 16 << 20;

/// One machine's cozy/1 listener state.
pub(super) struct Media<B> {
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
    cert: DtlsCert,
    crypto: Arc<CryptoProvider>,
    admitted: Mutex<Admitted>,
}

#[derive(Default)]
struct Admitted {
    /// In arrival order.
    seats: Vec<Arc<Seat>>,
    /// Connections per ufrag.
    live: HashMap<String, usize>,
    ended: VecDeque<String>,
}

/// One connection's place among the machine's sessions.
struct Seat {
    ip: IpAddr,
    /// The key that opened it, once its hello verified.
    key: Mutex<Option<String>>,
    task: OnceLock<tokio::task::AbortHandle>,
}

impl<B: MachineBackend> Media<B> {
    pub(super) fn new(identity: Arc<MachineIdentity>, backend: Arc<B>) -> io::Result<Arc<Self>> {
        let invalid = |what: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the machine leaf's {what} is unreadable"),
            )
        };
        let certificate = rustls_pemfile::certs(&mut identity.cert_pem.as_bytes())
            .next()
            .ok_or_else(|| invalid("certificate"))??
            .to_vec();
        let private_key = rustls_pemfile::private_key(&mut identity.key_pem.as_bytes())?
            .ok_or_else(|| invalid("key"))?
            .secret_der()
            .to_vec();
        Ok(Arc::new(Self {
            identity,
            backend,
            cert: DtlsCert {
                certificate,
                private_key,
            },
            crypto: Arc::new(str0m::crypto::from_feature_flags()),
            admitted: Mutex::default(),
        }))
    }

    fn admit(self: &Arc<Self>, ip: IpAddr) -> Option<Admission<B>> {
        let mut admitted = self.admitted.lock().unwrap();
        loop {
            let pending: Vec<&Arc<Seat>> =
                admitted.seats.iter().filter(|s| s.key.lock().unwrap().is_none()).collect();
            let own = pending.iter().filter(|s| s.ip == ip).count();
            if own < MAX_CONNS_PER_IP && pending.len() < MAX_CONNS {
                let seat = Arc::new(Seat { ip, key: Mutex::default(), task: OnceLock::new() });
                admitted.seats.push(seat.clone());
                return Some(Admission { media: self.clone(), seat, ufrag: None });
            }
            let pending = |from: Option<IpAddr>| {
                admitted.seats.iter().position(|s| {
                    s.key.lock().unwrap().is_none() && from.is_none_or(|ip| s.ip == ip)
                })
            };
            let victim = pending(Some(ip)).or_else(|| match own < MAX_CONNS_PER_IP {
                true => pending(None),
                false => None,
            })?;
            if let Some(task) = admitted.seats.remove(victim).task.get() {
                task.abort();
            }
        }
    }
}

/// One admitted connection; dropping it gives its seat and its ufrag back.
struct Admission<B: MachineBackend> {
    media: Arc<Media<B>>,
    seat: Arc<Seat>,
    ufrag: Option<String>,
}
impl<B: MachineBackend> Admission<B> {
    /// A ufrag names one session for its whole life. After a dropped TCP connection a browser
    /// reconnects with the same ufrag and resumes DTLS records no new session could read;
    /// closing that connection unanswered fails its PeerConnection, so the client reconnects
    /// afresh and resumes at its cursor.
    fn claim(&mut self, ufrag: &str) -> bool {
        let mut admitted = self.media.admitted.lock().unwrap();
        if admitted.ended.iter().any(|ended| ended == ufrag) {
            return false;
        }
        *admitted.live.entry(ufrag.into()).or_default() += 1;
        self.ufrag = Some(ufrag.into());
        true
    }

    /// Seats the session under the key that opened it: it no longer counts as a handshake.
    fn authenticate(&self, key: &str) {
        *self.seat.key.lock().unwrap() = Some(key.into());
    }
}
impl<B: MachineBackend> Drop for Admission<B> {
    fn drop(&mut self) {
        let mut admitted = self.media.admitted.lock().unwrap();
        admitted.seats.retain(|s| !Arc::ptr_eq(s, &self.seat));
        if let Some(ufrag) = self.ufrag.take() {
            let live = admitted.live.entry(ufrag.clone()).or_default();
            *live = live.saturating_sub(1);
            if *live == 0 {
                admitted.live.remove(&ufrag);
                if admitted.ended.len() == ENDED_UFRAGS {
                    admitted.ended.pop_front();
                }
                admitted.ended.push_back(ufrag);
            }
        }
    }
}

/// Serves cozy/1 to each connection the WebRTC port accepts.
pub(super) async fn serve<B: MachineBackend>(media: Arc<Media<B>>, listener: TcpListener) {
    loop {
        let Ok((tcp, peer)) = listener.accept().await else {
            // Out of descriptors, say: the listener itself still stands.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            continue;
        };
        let (work_tx, work) = mpsc::unbounded_channel();
        if let Some(admission) = media.admit(peer.ip()) {
            let seat = admission.seat.clone();
            let task = tokio::spawn(async move {
                let _ = session(tcp, peer, admission, work_tx, work).await;
            });
            let _ = seat.task.set(task.abort_handle());
        }
    }
}

fn framing() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "the connection is not ICE-TCP from a cozy client",
    )
}

async fn read_frame(reader: &mut OwnedReadHalf) -> io::Result<Vec<u8>> {
    let length = reader.read_u16().await? as usize;
    if length == 0 {
        return Err(framing());
    }
    let mut frame = vec![0; length];
    reader.read_exact(&mut frame).await?;
    Ok(frame)
}

/// The USERNAME of a STUN Binding request: (this end's ufrag as the client chose it, its own).
fn binding_username(frame: &[u8]) -> Option<(String, String)> {
    if frame.len() < 20 || frame[0..2] != [0x00, 0x01] || frame[4..8] != [0x21, 0x12, 0xa4, 0x42] {
        return None;
    }
    let mut at = 20;
    while at + 4 <= frame.len() {
        let kind = u16::from_be_bytes([frame[at], frame[at + 1]]);
        let length = u16::from_be_bytes([frame[at + 2], frame[at + 3]]) as usize;
        let value = frame.get(at + 4..at + 4 + length)?;
        if kind == 0x0006 {
            let (local, remote) = std::str::from_utf8(value).ok()?.split_once(':')?;
            return local
                .starts_with(UFRAG_PREFIX)
                .then(|| (local.into(), remote.into()));
        }
        at += 4 + length.div_ceil(4) * 4;
    }
    None
}

/// A certificate's pin as SDP's a=fingerprint spells it: "sha-256 AB:CD:…".
fn spelled(fingerprint: &Fingerprint) -> String {
    let hex: Vec<String> = fingerprint
        .bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect();
    format!("{} {}", fingerprint.hash_func, hex.join(":"))
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() as i64)
}

async fn session<B: MachineBackend>(
    tcp: TcpStream,
    peer: SocketAddr,
    mut admission: Admission<B>,
    work_tx: mpsc::UnboundedSender<Work>,
    mut work: mpsc::UnboundedReceiver<Work>,
) -> io::Result<()> {
    let media = admission.media.clone();
    tcp.set_nodelay(true)?;
    let local = tcp.local_addr()?;
    let (mut reader, mut writer) = tcp.into_split();
    // A first frame that is not a cozy client's Binding request, or whose ufrag is refused,
    // closes the connection unanswered, before any DTLS state exists.
    let first = read_frame(&mut reader).await?;
    let (ufrag, remote) = binding_username(&first).ok_or_else(framing)?;
    if !admission.claim(&ufrag) {
        return Err(framing());
    }
    let mut rtc = Rtc::builder()
        .set_ice_lite(true)
        .set_local_ice_credentials(IceCreds {
            ufrag: ufrag.clone(),
            pass: ufrag,
        })
        .set_dtls_cert(media.cert.clone())
        .set_crypto_provider(media.crypto.clone())
        .set_fingerprint_verification(false)
        .set_sctp_max_buffered_amount(MAX_WINDOW)
        .build(Instant::now());
    {
        let mut direct = rtc.direct_api();
        direct.set_ice_controlling(false);
        direct.set_remote_ice_credentials(IceCreds {
            ufrag: remote,
            pass: String::new(),
        });
        // The client's certificate is unknown before its handshake; its hello's capability
        // names it, and is checked against the certificate DTLS saw.
        direct.set_remote_fingerprint(Fingerprint {
            hash_func: "sha-256".into(),
            bytes: vec![0; 32],
        });
        // The client is the active end of DTLS and opens the SCTP association and the channel.
        direct.start_dtls(false).map_err(io::Error::other)?;
        direct.start_sctp(false);
    }
    let candidate = |address, kind| {
        Candidate::builder()
            .tcp()
            .host(address)
            .tcptype(kind)
            .build()
            .map_err(io::Error::other)
    };
    rtc.add_local_candidate(candidate(local, TcpType::Passive)?);
    rtc.add_remote_candidate(candidate(peer, TcpType::Active)?);
    let receive = |rtc: &mut Rtc, frame: &[u8]| -> io::Result<()> {
        let contents = frame.try_into().map_err(|_| framing())?;
        let received = Receive {
            proto: Protocol::Tcp,
            source: peer,
            destination: local,
            contents,
        };
        rtc.handle_input(Input::Receive(Instant::now(), received))
            .map_err(io::Error::other)
    };
    receive(&mut rtc, &first)?;
    // Frames arrive on their own task: a read is never cut by the select below.
    let (frames_tx, mut frames) = mpsc::channel::<io::Result<Vec<u8>>>(64);
    tokio::spawn(async move {
        loop {
            let frame = read_frame(&mut reader).await;
            let end = frame.is_err();
            if frames_tx.send(frame).await.is_err() || end {
                return;
            }
        }
    });
    let mut cozy = Cozy::new(media.clone(), work_tx);
    // Set once the hello verifies: the session ends at revocation or the link's expiry.
    let mut authority: Option<StreamAuthority> = None;
    let mut ended: Option<Pin<Box<dyn Future<Output = Lapse> + Send>>> = None;
    loop {
        // Checked before pumping, so sustained output cannot starve a lapse.
        if let Some(Err(lapse)) = authority.as_ref().map(StreamAuthority::check) {
            cozy.bye(lapse.code(), lapse.message());
        }
        let deadline = loop {
            match rtc.poll_output().map_err(io::Error::other)? {
                Output::Timeout(at) => break at,
                Output::Transmit(transmit) => {
                    let length = u16::try_from(transmit.contents.len()).map_err(|_| framing())?;
                    let mut framed = length.to_be_bytes().to_vec();
                    framed.extend_from_slice(&transmit.contents);
                    writer.write_all(&framed).await?;
                }
                Output::Event(event) => {
                    if !cozy.event(&mut rtc, event) {
                        return Ok(());
                    }
                }
            }
        };
        if let (Some((grant, actor)), true) = (&cozy.grant, authority.is_none()) {
            admission.authenticate(&grant.key);
            let keys = media.identity.authority.keys.clone();
            let granted = StreamAuthority::new(keys, *actor, Some(grant.expires));
            ended = Some(Box::pin(granted.clone().ended()));
            authority = Some(granted);
        }
        if cozy.pump(&mut rtc) {
            continue; // what it wrote is transmitted first
        }
        // The bye is the last thing sent; the session ends once the client has it.
        if cozy.done && cozy.flushed(&mut rtc) {
            rtc.disconnect();
            return Ok(());
        }
        let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
        let authority_ended = async {
            match ended.as_mut() {
                Some(ended) => ended.await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            frame = frames.recv() => match frame {
                Some(Ok(frame)) => receive(&mut rtc, &frame)?,
                _ => return Ok(()),
            },
            _ = sleep => rtc.handle_input(Input::Timeout(Instant::now())).map_err(io::Error::other)?,
            Some(done) = work.recv() => cozy.work(done),
            lapse = authority_ended => {
                ended = None;
                cozy.bye(lapse.code(), lapse.message());
            }
        }
    }
}

/// What a stream's task, or the listener, hands the session.
enum Work {
    Push {
        number: u32,
        items: Vec<Item>,
    },
    Stop {
        number: u32,
        message: Value,
    },
}

/// A control message, an output's log entry, or bytes [from, to) of a body.
enum Item {
    Message { message: Value, last: bool },
    Entry(Value),
    Body { body: Arc<Body>, from: u64, to: u64 },
}

/// An output's bytes at one revision: its parts concatenated.
struct Body {
    parts: Vec<(File, u64)>,
}
impl Body {
    fn read_at(&self, mut offset: u64, buffer: &mut [u8]) -> io::Result<()> {
        let mut filled = 0;
        for (file, length) in &self.parts {
            if offset >= *length {
                offset -= length;
                continue;
            }
            let want = ((buffer.len() - filled) as u64).min(length - offset) as usize;
            file.read_exact_at(&mut buffer[filled..filled + want], offset)?;
            (filled, offset) = (filled + want, 0);
            if filled == buffer.len() {
                return Ok(());
            }
        }
        Err(io::ErrorKind::UnexpectedEof.into())
    }
}

struct Stream {
    id: Value,
    number: u32,
    get: bool,
    items: VecDeque<Item>,
    /// Set when the stream ends early: what is sent in place of what it still held.
    stop: Option<Value>,
    task: Option<tokio::task::AbortHandle>,
}

/// The sender's measurement of the path. The channel's unacknowledged bytes stay under half
/// a second of the measured delivery rate: over TCP SCTP sees no loss, and a queue longer
/// than its retransmission timeout would collapse it.
struct Pace {
    written: u64,
    acked: u64,
    at: Instant,
    rates: [f64; 8],
    sample: usize,
}
impl Pace {
    fn window(&mut self, buffered: usize) -> usize {
        let elapsed = self.at.elapsed().as_secs_f64();
        if elapsed >= 0.1 {
            let acked = self.written.saturating_sub(buffered as u64);
            self.rates[self.sample % self.rates.len()] =
                acked.saturating_sub(self.acked) as f64 / elapsed;
            self.sample += 1;
            (self.at, self.acked) = (Instant::now(), acked);
        }
        let rate = self.rates.iter().copied().fold(0.0, f64::max);
        ((rate / 2.0) as usize).clamp(MIN_WINDOW, MAX_WINDOW - MAX_MESSAGE)
    }
}

/// One session's cozy/1 state: pending until its hello verifies, then serving follow and get
/// requests over its one channel.
struct Cozy<B> {
    media: Arc<Media<B>>,
    work: mpsc::UnboundedSender<Work>,
    channel: Option<ChannelId>,
    grant: Option<(Grant, VerifiedActor)>,
    /// Session messages, sent before any stream's: welcome, entry, error, bye.
    out: VecDeque<Value>,
    /// In turn order.
    streams: Vec<Stream>,
    number: u32,
    /// Binary payload the client granted, cumulative; and what was sent of it.
    credit: u64,
    sent: u64,
    pace: Pace,
    /// A bye is queued: nothing else is sent or accepted.
    ending: bool,
    done: bool,
}

impl<B> Drop for Cozy<B> {
    fn drop(&mut self) {
        for stream in &mut self.streams {
            if let Some(task) = stream.task.take() {
                task.abort();
            }
        }
    }
}

fn error(id: Option<&Value>, code: &str, message: &str) -> Value {
    let mut message = json!({"t": "error", "code": code, "message": message});
    if let Some(id) = id {
        message["id"] = id.clone();
    }
    message
}

impl<B: MachineBackend> Cozy<B> {
    fn new(media: Arc<Media<B>>, work: mpsc::UnboundedSender<Work>) -> Self {
        Self {
            media,
            work,
            channel: None,
            grant: None,
            out: VecDeque::new(),
            streams: vec![],
            number: 0,
            credit: 0,
            sent: 0,
            pace: Pace {
                written: 0,
                acked: 0,
                at: Instant::now(),
                rates: [0.0; 8],
                sample: 0,
            },
            ending: false,
            done: false,
        }
    }

    /// Whether everything written was acknowledged, or its channel is gone.
    fn flushed(&self, rtc: &mut Rtc) -> bool {
        match self.channel {
            Some(id) => rtc
                .channel(id)
                .is_none_or(|mut channel| channel.buffered_amount() == 0),
            None => true,
        }
    }

    /// Ends the session once the message is sent.
    fn bye(&mut self, code: &str, message: &str) {
        if !self.ending {
            self.ending = true;
            self.out.clear();
            for stream in self.streams.drain(..) {
                if let Some(task) = stream.task {
                    task.abort();
                }
            }
            self.out
                .push_back(json!({"t": "bye", "code": code, "message": message}));
        }
    }

    /// One str0m event; false ends the session at once.
    fn event(&mut self, rtc: &mut Rtc, event: Event) -> bool {
        match event {
            Event::ChannelOpen(id, label) => {
                let protocol = rtc
                    .channel(id)
                    .and_then(|c| c.config().map(|c| c.protocol.clone()))
                    .unwrap_or_default();
                // A session has one channel, and it speaks cozy/1.
                let first = self.channel.replace(id).is_none();
                first && label == "cozy" && protocol == "cozy/1"
            }
            Event::ChannelData(data) if Some(data.id) == self.channel => {
                if data.data.len() > MAX_MESSAGE {
                    return false;
                }
                if !self.ending {
                    match &self.grant {
                        None => self.hello(rtc, &data.data, !data.binary),
                        Some(_) => self.request(&data.data, !data.binary),
                    }
                }
                true
            }
            Event::ChannelClose(_) => false,
            Event::IceConnectionStateChange(str0m::IceConnectionState::Disconnected) => false,
            _ => true,
        }
    }

    /// Verifies the capability, bound to the certificate DTLS saw; nothing is read, listed or
    /// opened before it does.
    fn hello(&mut self, rtc: &mut Rtc, raw: &[u8], text: bool) {
        let hello: Option<Value> = (text && raw.len() <= MAX_HELLO)
            .then(|| serde_json::from_slice(raw).ok())
            .flatten();
        let Some(token) = hello
            .as_ref()
            .filter(|h| h["t"] == "hello" && h["v"] == 1)
            .and_then(|h| h["cap"].as_str())
        else {
            return self.bye("auth", "the first message must be hello{v: 1, cap}");
        };
        let binding = rtc
            .direct_api()
            .remote_dtls_fingerprint()
            .map(spelled)
            .unwrap_or_default();
        let authority = &self.media.identity.authority;
        let verified = capability::verify_signer(
            token,
            &authority.worker_id,
            &authority.keys.admitted(),
            unix_now(),
            &binding,
        );
        match verified {
            Ok((grant, signer)) => {
                let mut welcome = json!({"t": "welcome", "v": 1, "machine": grant.machine,
                    "run": grant.run, "expires": grant.expires});
                if !grant.outputs.is_empty() {
                    welcome["outputs"] = json!(grant.outputs);
                }
                self.out.push_back(welcome);
                self.grant = Some((
                    grant,
                    VerifiedActor {
                        public_key: signer.to_bytes(),
                    },
                ));
            }
            Err(capability::Refusal::Expired) => {
                self.bye("expired", &capability::Refusal::Expired.to_string())
            }
            Err(refusal) => self.bye("auth", &refusal.to_string()),
        }
    }

    fn request(&mut self, raw: &[u8], text: bool) {
        let Some(request) = text
            .then(|| serde_json::from_slice::<Value>(raw).ok())
            .flatten()
        else {
            return self.out.push_back(error(
                None,
                "bad_request",
                "requests are JSON text messages",
            ));
        };
        let id = request.get("id").cloned().filter(|id| !id.is_null());
        match request["t"].as_str().unwrap_or_default() {
            "credit" => self.credit = self.credit.max(request["bytes"].as_u64().unwrap_or(0)),
            "cancel" => {
                let open = self.streams.iter_mut().find(|s| Some(&s.id) == id.as_ref());
                if let Some(stream) = open {
                    stream
                        .stop
                        .get_or_insert_with(|| json!({"t": "end", "id": stream.id}));
                }
            }
            kind @ ("follow" | "get") => self.open(kind == "get", id, &request),
            other => self.out.push_back(error(
                id.as_ref(),
                "bad_request",
                &format!("unknown message type {other:?}"),
            )),
        }
    }

    fn open(&mut self, get: bool, id: Option<Value>, request: &Value) {
        const SHAPE: &str =
            "a request needs an id, a run, an output, and no negative index, offset or length";
        let (grant, actor) = self.grant.clone().expect("a request follows the hello");
        if unix_now() >= grant.expires {
            return self.bye("expired", &capability::Refusal::Expired.to_string());
        }
        // A present number must be a non-negative integer; an absent one is 0.
        let number = |name: &str| match request.get(name).filter(|v| !v.is_null()) {
            None => Some(0),
            Some(value) => value.as_u64(),
        };
        let run = request["run"].as_str().unwrap_or_default().to_string();
        let output = request["output"].as_str().unwrap_or_default().to_string();
        let numbers = (
            number("index").and_then(|i| u32::try_from(i).ok()),
            number("offset"),
            number("length"),
            number("after"),
        );
        let (Some(id), (Some(index), Some(offset), Some(length), Some(after))) = (id, numbers)
        else {
            return self
                .out
                .push_back(error(request.get("id"), "bad_request", SHAPE));
        };
        if run.is_empty() || output.is_empty() {
            return self.out.push_back(error(Some(&id), "bad_request", SHAPE));
        }
        let item = (index > 0).then_some(index);
        if grant.action != capability::MACHINE && !grant.allows(&run, &output, item) {
            let refused = capability::Refusal::Scope.to_string();
            return self.out.push_back(error(Some(&id), "scope", &refused));
        }
        if self.streams.iter().any(|s| s.id == id) {
            let open = "a request with this id is open";
            return self.out.push_back(error(Some(&id), "bad_request", open));
        }
        if self.streams.len() >= MAX_STREAMS {
            let limit = "a session keeps at most 8 open requests";
            return self.out.push_back(error(Some(&id), "limit", limit));
        }
        self.number += 1;
        let target = Target {
            number: self.number,
            id: id.clone(),
            run,
            output,
            index,
            offset,
            length,
            after,
            etag: request["etag"].as_str().unwrap_or_default().into(),
        };
        let (backend, work) = (self.media.backend.clone(), self.work.clone());
        // Off the session: opening an output may wait on the journal.
        let task = match get {
            true => tokio::spawn(serve_get(backend, actor, target, work)),
            false => tokio::spawn(serve_follow(backend, actor, target, work)),
        };
        self.streams.push(Stream {
            id,
            number: self.number,
            get,
            items: VecDeque::new(),
            stop: None,
            task: Some(task.abort_handle()),
        });
    }

    /// Entries go out as soon as they are journaled, ahead of every stream's bytes, so a
    /// follower learns an output's whole map at once; everything else keeps byte order.
    fn work(&mut self, work: Work) {
        match work {
            Work::Push { number, items } => {
                let open = self
                    .streams
                    .iter_mut()
                    .find(|s| s.number == number && s.stop.is_none());
                if let Some(stream) = open {
                    for item in items {
                        match item {
                            Item::Entry(entry) => self.out.push_back(entry),
                            item => stream.items.push_back(item),
                        }
                    }
                }
            }
            Work::Stop { number, message } => {
                if let Some(stream) = self.streams.iter_mut().find(|s| s.number == number) {
                    stream.stop.get_or_insert(message);
                }
            }
        }
    }

    /// Writes what the channel takes now: session messages, then each stream's control
    /// messages, then data while credit and the window allow, gets before follows, streams of
    /// a class taking turns. True when it wrote anything.
    fn pump(&mut self, rtc: &mut Rtc) -> bool {
        let Some(id) = self.channel else { return false };
        let Some(mut channel) = rtc.channel(id) else {
            return false;
        };
        let mut wrote = false;
        loop {
            // A stopped stream ends with its stop message in place of what it still held.
            self.streams.retain_mut(|stream| match stream.stop.take() {
                Some(stop) => {
                    if let Some(task) = stream.task.take() {
                        task.abort();
                    }
                    self.out.push_back(stop);
                    false
                }
                None => true,
            });
            if let Some(message) = self.out.front() {
                let text = message.to_string();
                if !matches!(channel.write(false, text.as_bytes()), Ok(true)) {
                    return wrote;
                }
                self.pace.written += text.len() as u64;
                self.done = message["t"] == "bye";
                self.out.pop_front();
                if self.done {
                    return true;
                }
                wrote = true;
                continue;
            }
            if self.ending {
                return wrote;
            }
            let control = self
                .streams
                .iter()
                .position(|s| matches!(s.items.front(), Some(Item::Message { .. })));
            if let Some(at) = control {
                let Some(Item::Message { message, last }) = self.streams[at].items.front() else {
                    unreachable!()
                };
                let (text, last) = (message.to_string(), *last);
                if !matches!(channel.write(false, text.as_bytes()), Ok(true)) {
                    return wrote;
                }
                self.pace.written += text.len() as u64;
                self.streams[at].items.pop_front();
                if last {
                    self.streams.remove(at);
                }
                wrote = true;
                continue;
            }
            let buffered = channel.buffered_amount();
            if self.sent >= self.credit || buffered >= self.pace.window(buffered) {
                return wrote;
            }
            let body = |s: &Stream| matches!(s.items.front(), Some(Item::Body { .. }));
            let pick = self
                .streams
                .iter()
                .position(|s| s.get && body(s))
                .or_else(|| self.streams.iter().position(body));
            let Some(at) = pick else { return wrote };
            let mut stream = self.streams.remove(at);
            let Some(Item::Body { body, from, to }) = stream.items.front_mut() else {
                unreachable!()
            };
            let n = (*to - *from)
                .min((MAX_MESSAGE - HEADER) as u64)
                .min(self.credit - self.sent) as usize;
            let mut frame = vec![0; HEADER + n];
            frame[..4].copy_from_slice(&stream.number.to_be_bytes());
            frame[4..HEADER].copy_from_slice(&from.to_be_bytes());
            let read = body.read_at(*from, &mut frame[HEADER..]);
            let accepted = read.is_ok() && matches!(channel.write(true, &frame), Ok(true));
            if accepted {
                *from += n as u64;
                self.sent += n as u64;
                self.pace.written += frame.len() as u64;
                if *from == *to {
                    stream.items.pop_front();
                }
            }
            if read.is_err() {
                let unreadable = "the output's bytes are unreadable";
                stream.stop = Some(error(Some(&stream.id), "unavailable", unreadable));
            }
            // Streams of a class take turns: one that wrote goes to the back.
            match accepted {
                true => self.streams.push(stream),
                false => self.streams.insert(at, stream),
            }
            if !accepted && read.is_ok() {
                return wrote;
            }
            wrote |= accepted;
        }
    }
}

struct Target {
    number: u32,
    id: Value,
    run: String,
    output: String,
    /// A list item's 1-based index; 0 for a single output.
    index: u32,
    offset: u64,
    length: u64,
    after: u64,
    etag: String,
}

fn source_error(id: &Value, status: &tonic::Status) -> Value {
    let code = match status.code() {
        tonic::Code::NotFound => "not_found",
        tonic::Code::PermissionDenied | tonic::Code::Unauthenticated => "scope",
        _ => "unavailable",
    };
    error(Some(id), code, status.message())
}

async fn open_snapshot<B: MachineBackend>(
    backend: &Arc<B>,
    actor: VerifiedActor,
    target: &Target,
) -> Result<OutputSnapshot, tonic::Status> {
    let (backend, run, output) = (backend.clone(), target.run.clone(), target.output.clone());
    let index = (target.index > 0).then_some(target.index);
    tokio::task::spawn_blocking(move || {
        let record = backend.get(actor, query(&*backend, actor, &run)?)?;
        backend.open_output(actor, record.number, &output, index)
    })
    .await
    .map_err(|_| tonic::Status::internal("machine operation stopped"))?
}

/// The `end` of a stream over `snapshot`'s output.
fn end(target: &Target, status: Option<&str>, snapshot: Option<&OutputSnapshot>) -> Item {
    let mut message = json!({"t": "end", "id": target.id});
    if let Some(status) = status {
        message["status"] = status.into();
    }
    if let Some(snapshot) = snapshot {
        message["length"] = snapshot.length.into();
        if let Some(sha256) = &snapshot.sha256 {
            message["sha256"] = sha256.as_str().into();
        }
    }
    Item::Message {
        message,
        last: true,
    }
}

/// One byte range of an output's current bytes; a stale etag answers `changed`.
async fn serve_get<B: MachineBackend>(
    backend: Arc<B>,
    actor: VerifiedActor,
    target: Target,
    work: mpsc::UnboundedSender<Work>,
) {
    let number = target.number;
    let stop = |message| {
        let _ = work.send(Work::Stop { number, message });
    };
    let snapshot = match open_snapshot(&backend, actor, &target).await {
        Ok(snapshot) => snapshot,
        Err(status) => return stop(source_error(&target.id, &status)),
    };
    let etag = format!("r{}", snapshot.rev);
    let total = snapshot.length;
    if !target.etag.is_empty() && target.etag != etag {
        let changed = format!("the output is at {etag}");
        return stop(error(Some(&target.id), "changed", &changed));
    }
    if target.offset > total {
        let past = "the offset is past the output's length";
        return stop(error(Some(&target.id), "bad_request", past));
    }
    let to = match target.length {
        0 => total,
        length => total.min(target.offset.saturating_add(length)),
    };
    let finished = end(&target, None, Some(&snapshot));
    let mut items = vec![Item::Message {
        message: json!({"t": "open", "id": target.id, "stream": number, "offset": target.offset,
            "length": to - target.offset, "etag": etag}),
        last: false,
    }];
    if to > target.offset {
        items.push(Item::Body {
            body: Arc::new(Body {
                parts: snapshot.parts,
            }),
            from: target.offset,
            to,
        });
    }
    items.push(finished);
    let _ = work.send(Work::Push { number, items });
}

/// Streams an output from the client's cursor (`after`, `offset`) as the run's log announces
/// its revisions, until the run's end. Bytes are never read past the length the latest
/// revision committed.
async fn serve_follow<B: MachineBackend>(
    backend: Arc<B>,
    actor: VerifiedActor,
    target: Target,
    work: mpsc::UnboundedSender<Work>,
) {
    let number = target.number;
    let stop = |message| {
        let _ = work.send(Work::Stop { number, message });
    };
    let (events_tx, mut events) = mpsc::channel(256);
    let request = v1::RunRequest {
        id: target.run.clone(),
        after: 0,
        spec: None,
    };
    let log = tokio::spawn(stream_run(backend.clone(), actor, request, None, events_tx));
    let mut held = target.offset;
    let mut pending: Vec<(u64, v1::Product)> = vec![];
    let mut terminal = None;
    while terminal.is_none() {
        let Some(event) = events.recv().await else {
            let status = match log.await {
                Ok(Err(status)) => status,
                _ => tonic::Status::unavailable("the run's log ended before its outcome"),
            };
            return stop(source_error(&target.id, &status));
        };
        let Ok(event) = event else { continue };
        match event.event {
            Some(v1::run_event::Event::Product(product))
                if product.output == target.output
                    && product.index == target.index
                    && event.sequence > target.after =>
            {
                pending.push((event.sequence, product))
            }
            Some(v1::run_event::Event::Outcome(outcome)) => terminal = Some(outcome.status),
            _ => {}
        }
        // A burst of revisions is emitted once it settles: the next event is not yet here.
        if pending.is_empty() || terminal.is_none() && !events.is_empty() {
            continue;
        }
        let snapshot = match open_snapshot(&backend, actor, &target).await {
            Ok(snapshot) => snapshot,
            Err(status) => return stop(source_error(&target.id, &status)),
        };
        let last = pending[pending.len() - 1].1.rev;
        if snapshot.rev > last && terminal.is_none() {
            continue; // the log holds newer revisions: read them first
        }
        if snapshot.rev != last {
            let disagree = "the output's bytes and its log disagree";
            return stop(error(Some(&target.id), "unavailable", disagree));
        }
        let (items, now) = emit(&target, std::mem::take(&mut pending), snapshot, held);
        held = now;
        if work.send(Work::Push { number, items }).is_err() {
            return;
        }
    }
    let status = match terminal.as_deref() {
        Some("succeeded") => "completed",
        Some(other) => other,
        None => unreachable!(),
    };
    let snapshot = open_snapshot(&backend, actor, &target).await.ok();
    let items = vec![end(&target, Some(status), snapshot.as_ref())];
    let _ = work.send(Work::Push { number, items });
}

/// The followed output's new revisions as the client lacks them: a reset when a replacement
/// voids the bytes it holds, the revisions still current, then the bytes past `held`.
fn emit(
    target: &Target,
    pending: Vec<(u64, v1::Product)>,
    snapshot: OutputSnapshot,
    mut held: u64,
) -> (Vec<Item>, u64) {
    let replaced = pending.iter().rposition(|(_, p)| p.appended_from.is_none());
    let from = replaced.unwrap_or(0);
    let last = &pending[pending.len() - 1].1;
    let mut items = vec![];
    if replaced.is_some() && held > 0 || held > last.length {
        items.push(Item::Message {
            message: json!({"t": "reset", "id": target.id, "seq": pending[from].0}),
            last: false,
        });
        held = 0;
    }
    for (seq, product) in &pending[from..] {
        let mut entry = json!({"t": "entry", "id": target.id, "seq": seq, "output": product.output,
            "rev": product.rev, "length": product.length});
        if product.index > 0 {
            entry["index"] = product.index.into();
        }
        if let Some(appended) = product.appended_from {
            entry["appended_from"] = appended.into();
        }
        if product.duration_us > 0 {
            entry["duration_us"] = product.duration_us.into();
        }
        for (field, value) in [
            ("media_type", &product.media_type),
            ("label", &product.label),
        ] {
            if !value.is_empty() {
                entry[field] = value.as_str().into();
            }
        }
        items.push(Item::Entry(entry));
    }
    if held < last.length {
        items.push(Item::Message {
            message: json!({"t": "open", "id": target.id, "stream": target.number, "offset": held,
                "length": last.length - held, "etag": format!("r{}", last.rev)}),
            last: false,
        });
        items.push(Item::Body {
            body: Arc::new(Body {
                parts: snapshot.parts,
            }),
            from: held,
            to: last.length,
        });
        held = last.length;
    }
    (items, held)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binding_names_the_credential_the_client_chose() {
        let username = b"cozy+webrtc+v1/abc:client";
        let mut frame = vec![0x00, 0x01, 0x00, 0x00, 0x21, 0x12, 0xa4, 0x42];
        frame.extend([7; 12]);
        frame.extend([0x00, 0x06, 0x00, username.len() as u8]);
        frame.extend(username);
        frame.extend(vec![0; username.len().div_ceil(4) * 4 - username.len()]);
        assert_eq!(
            binding_username(&frame),
            Some(("cozy+webrtc+v1/abc".into(), "client".into()))
        );
        frame[8 + 12 + 4] = b'x'; // another prefix
        assert_eq!(binding_username(&frame), None);
    }

    struct Backend;
    impl MachineBackend for Backend {
        fn workspace(
            &self,
            _: VerifiedActor,
            _: super::super::pb::MachineExecutionWorkspaceQuery,
        ) -> Result<super::super::pb::MachineExecutionWorkspace, tonic::Status> {
            Ok(Default::default())
        }
    }

    /// Only handshakes are capped: any number of a key's viewers stay seated.
    #[test]
    fn authenticated_viewers_are_not_capped() {
        let signer = ed25519_dalek::SigningKey::from_bytes(&[84; 32]);
        let identity =
            MachineIdentity::ephemeral("media-test".into(), vec![signer.verifying_key()], vec![7; 32])
                .unwrap();
        let media = Media::new(Arc::new(identity), Arc::new(Backend)).unwrap();
        let address = "127.0.0.1".parse().unwrap();
        let viewers: Vec<_> = (0..MAX_CONNS + 1)
            .map(|_| {
                let viewer = media.admit(address).expect("an authenticated viewer was capped");
                viewer.authenticate("one-owner-key");
                viewer
            })
            .collect();
        assert_eq!(media.admitted.lock().unwrap().seats.len(), MAX_CONNS + 1);
        drop(viewers);
        assert!(media.admitted.lock().unwrap().seats.is_empty());
    }
}
