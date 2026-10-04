//! Degree 2 custody: GPU weight regions an executor filled, exported and offered stay alive
//! here as the driver's own fds. This process never loads CUDA: an exported fd is the
//! allocation's reference, closing the last reference anywhere frees it. Other executors on
//! the GPU attach duplicates read-only. Each reader holds a lease: one end of a socket pair whose
//! close (release, or the kernel at process death) ends it. Bytes are counted once per GPU, until
//! every lease ended and the fds here are closed.
use crate::execution::process_ended;
use crate::journal::ProcessBirth;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::Instant;

/// Region spans are multiples of the VMM granularity every supported GPU divides.
const GRANULE: u64 = 2 << 20;

/// One holding: one weight-set layout on one GPU, shared by every executor of the pod whose
/// layout matches, whoever submitted and whichever package it runs. A pod-level isolation
/// policy (per publisher) would add its domain as a field here.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct HoldingKey {
    /// GPU UUID.
    pub device: String,
    /// TensorFS plane layout digest: the same bytes at the same offsets.
    pub layout: String,
    /// What the regions hold beside the layout's stored bytes: empty (nothing), or the digest
    /// of the LoRA deltas an executor baked into them (Runtime `weights._variant`). A baked
    /// holding is its own allocation, kept and revoked like any other.
    pub variant: String,
}

/// One plane region: its chunks' bytes in address order (one fd each).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedRegion {
    pub region: u32,
    pub chunks: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Attachable.
    Ready,
    /// No new attachments; charged until every reader released or ended.
    Revoking,
}

#[derive(Clone, Debug, Serialize)]
pub struct HoldingFacts {
    pub key: HoldingKey,
    pub name: String,
    pub generation: u64,
    pub bytes: u64,
    pub phase: Phase,
    pub readers: Vec<ProcessBirth>,
    pub idle_ms: u64,
}

pub struct Attachment {
    pub generation: u64,
    pub regions: Vec<SharedRegion>,
    /// Duplicates for the reader, in `regions` order; close-on-exec.
    pub fds: Vec<OwnedFd>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Offered {
    Kept {
        generation: u64,
    },
    /// Another executor's offer of the same layout is held; the offered fds were closed.
    Duplicate,
}

/// One reader lease on one holding. The executor holds the other end of `lease` while it maps
/// the holding; its close, by the executor once it let the bytes go or by the kernel when the
/// process dies, ends the lease with no message (as GMS ties a lease to its connection). The
/// pidfd and exact birth are the machine's own observation of that death, as everywhere else.
pub struct Reader {
    pub birth: ProcessBirth,
    pub exit: File,
    lease: UnixStream,
}

impl Reader {
    /// A new lease and the end the executor keeps (close-on-exec until it is sent).
    pub fn lease(birth: ProcessBirth, exit: File) -> io::Result<(Reader, OwnedFd)> {
        let (ours, theirs) = UnixStream::pair()?;
        ours.set_nonblocking(true)?;
        Ok((
            Reader {
                birth,
                exit,
                lease: ours,
            },
            theirs.into(),
        ))
    }

    fn ended(&self) -> bool {
        hung_up(&self.lease)
            || (crate::os::ended(&self.exit) && process_ended(&self.birth).unwrap_or(false))
    }
}

/// The peer end of a lease is closed: the reader released, or its process is gone.
fn hung_up(lease: &UnixStream) -> bool {
    let mut poll = libc::pollfd {
        fd: lease.as_raw_fd(),
        events: libc::POLLIN | libc::POLLRDHUP,
        revents: 0,
    };
    // SAFETY: one live descriptor, instantaneous observation.
    let ready = unsafe { libc::poll(&mut poll, 1, 0) } > 0;
    ready && poll.revents & (libc::POLLHUP | libc::POLLRDHUP | libc::POLLERR) != 0
}

struct Holding {
    name: String,
    generation: u64,
    regions: Vec<SharedRegion>,
    fds: Vec<OwnedFd>,
    bytes: u64,
    phase: Phase,
    readers: Vec<Reader>,
    used: Instant,
}

#[derive(Default)]
pub struct ResidentCustody {
    generation: u64,
    holdings: BTreeMap<HoldingKey, Holding>,
}

impl ResidentCustody {
    /// Keep an executor's exported regions. The offering executor is a reader (its own
    /// mapping is a reference too). Regions a Ready holding lacks extend it; an overlap is a
    /// duplicate.
    pub fn offer(
        &mut self,
        key: HoldingKey,
        name: &str,
        regions: Vec<SharedRegion>,
        fds: Vec<OwnedFd>,
        reader: Reader,
    ) -> io::Result<Offered> {
        validate(&key, &regions, &fds)?;
        if let Some(held) = self.holdings.get_mut(&key) {
            if held.phase == Phase::Revoking {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "this layout is being revoked on this GPU",
                ));
            }
            let overlap = regions
                .iter()
                .any(|r| held.regions.iter().any(|h| h.region == r.region));
            if overlap {
                return Ok(Offered::Duplicate);
            }
            held.bytes += regions.iter().flat_map(|r| &r.chunks).sum::<u64>();
            held.regions.extend(regions);
            held.fds.extend(fds);
            held.readers.push(reader);
            held.used = Instant::now();
            return Ok(Offered::Kept {
                generation: held.generation,
            });
        }
        self.generation += 1;
        let bytes = regions.iter().flat_map(|r| &r.chunks).sum();
        self.holdings.insert(
            key,
            Holding {
                name: name.into(),
                generation: self.generation,
                regions,
                fds,
                bytes,
                phase: Phase::Ready,
                readers: vec![reader],
                used: Instant::now(),
            },
        );
        Ok(Offered::Kept {
            generation: self.generation,
        })
    }

    /// Duplicates of a Ready holding for `reader`, recorded as its lease. None: not held.
    pub fn attach(&mut self, key: &HoldingKey, reader: Reader) -> io::Result<Option<Attachment>> {
        let Some(held) = self.holdings.get_mut(key) else {
            return Ok(None);
        };
        if held.phase != Phase::Ready {
            return Ok(None);
        }
        let fds = held
            .fds
            .iter()
            .map(|fd| fd.try_clone())
            .collect::<io::Result<Vec<_>>>()?;
        held.readers.push(reader);
        held.used = Instant::now();
        Ok(Some(Attachment {
            generation: held.generation,
            regions: held.regions.clone(),
            fds,
        }))
    }

    /// Fence new attachments of one generation; returns the readers to ask to release it.
    pub fn begin_revoke(
        &mut self,
        key: &HoldingKey,
        generation: u64,
    ) -> io::Result<Vec<ProcessBirth>> {
        let held = self
            .holdings
            .get_mut(key)
            .filter(|h| h.generation == generation)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such holding generation"))?;
        held.phase = Phase::Revoking;
        Ok(held.readers.iter().map(|r| r.birth.clone()).collect())
    }

    /// A reader answered `revoke`: it unmapped and released the generation after its queued
    /// work. A reader that could not stays charged until its process ends.
    pub fn released(&mut self, key: &HoldingKey, generation: u64, birth: &ProcessBirth) {
        if let Some(held) = self
            .holdings
            .get_mut(key)
            .filter(|h| h.generation == generation)
        {
            held.readers.retain(|r| r.birth != *birth);
        }
    }

    /// End every lease whose connection closed (released, or the process died; their driver
    /// references ended with them), then close every revoked holding no reader maps. Returns
    /// what was released.
    pub fn collect(&mut self) -> Vec<(HoldingKey, u64)> {
        for held in self.holdings.values_mut() {
            held.readers.retain(|r| !r.ended());
        }
        let done: Vec<HoldingKey> = self
            .holdings
            .iter()
            .filter(|(_, h)| h.phase == Phase::Revoking && h.readers.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        done.into_iter()
            .filter_map(|key| self.holdings.remove(&key).map(|h| (key, h.bytes)))
            .collect()
    }

    /// Holdings `birth` maps: (key, generation).
    pub fn read_by(&self, birth: &ProcessBirth) -> Vec<(HoldingKey, u64)> {
        self.holdings
            .iter()
            .filter(|(_, h)| h.readers.iter().any(|r| r.birth == *birth))
            .map(|(k, h)| (k.clone(), h.generation))
            .collect()
    }

    /// The baked variant of `layout` on `device` offered last and still attachable.
    pub fn latest_variant(&self, device: &str, layout: &str) -> Option<String> {
        self.holdings
            .iter()
            .filter(|(k, h)| {
                k.device == device
                    && k.layout == layout
                    && !k.variant.is_empty()
                    && h.phase == Phase::Ready
            })
            .max_by_key(|(_, h)| h.generation)
            .map(|(k, _)| k.variant.clone())
    }

    /// Every holding on `device`, Ready or Revoking: counted once, until released.
    pub fn charged_bytes(&self, device: &str) -> u64 {
        self.holdings
            .iter()
            .filter(|(k, _)| k.device == device)
            .map(|(_, h)| h.bytes)
            .sum()
    }

    pub fn holdings(&self) -> Vec<HoldingFacts> {
        self.holdings
            .iter()
            .map(|(key, h)| HoldingFacts {
                key: key.clone(),
                name: h.name.clone(),
                generation: h.generation,
                bytes: h.bytes,
                phase: h.phase,
                readers: h.readers.iter().fold(Vec::new(), |mut births, r| {
                    if !births.contains(&r.birth) {
                        births.push(r.birth.clone());
                    }
                    births
                }),
                idle_ms: h.used.elapsed().as_millis() as u64,
            })
            .collect()
    }
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail.to_string())
}

fn validate(key: &HoldingKey, regions: &[SharedRegion], fds: &[OwnedFd]) -> io::Result<()> {
    let hex = key.layout.len() == 64
        && key
            .layout
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if key.device.is_empty() || !hex || regions.is_empty() {
        return Err(invalid(
            "a holding needs a GPU, a layout digest and regions",
        ));
    }
    if !(key.variant.is_empty() || key.variant.starts_with("sha256:")) {
        return Err(invalid("a holding's variant is a digest"));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut chunks = 0;
    for r in regions {
        if !seen.insert(r.region) || r.chunks.is_empty() {
            return Err(invalid("regions repeat or carry no chunks"));
        }
        if r.chunks.iter().any(|&n| n == 0 || n % GRANULE != 0) {
            return Err(invalid("a chunk is not a whole number of 2 MiB granules"));
        }
        chunks += r.chunks.len();
    }
    if chunks != fds.len() {
        return Err(invalid("one fd per chunk"));
    }
    for fd in fds {
        // An exported VMM handle is a descriptor of the driver's control device.
        let stat = nix::sys::stat::fstat(fd.as_fd()).map_err(io::Error::from)?;
        if stat.st_mode & libc::S_IFMT != libc::S_IFCHR {
            return Err(invalid("a GPU allocation must be a driver descriptor"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::process_birth;
    use std::os::fd::FromRawFd;
    use std::process::{Child, Command, Stdio};

    fn devnull() -> OwnedFd {
        // A character device stands in for a driver fd: custody never interprets it.
        File::open("/dev/null").unwrap().into()
    }
    fn pidfd(pid: u32) -> File {
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        assert!(raw >= 0);
        unsafe { File::from_raw_fd(raw) }
    }
    /// A lease whose executor end this test process keeps.
    fn mine() -> (Reader, OwnedFd) {
        let me = std::process::id();
        Reader::lease(process_birth(me).unwrap(), pidfd(me)).unwrap()
    }
    /// A real executor stand-in: a child process holding its lease end as stdin. The reader's
    /// birth is this test process, which stays alive: only the connection can end the lease.
    fn held_by_child() -> (Reader, Child) {
        let (reader, end) = mine();
        let child = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::from(end))
            .spawn()
            .unwrap();
        (reader, child)
    }
    /// An executor of its own: a real process whose birth and exit are the lease's.
    fn executor() -> (Reader, OwnedFd, Child) {
        let child = Command::new("sleep").arg("60").spawn().unwrap();
        let (reader, end) =
            Reader::lease(process_birth(child.id()).unwrap(), pidfd(child.id())).unwrap();
        (reader, end, child)
    }
    /// The layout whose digest repeats `digit`, on GPU-1.
    fn key(digit: &str) -> HoldingKey {
        HoldingKey {
            device: "GPU-1".into(),
            layout: digit.repeat(64),
            variant: String::new(),
        }
    }
    fn region(index: u32, chunks: Vec<u64>) -> SharedRegion {
        SharedRegion {
            region: index,
            chunks,
        }
    }
    fn regions() -> Vec<SharedRegion> {
        vec![
            region(0, vec![GRANULE]),
            region(2, vec![64 << 20, 2 * GRANULE]),
        ]
    }
    fn fds(n: usize) -> Vec<OwnedFd> {
        (0..n).map(|_| devnull()).collect()
    }

    #[test]
    fn a_crashed_readers_lease_ends_with_its_connection_and_the_holding_stays() {
        let mut custody = ResidentCustody::default();
        let (offerer, mut child) = held_by_child();
        let kept = custody
            .offer(key("a"), "sdxl/unet", regions(), fds(3), offerer)
            .unwrap();
        assert_eq!(kept, Offered::Kept { generation: 1 });
        let total = GRANULE + (64 << 20) + 2 * GRANULE;
        assert_eq!(custody.charged_bytes("GPU-1"), total);
        assert!(custody.collect().is_empty());
        assert_eq!(
            custody.holdings()[0].readers.len(),
            1,
            "the live connection is a lease"
        );

        // The executor crashes: no message, its kernel-closed connection ends the lease.
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(
            custody.collect().is_empty(),
            "a Ready holding outlives its readers"
        );
        assert!(custody.holdings()[0].readers.is_empty());

        // A replacement attaches duplicates under a lease of its own.
        let (reader, end) = mine();
        let a = custody.attach(&key("a"), reader).unwrap().unwrap();
        assert_eq!((a.generation, a.fds.len(), a.regions), (1, 3, regions()));
        assert!(
            custody.attach(&key("b"), mine().0).unwrap().is_none(),
            "another layout is another holding"
        );

        // Revoke fences new attachments; the bytes stay charged until the lease ends, which
        // the reader does by closing its end once it let the bytes go (still alive here).
        custody.begin_revoke(&key("a"), 1).unwrap();
        assert!(custody.attach(&key("a"), mine().0).unwrap().is_none());
        assert!(custody.collect().is_empty());
        assert_eq!(custody.charged_bytes("GPU-1"), total);
        drop(end);
        // Another test's spawn holds a copy of the end between its fork and exec: the close
        // is seen once that child has exec'd.
        let collected = (0..500).find_map(|_| {
            let ended = custody.collect();
            if ended.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Some(ended).filter(|ended| !ended.is_empty())
        });
        assert_eq!(collected, Some(vec![(key("a"), total)]));
        assert_eq!(custody.charged_bytes("GPU-1"), 0);
    }

    #[test]
    fn every_executor_of_the_pod_shares_a_matching_layout_once_per_gpu() {
        let mut custody = ResidentCustody::default();
        let one = || vec![region(0, vec![GRANULE])];
        // Two packages' executors (real processes) bind the same checkpoint: the first offers,
        // the second attaches, and the GPU is charged the bytes once.
        let (first, _a_end, mut package_a) = executor();
        custody
            .offer(key("a"), "sdxl", one(), fds(1), first)
            .unwrap();
        let (second, _b_end, mut package_b) = executor();
        let attached = custody.attach(&key("a"), second).unwrap().unwrap();
        assert_eq!((attached.generation, attached.fds.len()), (1, 1));
        assert_eq!(custody.holdings().len(), 1);
        assert_eq!(custody.holdings()[0].readers.len(), 2);
        assert_eq!(custody.charged_bytes("GPU-1"), GRANULE);
        // Another layout, and the same layout on another GPU, are holdings of their own.
        assert!(custody.attach(&key("b"), mine().0).unwrap().is_none());
        let (other, _other_end) = mine();
        custody
            .offer(key("b"), "anima", one(), fds(1), other)
            .unwrap();
        let elsewhere = HoldingKey {
            device: "GPU-2".into(),
            ..key("a")
        };
        assert!(custody.attach(&elsewhere, mine().0).unwrap().is_none());
        assert_eq!(custody.holdings().len(), 2);
        assert_eq!(custody.charged_bytes("GPU-1"), 2 * GRANULE);
        // Either executor's end leaves the other's lease and the holding as they were.
        package_a.kill().unwrap();
        package_a.wait().unwrap();
        assert!(custody.collect().is_empty());
        let shared = custody
            .holdings()
            .into_iter()
            .find(|h| h.key == key("a"))
            .unwrap();
        assert_eq!(shared.readers.len(), 1);
        package_b.kill().unwrap();
        package_b.wait().unwrap();
    }

    #[test]
    fn a_baked_variant_is_a_holding_of_its_own_beside_the_stored_bytes() {
        let mut custody = ResidentCustody::default();
        let baked = |digit: &str| HoldingKey {
            variant: format!("sha256:{}", digit.repeat(64)),
            ..key("a")
        };
        let one = || vec![region(1, vec![GRANULE])];
        // The stored bytes and two baked variants of one layout: three holdings, charged apart.
        for holding in [key("a"), baked("1"), baked("2")] {
            let kept = custody.offer(holding, "h3/dit", one(), fds(1), mine().0);
            assert!(matches!(kept, Ok(Offered::Kept { .. })));
        }
        assert_eq!(custody.holdings().len(), 3);
        assert_eq!(custody.charged_bytes("GPU-1"), 3 * GRANULE);
        // A replacement asks for the layout's latest baked variant, and maps exactly that one.
        let latest = custody.latest_variant("GPU-1", &key("a").layout).unwrap();
        assert_eq!(latest, baked("2").variant);
        let attached = custody.attach(&baked("2"), mine().0).unwrap().unwrap();
        assert_eq!(attached.fds.len(), 1);
        // A variant never answers for the stored bytes, nor another variant, nor another layout.
        assert!(custody.attach(&baked("3"), mine().0).unwrap().is_none());
        assert!(custody.latest_variant("GPU-1", &key("b").layout).is_none());
        assert!(custody.latest_variant("GPU-2", &key("a").layout).is_none());
        // A variant that is not a digest is refused before custody.
        let named = HoldingKey {
            variant: "*".into(),
            ..key("a")
        };
        let refused = custody.offer(named, "h3/dit", one(), fds(1), mine().0);
        assert!(refused.is_err());
    }

    #[test]
    fn regions_a_holding_lacks_extend_it_and_an_overlap_is_a_duplicate() {
        let mut custody = ResidentCustody::default();
        let (first, _end) = mine();
        let one = vec![region(0, vec![GRANULE])];
        assert_eq!(
            custody.offer(key("a"), "n", one, fds(1), first).unwrap(),
            Offered::Kept { generation: 1 }
        );
        let (more, _more_end) = mine();
        let two = vec![region(1, vec![2 * GRANULE])];
        assert_eq!(
            custody.offer(key("a"), "n", two, fds(1), more).unwrap(),
            Offered::Kept { generation: 1 }
        );
        assert_eq!(custody.charged_bytes("GPU-1"), 3 * GRANULE);
        let (dup, _) = mine();
        let again = vec![region(1, vec![2 * GRANULE])];
        assert_eq!(
            custody.offer(key("a"), "n", again, fds(1), dup).unwrap(),
            Offered::Duplicate
        );
        let a = custody.attach(&key("a"), mine().0).unwrap().unwrap();
        assert_eq!((a.regions.len(), a.fds.len()), (2, 2));
        assert_eq!(
            custody.holdings()[0].readers.len(),
            1,
            "one process, counted once"
        );
    }

    #[test]
    fn offers_are_validated_before_custody() {
        let mut custody = ResidentCustody::default();
        let mut bad = key("a");
        bad.layout = "not-a-digest".into();
        assert!(custody
            .offer(bad, "n", regions(), fds(3), mine().0)
            .is_err());
        assert!(custody
            .offer(key("a"), "n", regions(), fds(2), mine().0)
            .is_err());
        let odd = vec![region(0, vec![GRANULE + 1])];
        assert!(custody.offer(key("a"), "n", odd, fds(1), mine().0).is_err());
        let file: OwnedFd = File::open("/proc/self/stat").unwrap().into();
        let one = vec![region(0, vec![GRANULE])];
        assert!(custody
            .offer(key("a"), "n", one, vec![file], mine().0)
            .is_err());
        assert_eq!(custody.charged_bytes("GPU-1"), 0);
    }
}
