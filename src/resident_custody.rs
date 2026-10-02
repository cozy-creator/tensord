//! Degree 2 custody: GPU weight regions an executor filled, exported and offered stay alive
//! here as the driver's own fds. This process never loads CUDA: an exported fd is the
//! allocation's reference, closing the last reference anywhere frees it. Other executors on
//! the GPU attach duplicates read-only. Bytes are counted once per GPU, until every reader has
//! released or ended and the fds here are closed.
use crate::execution::process_ended;
use crate::journal::ProcessBirth;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::time::Instant;

/// Region spans are multiples of the VMM granularity every supported GPU divides.
const GRANULE: u64 = 2 << 20;

/// One holding: one weight-set layout on one GPU, shared only within one actor.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct HoldingKey {
    pub actor: String,
    /// GPU UUID.
    pub device: String,
    /// TensorFS plane layout digest: the same bytes at the same offsets.
    pub layout: String,
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

/// An executor that maps a holding: its exact birth and its pidfd.
pub struct Reader {
    pub birth: ProcessBirth,
    pub exit: File,
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
    /// Keep an executor's exported regions. The offering executor is the first reader (its
    /// own mapping is a reference too).
    pub fn offer(
        &mut self,
        key: HoldingKey,
        name: &str,
        regions: Vec<SharedRegion>,
        fds: Vec<OwnedFd>,
        reader: Reader,
    ) -> io::Result<Offered> {
        validate(&key, &regions, &fds)?;
        if let Some(held) = self.holdings.get(&key) {
            return match held.phase {
                Phase::Ready => Ok(Offered::Duplicate),
                Phase::Revoking => Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "this layout is being revoked on this GPU",
                )),
            };
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
        if !held.readers.iter().any(|r| r.birth == reader.birth) {
            held.readers.push(reader);
        }
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

    /// Forget readers whose processes ended (their driver references ended with them), then
    /// close every revoked holding no reader maps. Returns what was released.
    pub fn collect(&mut self) -> Vec<(HoldingKey, u64)> {
        for held in self.holdings.values_mut() {
            held.readers.retain(|r| {
                !(crate::os::ended(&r.exit) && process_ended(&r.birth).unwrap_or(false))
            });
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
                readers: h.readers.iter().map(|r| r.birth.clone()).collect(),
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
    if key.actor.is_empty() || key.device.is_empty() || !hex || regions.is_empty() {
        return Err(invalid(
            "a holding needs an actor, a GPU, a layout digest and regions",
        ));
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

    fn devnull() -> OwnedFd {
        // A character device stands in for a driver fd: custody never interprets it.
        File::open("/dev/null").unwrap().into()
    }
    fn pidfd(pid: u32) -> File {
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        assert!(raw >= 0);
        unsafe { File::from_raw_fd(raw) }
    }
    fn reader(pid: u32) -> Reader {
        Reader {
            birth: process_birth(pid).unwrap(),
            exit: pidfd(pid),
        }
    }
    fn key(actor: &str) -> HoldingKey {
        HoldingKey {
            actor: actor.into(),
            device: "GPU-1".into(),
            layout: "a".repeat(64),
        }
    }
    fn regions() -> Vec<SharedRegion> {
        vec![
            SharedRegion {
                region: 0,
                chunks: vec![GRANULE],
            },
            SharedRegion {
                region: 2,
                chunks: vec![64 << 20, 2 * GRANULE],
            },
        ]
    }

    #[test]
    fn a_holding_is_counted_once_and_outlives_its_offerer_until_revoked() {
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let mut custody = ResidentCustody::default();
        let fds: Vec<OwnedFd> = (0..3).map(|_| devnull()).collect();
        let kept = custody
            .offer(key("x"), "sdxl/unet", regions(), fds, reader(child.id()))
            .unwrap();
        assert_eq!(kept, Offered::Kept { generation: 1 });
        let total = GRANULE + (64 << 20) + 2 * GRANULE;
        assert_eq!(custody.charged_bytes("GPU-1"), total);

        // A second offer of the same layout is a duplicate; another actor's is its own.
        let dup = custody.offer(
            key("x"),
            "sdxl/unet",
            regions(),
            (0..3).map(|_| devnull()).collect(),
            reader(std::process::id()),
        );
        assert_eq!(dup.unwrap(), Offered::Duplicate);
        assert_eq!(custody.charged_bytes("GPU-1"), total);
        assert!(custody
            .attach(&key("y"), reader(std::process::id()))
            .unwrap()
            .is_none());

        // The offerer dies; the holding stays and a replacement attaches duplicates.
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(custody.collect().is_empty());
        let me = process_birth(std::process::id()).unwrap();
        let a = custody
            .attach(&key("x"), reader(std::process::id()))
            .unwrap()
            .unwrap();
        assert_eq!((a.generation, a.fds.len(), a.regions), (1, 3, regions()));
        assert_eq!(custody.holdings()[0].readers, vec![me.clone()]);

        // Revoke fences new attachments; the bytes stay charged until the reader releases.
        assert_eq!(
            custody.begin_revoke(&key("x"), 1).unwrap(),
            vec![me.clone()]
        );
        assert!(custody
            .attach(&key("x"), reader(std::process::id()))
            .unwrap()
            .is_none());
        assert!(custody.collect().is_empty());
        assert_eq!(custody.charged_bytes("GPU-1"), total);
        custody.released(&key("x"), 1, &me);
        assert_eq!(custody.collect(), vec![(key("x"), total)]);
        assert_eq!(custody.charged_bytes("GPU-1"), 0);
    }

    #[test]
    fn offers_are_validated_before_custody() {
        let mut custody = ResidentCustody::default();
        let me = std::process::id();
        let mut bad = key("x");
        bad.layout = "not-a-digest".into();
        assert!(custody
            .offer(
                bad,
                "n",
                regions(),
                (0..3).map(|_| devnull()).collect(),
                reader(me)
            )
            .is_err());
        assert!(custody
            .offer(
                key("x"),
                "n",
                regions(),
                (0..2).map(|_| devnull()).collect(),
                reader(me)
            )
            .is_err());
        let odd = vec![SharedRegion {
            region: 0,
            chunks: vec![GRANULE + 1],
        }];
        assert!(custody
            .offer(key("x"), "n", odd, vec![devnull()], reader(me))
            .is_err());
        let file: OwnedFd = File::open("/proc/self/stat").unwrap().into();
        let one = vec![SharedRegion {
            region: 0,
            chunks: vec![GRANULE],
        }];
        assert!(custody
            .offer(key("x"), "n", one, vec![file], reader(me))
            .is_err());
        assert_eq!(custody.charged_bytes("GPU-1"), 0);
    }
}
