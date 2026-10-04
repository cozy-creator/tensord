use nix::sys::socket::{sendmsg, ControlMessage, MsgFlags};
use serde::{Deserialize, Serialize};
use std::io::{self, IoSlice, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;

pub const MAX_FRAME: usize = 64 * 1024;
pub const CAPS: &[&str] = &["weights.hosted/1", "objects.put-fd/1", "execution.cpu/1"];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Object {
    pub sha256: String,
    pub length: u64,
}
#[derive(Debug, Deserialize)]
pub struct Request {
    pub seq: u64,
    #[serde(flatten)]
    pub command: Command,
}
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Command {
    Hello {
        #[serde(default)]
        runtime: String,
        #[serde(default)]
        capabilities: Vec<String>,
    },
    Import {
        object: Object,
    },
    Attach {
        object: Object,
    },
    Release {
        lease: u64,
        incarnation: String,
    },
    Stats,
    Submit {
        key: String,
        generation: String,
        entrypoint: String,
        input: serde_json::Value,
    },
    Execution {
        id: String,
    },
    Executions,
    Cancel {
        id: String,
    },
    ReadResult {
        id: String,
        index: usize,
    },
    Shutdown,
    #[serde(other)]
    Unknown,
}
#[derive(Debug, Serialize)]
pub struct Reply {
    pub seq: u64,
    #[serde(flatten)]
    pub body: Body,
}
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Body {
    Hello {
        version: &'static str,
        tensorfs: &'static str,
        incarnation: String,
        capabilities: Vec<String>,
    },
    Imported {
        object: Object,
        admitted: bool,
    },
    Attached {
        object: Object,
        lease: u64,
        incarnation: String,
        tier: &'static str,
    },
    Released,
    Stats {
        host_bytes: u64,
        cached_objects: usize,
        active_leases: usize,
        host_budget: u64,
    },
    Shutdown,
    Execution {
        record: Box<crate::journal::Execution>,
    },
    Executions {
        records: Vec<crate::journal::Execution>,
    },
    ResultArtifact {
        id: String,
        artifact: crate::journal::Artifact,
    },
    Error {
        code: &'static str,
        detail: String,
    },
}
pub fn read(stream: &mut UnixStream) -> io::Result<Option<Request>> {
    let mut header = [0; 4];
    // EOF is clean only between frames; truncated headers are errors.
    if stream.read(&mut header[..1])? == 0 {
        return Ok(None);
    }
    stream.read_exact(&mut header[1..])?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid control frame length",
        ));
    }
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes)?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(io::Error::other)
}
pub fn write(stream: &mut UnixStream, reply: &Reply) -> io::Result<()> {
    let bytes = serde_json::to_vec(reply)?;
    if bytes.len() > MAX_FRAME {
        return Err(io::Error::other("reply exceeds control frame"));
    }
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(&bytes)
}
pub fn send_fd(stream: &UnixStream, fd: &impl AsRawFd) -> io::Result<()> {
    let fds = [fd.as_raw_fd()];
    let sent = sendmsg::<()>(
        stream.as_raw_fd(),
        &[IoSlice::new(&[0])],
        &[ControlMessage::ScmRights(&fds)],
        MsgFlags::MSG_NOSIGNAL,
        None,
    )?;
    if sent != 1 {
        return Err(io::Error::other("descriptor marker truncated"));
    }
    Ok(())
}
pub fn recv_fd(stream: &UnixStream) -> io::Result<OwnedFd> {
    let mut marker = [255u8];
    // usize storage satisfies cmsghdr alignment; every iteration is bounded by
    // the kernel-returned control length. Parse delivered fds even on CTRUNC,
    // solely to close that prefix before rejecting the message.
    let mut control = [0usize; 8];
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    // SAFETY: zero is a valid initial msghdr; all writable buffers live through recvmsg.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = std::mem::size_of_val(&control);
    // SAFETY: correctly sized aligned control/data buffers and live socket.
    let count = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
    if count < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut valid =
        count == 1 && marker == [0] && msg.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC) == 0;
    let mut received = Vec::new();
    // SAFETY: msghdr contains buffers initialized by recvmsg; bounds checked below.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        while !header.is_null() {
            let start = header as usize - msg.msg_control as usize;
            let length = (*header).cmsg_len;
            let base = libc::CMSG_LEN(0) as usize;
            if length < base || start.saturating_add(length) > msg.msg_controllen {
                valid = false;
                break;
            }
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let bytes = length - base;
                valid &= bytes.is_multiple_of(std::mem::size_of::<i32>());
                for i in 0..bytes / std::mem::size_of::<i32>() {
                    let fd = std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<i32>().add(i));
                    received.push(OwnedFd::from_raw_fd(fd));
                }
            } else {
                valid = false;
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    if !valid || received.len() != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected exactly one descriptor",
        ));
    }
    Ok(received.pop().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncated_descriptor_batch_closes_delivered_prefix() {
        let (send, receive) = UnixStream::pair().unwrap();
        let file = crate::os::memfd().unwrap();
        // Other tests open files concurrently. Count only this unique memfd's
        // descriptors so unrelated activity cannot hide a leak or fail this check.
        use std::os::unix::fs::MetadataExt;
        let original = file.metadata().unwrap();
        let copies = || {
            std::fs::read_dir("/proc/self/fd")
                .unwrap()
                .filter_map(Result::ok)
                .filter_map(|entry| std::fs::metadata(entry.path()).ok())
                .filter(|metadata| {
                    metadata.dev() == original.dev() && metadata.ino() == original.ino()
                })
                .count()
        };
        assert_eq!(copies(), 1);
        let rights = vec![file.as_raw_fd(); 32];
        sendmsg::<()>(
            send.as_raw_fd(),
            &[IoSlice::new(&[0])],
            &[ControlMessage::ScmRights(&rights)],
            MsgFlags::empty(),
            None,
        )
        .unwrap();
        assert!(recv_fd(&receive).is_err());
        assert_eq!(copies(), 1);
    }
}
