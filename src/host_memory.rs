//! Live host memory: what this process can still allocate before its tightest limit. Read on
//! every decision, never cached: a budget fixed at boot describes a machine nobody else is on.
//! Port of Runtime `proctree.host_memory` (cgroup v1 and v2, every parent, and `MemAvailable`).
use serde::Serialize;
use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct HostMemory {
    /// Bytes left before the tightest limit on this process's cgroup path, or the host's
    /// `MemAvailable` when that is tighter. -1: unreadable (never 0).
    pub available: i64,
    /// Shared memory (memfd/tmpfs) charged where that limit is: sealed and pinned weight tiers.
    pub shmem: i64,
    /// The host's `MemAvailable`, for the record; -1 when unreadable.
    pub mem_available: i64,
}

/// The host's `MemTotal`; -1 when unreadable.
pub fn total() -> i64 {
    meminfo("MemTotal")
}

fn meminfo(field: &str) -> i64 {
    let Ok(file) = fs::File::open("/proc/meminfo") else {
        return -1;
    };
    for line in io::BufReader::new(file).lines().map_while(Result::ok) {
        if let Some(rest) = line.strip_prefix(field).and_then(|r| r.strip_prefix(':')) {
            return rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<i64>().ok())
                .map_or(-1, |kib| kib * 1024);
        }
    }
    -1
}

fn number(path: &Path) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let text = text.trim();
    // cgroup v2 spells "unlimited" `max`; v1 a page-rounded value near LONG_MAX
    text.parse::<u64>().ok().filter(|&n| n > 0 && n < (1 << 60))
}

/// (`inactive_file` + `active_file`, `shmem`) from a cgroup's `memory.stat`; v1 states its
/// subtree's under `total_`.
fn stat(cgroup: &Path) -> Option<(u64, u64)> {
    let text = fs::read_to_string(cgroup.join("memory.stat")).ok()?;
    let row = |name: &str| -> u64 {
        let find = |key: &str| {
            text.lines().find_map(|l| {
                l.strip_prefix(key)
                    .and_then(|r| r.strip_prefix(' '))
                    .and_then(|v| v.trim().parse().ok())
            })
        };
        find(&format!("total_{name}"))
            .or_else(|| find(name))
            .unwrap_or(0)
    };
    Some((row("inactive_file") + row("active_file"), row("shmem")))
}

/// This process's memory cgroup and every parent up to the controller root, innermost first,
/// with the limit files and the usage file of its version.
fn cgroups() -> Option<(Vec<PathBuf>, &'static [&'static str], &'static str)> {
    let root = Path::new("/sys/fs/cgroup");
    let memberships = fs::read_to_string("/proc/self/cgroup").ok()?;
    let (leaf, top, limits, usage): (PathBuf, PathBuf, &[&str], &str) =
        if root.join("cgroup.controllers").is_file() {
            let relative = memberships.lines().find_map(|l| l.strip_prefix("0::"))?;
            (
                root.join(relative.trim_start_matches('/')),
                root.to_path_buf(),
                &["memory.high", "memory.max"],
                "memory.current",
            )
        } else {
            let relative = memberships.lines().find_map(|l| {
                let mut fields = l.splitn(3, ':');
                let (_, names, path) = (fields.next()?, fields.next()?, fields.next()?);
                names.split(',').any(|n| n == "memory").then_some(path)
            })?;
            // The memory controller's mount: its root (mountinfo field 4) is the part of the
            // membership path the mount hides (a container sees its own cgroup at the mount).
            let mounts = fs::read_to_string("/proc/self/mountinfo").ok()?;
            let (mount_root, point) = mounts.lines().find_map(|l| {
                let fields: Vec<&str> = l.split(' ').collect();
                let dash = fields.iter().position(|f| *f == "-")?;
                (fields.get(dash + 1) == Some(&"cgroup")
                    && fields.get(dash + 3)?.split(',').any(|o| o == "memory"))
                .then(|| (fields[3].to_string(), PathBuf::from(fields[4])))
            })?;
            let inside = relative
                .strip_prefix(mount_root.trim_end_matches('/'))
                .unwrap_or(relative);
            (
                point.join(inside.trim_start_matches('/')),
                point,
                &["memory.limit_in_bytes"],
                "memory.usage_in_bytes",
            )
        };
    let mut path = Vec::new();
    let mut at = Some(leaf.as_path());
    while let Some(cgroup) = at {
        path.push(cgroup.to_path_buf());
        if cgroup == top {
            break;
        }
        at = cgroup.parent().filter(|p| p.starts_with(&top));
    }
    Some((path, limits, usage))
}

/// The live budget: the tightest of every cgroup on this process's path (each `memory.high`
/// and `memory.max`, less what that cgroup uses and cannot give back; page cache is
/// reclaimable) and the host's `MemAvailable`.
pub fn read() -> HostMemory {
    let mem_available = meminfo("MemAvailable");
    let mut best = HostMemory {
        available: mem_available,
        shmem: meminfo("Shmem").max(0),
        mem_available,
    };
    let Some((path, limits, usage)) = cgroups() else {
        return best;
    };
    for cgroup in path {
        let Some(bound) = limits
            .iter()
            .filter_map(|name| number(&cgroup.join(name)))
            .min()
        else {
            continue;
        };
        let (Some(used), Some((cache, shmem))) = (number(&cgroup.join(usage)), stat(&cgroup))
        else {
            continue;
        };
        let room = bound.saturating_sub(used.saturating_sub(cache)) as i64;
        if best.available < 0 || room < best.available {
            best = HostMemory {
                available: room,
                shmem: shmem as i64,
                mem_available,
            };
        }
    }
    best
}

/// One process's resident and proportional set sizes (`/proc/<pid>/smaps_rollup`): its PSS
/// counts a shared sealed layout once across every process that maps it.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct ProcessMemory {
    pub rss: u64,
    pub pss: u64,
    pub pss_shmem: u64,
}

pub fn process(pid: u32) -> io::Result<ProcessMemory> {
    let text = fs::read_to_string(format!("/proc/{pid}/smaps_rollup"))?;
    let field = |name: &str| {
        text.lines()
            .find_map(|l| {
                l.strip_prefix(name)?
                    .strip_prefix(':')?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .map_or(0, |kib| kib * 1024)
    };
    Ok(ProcessMemory {
        rss: field("Rss"),
        pss: field("Pss"),
        pss_shmem: field("Pss_Shmem"),
    })
}

/// Private bytes (PSS less shared pages) of `pid` and every process descended from it: what a
/// process tree holds that no one else counts.
pub fn tree_private(pid: u32) -> u64 {
    let mut parents: Vec<(u32, u32)> = vec![];
    if let Ok(entries) = fs::read_dir("/proc") {
        for entry in entries.flatten() {
            let Some(child) = entry.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
                continue;
            };
            let Ok(stat) = fs::read_to_string(format!("/proc/{child}/stat")) else {
                continue;
            };
            // the parent is the second field after the command's closing parenthesis
            let parent = stat
                .rsplit_once(')')
                .and_then(|(_, rest)| rest.split_whitespace().nth(1))
                .and_then(|p| p.parse().ok());
            if let Some(parent) = parent {
                parents.push((child, parent));
            }
        }
    }
    let mut tree = vec![pid];
    let mut at = 0;
    while at < tree.len() {
        let parent = tree[at];
        tree.extend(parents.iter().filter(|(_, p)| *p == parent).map(|(c, _)| *c));
        at += 1;
    }
    tree.iter()
        .filter_map(|pid| process(*pid).ok())
        .map(|m| m.pss.saturating_sub(m.pss_shmem))
        .sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_live_budget_is_read_and_never_above_mem_available() {
        let facts = super::read();
        assert!(facts.mem_available > 0, "{facts:?}");
        assert!(
            facts.available > 0 && facts.available <= facts.mem_available,
            "{facts:?}"
        );
        let own = super::process(std::process::id()).unwrap();
        assert!(own.rss > 0 && own.pss > 0 && own.pss <= own.rss, "{own:?}");
        let mut child = std::process::Command::new("sleep").arg("5").spawn().unwrap();
        let tree = super::tree_private(std::process::id());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(tree > own.pss.saturating_sub(own.pss_shmem), "a child counts too: {tree}");
    }
}
