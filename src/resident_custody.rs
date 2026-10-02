//! Live resident custody, using the existing Engine's actor/attempt/process authority.
//! No scheduler, CUDA calls, timer kills, second journal or restart handle adoption.
use crate::execution::{process_birth, process_ended, Engine};
use crate::journal::{ProcessBirth, State};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AllocationId {
    pub owner_epoch: String,
    pub id: String,
    pub generation: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidentKey {
    pub actor: String,
    pub device_uuid: String,
    pub content_sha256: String,
    pub layout_sha256: String,
    pub representation: String,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Filling,
    Ready,
    Revoking,
    Quarantined,
    Releasing,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Writer,
    Reader,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Ticket {
    pub allocation: AllocationId,
    pub lease: String,
    pub execution_id: String,
    pub process: ProcessBirth,
    pub role: Role,
}
#[derive(Clone, Debug, Serialize)]
pub struct Record {
    pub allocation: AllocationId,
    pub key: ResidentKey,
    pub physical_bytes: u64,
    pub phase: Phase,
    pub recipients: Vec<Ticket>,
}
/// Actual native backing and host-source obligations, never serialized/adopted by an ID.
pub struct Resources {
    pub backing: Arc<dyn Send + Sync>,
    pub sources: Arc<dyn Send + Sync>,
}
struct Recipient {
    ticket: Ticket,
    death: File,
}
struct Resident {
    record: Record,
    resources: Option<Resources>,
    recipients: BTreeMap<String, Recipient>,
}

/// Completion cannot be made by deserializing a peer's boolean or progress counter.
pub struct NativeCompletion {
    ticket: Ticket,
}
impl NativeCompletion {
    /// # Safety
    /// Only a qualified native bridge may mint this after it fenced all new uses/views,
    /// proved every successful DMA/kernel complete, unmapped that generation, released
    /// tensor owners/imported handles and closed every recipient export-FD duplicate.
    /// A CPU acknowledgement, socket EOF or one copy-stream event does not suffice.
    pub unsafe fn after_native_release(ticket: Ticket) -> Self {
        Self { ticket }
    }
}

/// Single device actor's bounded live inventory. Hardware eligibility belongs to the
/// existing supervisor. These methods never advertise a capability or launch work.
pub struct ResidentCustody {
    engine: Arc<Engine>,
    epoch: String,
    maximum_allocations: usize,
    maximum_recipients: usize,
    residents: BTreeMap<String, Resident>,
}
fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}
fn conflict(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, detail)
}
impl ResidentCustody {
    pub fn new(engine: Arc<Engine>, maximum_allocations: usize, maximum_recipients: usize) -> Self {
        Self {
            engine,
            epoch: uuid::Uuid::new_v4().to_string(),
            maximum_allocations,
            maximum_recipients,
            residents: BTreeMap::new(),
        }
    }
    pub fn epoch(&self) -> &str {
        &self.epoch
    }
    pub fn charged_bytes(&self) -> u64 {
        self.residents
            .values()
            .map(|r| r.record.physical_bytes)
            .sum()
    }
    pub fn records(&self) -> Vec<Record> {
        self.residents
            .values()
            .map(|resident| {
                let mut record = resident.record.clone();
                record.recipients = resident
                    .recipients
                    .values()
                    .map(|r| r.ticket.clone())
                    .collect();
                record
            })
            .collect()
    }
    /// Register real backing already created by the native actor. Capacity is a typed
    /// waiting condition for the supervisor's paging/eviction ladder, not request refusal.
    pub fn register(
        &mut self,
        key: ResidentKey,
        physical_bytes: u64,
        resources: Resources,
    ) -> io::Result<AllocationId> {
        let digest = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        if key.actor.is_empty()
            || key.device_uuid.is_empty()
            || key.representation.is_empty()
            || !digest(&key.content_sha256)
            || !digest(&key.layout_sha256)
            || physical_bytes == 0
        {
            return Err(invalid("resident identity/layout/backing is incomplete"));
        }
        if self.residents.len() >= self.maximum_allocations {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "resident inventory full; wait/evict/use a lower rung",
            ));
        }
        self.charged_bytes()
            .checked_add(physical_bytes)
            .ok_or_else(|| invalid("physical charge overflow"))?;
        let allocation = AllocationId {
            owner_epoch: self.epoch.clone(),
            id: uuid::Uuid::new_v4().to_string(),
            generation: 1,
        };
        self.residents.insert(
            allocation.id.clone(),
            Resident {
                record: Record {
                    allocation: allocation.clone(),
                    key,
                    physical_bytes,
                    phase: Phase::Filling,
                    recipients: Vec::new(),
                },
                resources: Some(resources),
                recipients: BTreeMap::new(),
            },
        );
        self.engine.notify_activity();
        Ok(allocation)
    }
    /// Bind the retained pidfd to an accepted actor and the Engine's exact recorded birth.
    /// Supply the managed Executor's actual kernel pidfd, not one reconstructed from an ID.
    pub fn attach(
        &mut self,
        allocation: &AllocationId,
        execution_id: &str,
        role: Role,
        death: File,
    ) -> io::Result<Ticket> {
        let execution = self.engine.get(execution_id)?;
        if !matches!(execution.state, State::Starting | State::Running)
            || execution.cancel_actor.is_some()
        {
            return Err(conflict(
                "execution is not authorized for a new resident attachment",
            ));
        }
        let actor = &execution
            .submission
            .as_ref()
            .ok_or_else(|| conflict("resident attachment needs verified public actor context"))?
            .actor;
        let process = execution
            .process
            .ok_or_else(|| conflict("accepted execution has no registered process birth"))?;
        validate_pidfd(&death, &process)?;
        let maximum_recipients = self.maximum_recipients;
        let resident = self.resident_mut(allocation)?;
        if resident.record.key.actor != *actor {
            return Err(conflict("resident authorization domain differs"));
        }
        match role {
            Role::Writer
                if resident.record.phase == Phase::Filling && resident.recipients.is_empty() => {}
            Role::Reader if resident.record.phase == Phase::Ready => (),
            _ => return Err(conflict("generation is not attachable for this role")),
        }
        if resident.recipients.len() >= maximum_recipients {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "resident recipient capacity full; wait for actual release",
            ));
        }
        let ticket = Ticket {
            allocation: allocation.clone(),
            lease: uuid::Uuid::new_v4().to_string(),
            execution_id: execution_id.into(),
            process,
            role,
        };
        resident.recipients.insert(
            ticket.lease.clone(),
            Recipient {
                ticket: ticket.clone(),
                death,
            },
        );
        self.engine.notify_activity();
        Ok(ticket)
    }
    /// Publish only after the qualified fill writer released its writable mapping/FDs.
    /// Failed fills never become readable or reused as a completed generation.
    pub fn complete_fill(&mut self, completed: NativeCompletion) -> io::Result<()> {
        if completed.ticket.role != Role::Writer {
            return Err(invalid("fill completion does not identify its writer"));
        }
        let resident = self.resident_mut(&completed.ticket.allocation)?;
        if resident.record.phase != Phase::Filling {
            return Err(conflict("fill generation is not live"));
        }
        remove_exact(resident, &completed.ticket)?;
        resident.record.phase = Phase::Ready;
        self.engine.notify_activity();
        Ok(())
    }
    pub fn quarantine(&mut self, allocation: &AllocationId) -> io::Result<()> {
        self.resident_mut(allocation)?.record.phase = Phase::Quarantined;
        self.engine.notify_activity();
        Ok(())
    }
    /// Fence new leases, then ask existing holders to stop uses and release at a safe boundary.
    /// A progressing holder remains charged; this function has no kill/deadline policy.
    pub fn begin_revoke(&mut self, allocation: &AllocationId) -> io::Result<Vec<Ticket>> {
        let resident = self.resident_mut(allocation)?;
        if resident.record.phase == Phase::Releasing {
            return Err(conflict("allocation already awaits physical release"));
        }
        resident.record.phase = Phase::Revoking;
        let tickets = resident
            .recipients
            .values()
            .map(|r| r.ticket.clone())
            .collect();
        self.engine.notify_activity();
        Ok(tickets)
    }
    pub fn release_recipient(&mut self, completed: NativeCompletion) -> io::Result<()> {
        let resident = self.resident_mut(&completed.ticket.allocation)?;
        if resident.record.phase == Phase::Filling {
            return Err(conflict("writer must complete or quarantine the fill"));
        }
        remove_exact(resident, &completed.ticket)?;
        self.engine.notify_activity();
        Ok(())
    }
    /// Kernel death + the existing Engine's exact birth are authority; terminal status,
    /// cancellation, observer loss and telemetry silence alone cannot release a lease.
    pub fn reap_ended(&mut self) -> io::Result<Vec<Ticket>> {
        let mut ended = Vec::new();
        for resident in self.residents.values_mut() {
            let mut remove = Vec::new();
            for (id, recipient) in &resident.recipients {
                if crate::os::ended(&recipient.death) && process_ended(&recipient.ticket.process)? {
                    remove.push(id.clone())
                }
            }
            for id in remove {
                let recipient = resident.recipients.remove(&id).unwrap();
                if recipient.ticket.role == Role::Writer && resident.record.phase == Phase::Filling
                {
                    resident.record.phase = Phase::Quarantined;
                }
                ended.push(recipient.ticket);
            }
        }
        if !ended.is_empty() {
            self.engine.notify_activity()
        }
        Ok(ended)
    }
    /// Give the native device actor custody for physical release only after all recipients
    /// ended/released. Charge remains present until confirm_physical_release succeeds.
    pub fn take_for_release(&mut self, allocation: &AllocationId) -> io::Result<Resources> {
        let resident = self.resident_mut(allocation)?;
        if !matches!(resident.record.phase, Phase::Revoking | Phase::Quarantined)
            || !resident.recipients.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "resident remains live or has recipient obligations",
            ));
        }
        resident.record.phase = Phase::Releasing;
        let resources = resident
            .resources
            .take()
            .ok_or_else(|| conflict("native backing already handed to release actor"))?;
        self.engine.notify_activity();
        Ok(resources)
    }
    /// # Safety
    /// The qualified actor must have closed every owner/keeper/export reference, released
    /// native handles, and observed physical release. Logical counters, dropping an Arc
    /// or receiving an executor's CPU acknowledgement are insufficient evidence.
    pub unsafe fn confirm_physical_release(&mut self, allocation: &AllocationId) -> io::Result<()> {
        let resident = self.resident_mut(allocation)?;
        if resident.record.phase != Phase::Releasing
            || resident.resources.is_some()
            || !resident.recipients.is_empty()
        {
            return Err(conflict("allocation physical release is not pending"));
        }
        self.residents.remove(&allocation.id);
        self.engine.notify_activity();
        Ok(())
    }
    fn resident_mut(&mut self, id: &AllocationId) -> io::Result<&mut Resident> {
        let resident = self.residents.get_mut(&id.id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "resident generation does not exist",
            )
        })?;
        if resident.record.allocation != *id {
            return Err(conflict("resident owner epoch/generation differs"));
        }
        Ok(resident)
    }
}
fn remove_exact(resident: &mut Resident, ticket: &Ticket) -> io::Result<()> {
    if resident
        .recipients
        .get(&ticket.lease)
        .is_none_or(|r| r.ticket != *ticket)
    {
        return Err(conflict(
            "native release names another lease/process/generation",
        ));
    }
    resident.recipients.remove(&ticket.lease);
    Ok(())
}
fn validate_pidfd(death: &File, birth: &ProcessBirth) -> io::Result<()> {
    let fd = death.as_raw_fd();
    if fs::read_link(format!("/proc/self/fd/{fd}"))?.to_string_lossy() != "anon_inode:[pidfd]" {
        return Err(invalid("recipient custody requires a kernel pidfd"));
    }
    let info = fs::read_to_string(format!("/proc/self/fdinfo/{fd}"))?;
    let pid = info.lines().find_map(|line| {
        line.strip_prefix("Pid:")
            .and_then(|n| n.trim().parse::<u32>().ok())
    });
    if pid != Some(birth.pid) || crate::os::ended(death) || process_birth(birth.pid)? != *birth {
        return Err(conflict(
            "pidfd does not identify the registered live process birth",
        ));
    }
    Ok(())
}
