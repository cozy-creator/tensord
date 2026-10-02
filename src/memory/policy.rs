//! Per-GPU ledger and decisions. Pure: callers bring NVML samples and executor facts, carry out
//! the step returned, sample again and ask again.
use std::collections::{BTreeMap, BTreeSet};

pub const MIB: u64 = 1 << 20;
/// Allocator fragmentation beside a plane budget (Runtime `weight_policy.MARGIN`).
pub const MARGIN: u64 = 64 * MIB;
/// Free bytes kept on a GPU that drives a display (Runtime `weight_policy.DISPLAY_FLOOR`).
pub const DISPLAY_FLOOR: u64 = 512 * MIB;
/// Free bytes kept on any other GPU.
pub const HEADLESS_FLOOR: u64 = 256 * MIB;
/// A new executor's context before any executor on this GPU measured one.
pub const UNMEASURED_CONTEXT: u64 = 1 << 30;

/// One NVML reading of a GPU.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    pub total: u64,
    pub free: u64,
    pub display: bool,
    /// Device bytes NVML charges each process; None where it cannot attribute them.
    pub processes: Option<BTreeMap<u32, u64>>,
}

/// What an executor last reported of itself. None is unknown, never zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub context: Option<u64>,
    /// Context + torch reserved + the plane's own maps.
    pub process: Option<u64>,
    /// Weights as stages count them (decoded copies included), and their lowest rung.
    pub weights: Option<u64>,
    pub weights_floor: Option<u64>,
    pub activation: Option<u64>,
}
impl Facts {
    /// Newer facts replace older ones field by field; an absent field keeps what was known.
    pub fn merge(&mut self, newer: Facts) {
        self.context = newer.context.or(self.context);
        self.process = newer.process.or(self.process);
        self.weights = newer.weights.or(self.weights);
        self.weights_floor = newer.weights_floor.or(self.weights_floor);
        self.activation = match (self.activation, newer.activation) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => b.or(a),
        };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Spawned or loading: charged its reservation until it reports.
    Starting,
    Idle,
    /// In a call: charged its cap, which it may grow into.
    Active,
}

#[derive(Clone, Debug)]
pub struct Tenant {
    pub pid: u32,
    pub phase: Phase,
    pub last_used: u64,
    pub reserved: u64,
    pub cap: u64,
    /// Whether its plane holds device weights a `Budget{vram 0}` would free.
    pub mapped: bool,
}

/// One Degree 2 holding as custody reports it (pull): counted once on this GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holding {
    /// Custody's name for it: layout and generation.
    pub id: String,
    pub bytes: u64,
    /// Executor pids that map it.
    pub readers: Vec<u32>,
    pub idle_ms: u64,
    pub revoking: bool,
}

/// What the caller does next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// `Budget{vram 0}`: the idle executor's weights leave the GPU, its pinned tier keeps them.
    Unmap(String),
    /// End the idle executor for the device context it holds.
    End(String),
    /// Lower a running executor's process cap through its budget cell.
    Shrink(String, u64),
    /// Revoke a Degree 2 holding the planned call does not read.
    Revoke(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Step(Step),
    /// Room made: the cap (bytes the whole process may hold).
    Go(u64),
    /// Another call holds what is missing and gives it back when its turn ends.
    Wait,
}

/// One decision's memory of what it already asked, so no step repeats.
#[derive(Default)]
pub struct Round {
    tried: BTreeSet<String>,
}

#[derive(Default)]
pub struct Gpu {
    /// Configured minimum free bytes; the larger of it and the default floor applies.
    pub floor_bytes: u64,
    tenants: BTreeMap<String, Tenant>,
    /// Facts per plan, kept past its executor so a respawn starts from them.
    facts: BTreeMap<String, Facts>,
    /// Degree 2 holdings custody charges on this GPU, counted once; the caller refreshes them
    /// before each decision.
    pub holdings: Vec<Holding>,
    clock: u64,
}

impl Gpu {
    pub fn with_floor(floor_bytes: u64) -> Self {
        Self {
            floor_bytes,
            ..Self::default()
        }
    }

    pub fn floor(&self, sample: &Sample) -> u64 {
        let default = if sample.display {
            DISPLAY_FLOOR.max(sample.total / 16)
        } else {
            HEADLESS_FLOOR
        };
        default.max(self.floor_bytes)
    }

    pub fn tenant(&self, plan: &str) -> Option<&Tenant> {
        self.tenants.get(plan)
    }

    pub fn facts(&self, plan: &str) -> Facts {
        self.facts.get(plan).copied().unwrap_or_default()
    }

    /// A spawn admitted: charged `reserved` until it reports.
    pub fn starting(&mut self, plan: &str, reserved: u64) {
        self.clock += 1;
        self.tenants.insert(
            plan.into(),
            Tenant {
                pid: 0,
                phase: Phase::Starting,
                last_used: self.clock,
                reserved,
                cap: reserved,
                mapped: false,
            },
        );
    }

    pub fn spawned(&mut self, plan: &str, pid: u32) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.pid = pid;
        }
    }

    /// Facts from a reply. A loaded executor is charged what it holds from now on.
    pub fn observe(&mut self, plan: &str, facts: Facts, mapped: Option<bool>) {
        self.facts.entry(plan.into()).or_default().merge(facts);
        if let Some(tenant) = self.tenants.get_mut(plan) {
            if let Some(mapped) = mapped {
                tenant.mapped = mapped;
            }
            if tenant.phase == Phase::Starting && facts.process.is_some() {
                tenant.phase = Phase::Idle;
                tenant.reserved = 0;
            }
        }
    }

    pub fn active(&mut self, plan: &str, cap: u64) {
        self.clock += 1;
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.phase = Phase::Active;
            tenant.cap = cap;
            tenant.reserved = 0;
            tenant.last_used = self.clock;
            tenant.mapped = true;
        }
    }

    pub fn idle(&mut self, plan: &str) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.phase = Phase::Idle;
        }
    }

    pub fn set_cap(&mut self, plan: &str, cap: u64) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.cap = cap;
        }
    }

    pub fn unmapped(&mut self, plan: &str) {
        if let Some(tenant) = self.tenants.get_mut(plan) {
            tenant.mapped = false;
        }
    }

    pub fn ended(&mut self, plan: &str) {
        self.tenants.remove(plan);
    }

    pub fn resident(&self) -> u64 {
        self.holdings.iter().map(|h| h.bytes).sum()
    }

    /// Bytes `tenant` holds now: NVML's charge where it can say, else its report, else its
    /// reservation. With Degree 2 holdings on the GPU its report comes first: NVML charges a
    /// shared allocation to whichever process the driver picks, and custody counts it once.
    fn held(&self, plan: &str, tenant: &Tenant, sample: &Sample) -> u64 {
        let measured = sample
            .processes
            .as_ref()
            .and_then(|table| table.get(&tenant.pid).copied())
            .filter(|_| tenant.pid != 0);
        let reported = self.facts(plan).process;
        let held = if self.holdings.is_empty() {
            measured.or(reported)
        } else {
            reported.or(measured)
        };
        held.unwrap_or(0).max(tenant.reserved)
    }

    /// Bytes `tenant` may come to hold: a call grows into its cap, a spawn into its reservation.
    fn charged(&self, plan: &str, tenant: &Tenant, sample: &Sample) -> u64 {
        let held = self.held(plan, tenant, sample);
        match tenant.phase {
            Phase::Active => held.max(tenant.cap),
            Phase::Starting | Phase::Idle => held,
        }
    }

    /// Device bytes held by anything the machine does not account for: the desktop, other
    /// processes, library memory an executor did not report.
    pub fn external(&self, sample: &Sample) -> u64 {
        let ours: u64 = self
            .tenants
            .iter()
            .map(|(plan, tenant)| self.held(plan, tenant, sample))
            .sum::<u64>()
            + self.resident();
        (sample.total.saturating_sub(sample.free)).saturating_sub(ours)
    }

    /// What `plan`'s process may hold: the card less the floor, everything external, every
    /// other tenant's charge and the resident allocations.
    pub fn room(&self, plan: &str, sample: &Sample) -> u64 {
        let others: u64 = self
            .tenants
            .iter()
            .filter(|(other, _)| other.as_str() != plan)
            .map(|(other, tenant)| self.charged(other, tenant, sample))
            .sum();
        sample
            .total
            .saturating_sub(self.floor(sample) + self.external(sample) + others + self.resident())
    }

    /// A new executor's context: twice the largest measured here (its library workspaces
    /// grow as much again), else `UNMEASURED_CONTEXT`.
    pub fn context_estimate(&self) -> u64 {
        match self.facts.values().filter_map(|f| f.context).max() {
            Some(measured) if measured > 0 => 2 * measured + MARGIN,
            _ => UNMEASURED_CONTEXT,
        }
    }

    /// Everything resident: context, every weight byte, activations. None: not known yet.
    pub fn want(&self, plan: &str) -> Option<u64> {
        let facts = self.facts(plan);
        Some(
            facts.context.unwrap_or(self.context_estimate())
                + facts.weights?
                + facts.activation?
                + MARGIN,
        )
    }

    /// The lowest rung: context, the weights' floor, activations.
    pub fn need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        facts.context.unwrap_or(self.context_estimate())
            + facts.weights_floor.unwrap_or(0)
            + facts.activation.unwrap_or(0)
            + MARGIN
    }

    /// What a spawn of `plan` reserves: a context and its first working set when known.
    pub fn spawn_need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        self.context_estimate() + facts.weights_floor.unwrap_or(0) + facts.activation.unwrap_or(0)
    }

    /// The next step toward `want` (None: everything) and `need` bytes for `plan`, or its cap.
    /// Idle weights go first (LRU), then idle processes for their contexts while `need` is
    /// unmet, then running calls shrink toward their own lowest rung. Never refuses by size:
    /// with nothing left to free, the cap is what there is.
    pub fn decide(
        &self,
        plan: &str,
        want: Option<u64>,
        need: u64,
        sample: &Sample,
        round: &mut Round,
    ) -> Decision {
        let room = self.room(plan, sample);
        let others = || {
            let mut rows: Vec<_> = self
                .tenants
                .iter()
                .filter(|(other, _)| other.as_str() != plan && !round.tried.contains(*other))
                .collect();
            rows.sort_by_key(|(_, tenant)| tenant.last_used);
            rows
        };
        if want.is_none_or(|want| room < want) {
            if let Some((other, _)) = others()
                .into_iter()
                .find(|(_, t)| t.phase == Phase::Idle && t.mapped)
            {
                round.tried.insert(other.clone());
                return Decision::Step(Step::Unmap(other.clone()));
            }
            let reader = self.tenant(plan).map_or(0, |t| t.pid);
            let busy: BTreeSet<u32> = self
                .tenants
                .iter()
                .filter(|(other, t)| other.as_str() != plan && t.phase != Phase::Idle)
                .map(|(_, t)| t.pid)
                .collect();
            if let Some(holding) = self
                .holdings
                .iter()
                .filter(|h| {
                    !h.revoking
                        && !round.tried.contains(&h.id)
                        && !h
                            .readers
                            .iter()
                            .any(|pid| *pid == reader || busy.contains(pid))
                })
                .max_by_key(|h| h.idle_ms)
            {
                round.tried.insert(holding.id.clone());
                return Decision::Step(Step::Revoke(holding.id.clone()));
            }
        }
        if room >= need {
            return Decision::Go(room);
        }
        let short = need - room;
        if let Some((other, _)) = others().into_iter().find(|(_, t)| t.phase == Phase::Idle) {
            round.tried.insert(other.clone());
            return Decision::Step(Step::End(other.clone()));
        }
        if let Some((other, tenant)) = others()
            .into_iter()
            .find(|(other, t)| t.phase == Phase::Active && t.cap > self.need(other))
        {
            round.tried.insert(other.clone());
            let lowered = tenant.cap.saturating_sub(short).max(self.need(other));
            return Decision::Step(Step::Shrink(other.clone(), lowered));
        }
        if self
            .tenants
            .iter()
            .any(|(other, t)| other.as_str() != plan && t.phase != Phase::Idle)
        {
            return Decision::Wait;
        }
        Decision::Go(room)
    }

    /// A sample below the floor during a call: the running tenant's lowered cap, never below
    /// its lowest rung. None while the floor holds or nobody runs.
    pub fn below_floor(&self, sample: &Sample) -> Option<(String, u64)> {
        let floor = self.floor(sample);
        if sample.free >= floor {
            return None;
        }
        let deficit = floor - sample.free + MARGIN;
        self.tenants
            .iter()
            .filter(|(_, t)| t.phase == Phase::Active)
            .map(|(plan, t)| {
                (
                    plan,
                    t.cap.saturating_sub(deficit).max(self.need(plan)),
                    t.cap,
                )
            })
            .find(|(_, lowered, cap)| lowered < cap)
            .map(|(plan, lowered, _)| (plan.clone(), lowered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    fn sample(total: u64, free: u64, processes: &[(u32, u64)]) -> Sample {
        Sample {
            total,
            free,
            display: false,
            processes: Some(processes.iter().copied().collect()),
        }
    }
    fn loaded(gpu: &mut Gpu, plan: &str, pid: u32, weights: u64, activation: u64) {
        gpu.starting(plan, gpu.spawn_need(plan));
        gpu.spawned(plan, pid);
        gpu.observe(
            plan,
            Facts {
                context: Some(GIB / 2),
                process: Some(GIB / 2),
                weights: Some(weights),
                weights_floor: Some(weights / 8),
                activation: Some(activation),
            },
            Some(false),
        );
    }

    #[test]
    fn a_display_gpu_keeps_its_floor_and_the_desktop_counts_as_external() {
        let gpu = Gpu::default();
        let mut s = sample(8 * GIB, 8 * GIB - 166 * MIB, &[]);
        s.display = true;
        assert_eq!(gpu.floor(&s), DISPLAY_FLOOR);
        assert_eq!(gpu.external(&s), 166 * MIB);
        assert_eq!(gpu.room("sdxl", &s), 8 * GIB - DISPLAY_FLOOR - 166 * MIB);
    }

    #[test]
    fn a_switch_unmaps_the_idle_tenant_then_grants_what_is_left() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.active("sdxl", 15 * GIB);
        gpu.idle("sdxl");
        loaded(&mut gpu, "anima", 11, 6 * GIB, 3 * GIB);
        // SDXL idle holds 8 GiB (context + weights); Anima its context.
        let s = sample(
            16 * GIB,
            16 * GIB - 8 * GIB - GIB / 2,
            &[(10, 8 * GIB), (11, GIB / 2)],
        );
        let mut round = Round::default();
        let want = gpu.want("anima");
        let need = gpu.need("anima");
        assert_eq!(
            gpu.decide("anima", want, need, &s, &mut round),
            Decision::Step(Step::Unmap("sdxl".into()))
        );
        gpu.unmapped("sdxl");
        let s = sample(16 * GIB, 15 * GIB, &[(10, GIB / 2), (11, GIB / 2)]);
        assert_eq!(
            gpu.decide("anima", want, need, &s, &mut round),
            Decision::Go(16 * GIB - HEADLESS_FLOOR - GIB / 2)
        );
    }

    #[test]
    fn idle_processes_end_only_for_the_lowest_rung() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "a", 10, 2 * GIB, GIB);
        loaded(&mut gpu, "b", 11, 2 * GIB, GIB);
        // A third model's lowest rung does not fit beside two idle contexts on a 4 GiB card.
        gpu.observe(
            "c",
            Facts {
                context: Some(GIB),
                weights: Some(GIB),
                weights_floor: Some(GIB),
                activation: Some(GIB / 2),
                ..Facts::default()
            },
            None,
        );
        gpu.starting("c", 0);
        let s = sample(4 * GIB, 3 * GIB - 512 * MIB, &[(10, GIB / 2), (11, GIB)]);
        let mut round = Round::default();
        assert_eq!(
            gpu.decide("c", gpu.want("c"), gpu.need("c"), &s, &mut round),
            Decision::Step(Step::End("a".into()))
        );
    }

    #[test]
    fn a_spawn_beside_a_running_call_shrinks_it_to_its_lowest_rung_at_most() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.active("sdxl", 15 * GIB);
        let s = sample(16 * GIB, 4 * GIB, &[(10, 12 * GIB)]);
        let mut round = Round::default();
        let need = 2 * GIB;
        let Decision::Step(Step::Shrink(plan, cap)) =
            gpu.decide("anima", Some(need), need, &s, &mut round)
        else {
            panic!("expected a shrink")
        };
        assert_eq!(plan, "sdxl");
        assert!(cap >= gpu.need("sdxl") && cap < 15 * GIB);
        gpu.set_cap("sdxl", cap);
        assert_eq!(
            gpu.decide("anima", Some(need), need, &s, &mut round),
            if gpu.room("anima", &s) >= need {
                Decision::Go(gpu.room("anima", &s))
            } else {
                Decision::Wait
            }
        );
    }

    #[test]
    fn a_holding_nobody_running_reads_is_revoked_before_any_process_ends() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "anima", 11, 6 * GIB, 3 * GIB);
        gpu.holdings = vec![
            Holding {
                id: "sdxl#1".into(),
                bytes: 7 * GIB,
                readers: vec![10],
                idle_ms: 5_000,
                revoking: false,
            },
            Holding {
                id: "anima#1".into(),
                bytes: 6 * GIB,
                readers: vec![11],
                idle_ms: 9_000,
                revoking: false,
            },
        ];
        let s = sample(16 * GIB, 2 * GIB, &[(11, GIB / 2)]);
        let mut round = Round::default();
        assert_eq!(
            gpu.decide(
                "anima",
                gpu.want("anima"),
                gpu.need("anima"),
                &s,
                &mut round
            ),
            Decision::Step(Step::Revoke("sdxl#1".into()))
        );
    }

    #[test]
    fn nothing_left_to_free_grants_what_there_is() {
        let gpu = Gpu::default();
        let s = sample(4 * GIB, GIB, &[]);
        let mut round = Round::default();
        assert_eq!(
            gpu.decide("big", Some(20 * GIB), 10 * GIB, &s, &mut round),
            Decision::Go(GIB - HEADLESS_FLOOR)
        );
    }

    #[test]
    fn a_squeeze_lowers_the_running_cap_by_the_deficit() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.active("sdxl", 15 * GIB);
        let s = sample(16 * GIB, 100 * MIB, &[(10, 12 * GIB)]);
        let (plan, cap) = gpu.below_floor(&s).expect("floor breached");
        assert_eq!(plan, "sdxl");
        assert_eq!(cap, 15 * GIB - (HEADLESS_FLOOR - 100 * MIB + MARGIN));
        assert!(gpu
            .below_floor(&sample(16 * GIB, GIB, &[(10, 12 * GIB)]))
            .is_none());
    }
}
