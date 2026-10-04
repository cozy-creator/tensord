//! Per-GPU ledger and decisions. Pure: callers bring NVML samples and executor facts, carry out
//! the step returned, sample again and ask again.
use super::learned::Learned;
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
    /// Host bytes its pinned tier holds, and the budget that bounds it.
    pub pinned: Option<u64>,
    pub pinned_budget: Option<u64>,
}
impl Facts {
    /// Newer facts replace older ones field by field; an absent field keeps what was known.
    pub fn merge(&mut self, newer: Facts) {
        self.context = newer.context.or(self.context);
        self.process = newer.process.or(self.process);
        self.weights = newer.weights.or(self.weights);
        self.weights_floor = newer.weights_floor.or(self.weights_floor);
        self.pinned = newer.pinned.or(self.pinned);
        self.pinned_budget = newer.pinned_budget.or(self.pinned_budget);
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

/// A holding's name without its generation (`<name>#<generation>`): one weight set on one GPU.
pub fn holding_name(id: &str) -> &str {
    id.rsplit_once('#').map_or(id, |(name, _)| name)
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
    /// What executors measured here and in earlier runs (contexts, activations per shape).
    pub learned: Learned,
    /// This GPU and driver, the key its contexts are learned under.
    pub device: String,
    /// Each plan's next request shape (PrepareRequest features), for its learned activations.
    shapes: BTreeMap<String, String>,
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

    /// `plan`'s facts as measured this run; before its first load here, its weights as an
    /// earlier run's load measured them.
    pub fn facts(&self, plan: &str) -> Facts {
        let mut facts = self.facts.get(plan).copied().unwrap_or_default();
        if let Some(learned) = self.learned.plans.get(plan) {
            let known = |bytes: u64| (bytes > 0).then_some(bytes);
            facts.weights = facts.weights.or(known(learned.weights));
            facts.weights_floor = facts.weights_floor.or(known(learned.weights_floor));
        }
        facts
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
        self.actual(plan, tenant, sample).max(tenant.reserved)
    }

    /// Bytes `tenant` is known to hold now: NVML's charge or its report, never a reservation
    /// (a spawn's reservation is room it may take, not memory anyone holds yet).
    fn actual(&self, plan: &str, tenant: &Tenant, sample: &Sample) -> u64 {
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
        held.unwrap_or(0)
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
        // Reservations are not subtracted: a starting executor's would hide that much foreign
        // memory (F, rtx-3070: a 2.16 GB spawn reservation hid a 2.15 GB pool ballast, and
        // Degree 2 engaged on a card it did not fit). What a spawn allocates before it
        // reports counts as external meanwhile: conservative, and brief.
        let ours: u64 = self
            .tenants
            .iter()
            .map(|(plan, tenant)| self.actual(plan, tenant, sample))
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

    /// A new executor's context: the largest an executor measured on this GPU and driver,
    /// here or in an earlier run (taken after its libraries' first launches), else twice the
    /// largest measured this run, else `UNMEASURED_CONTEXT`.
    pub fn context_estimate(&self) -> u64 {
        if let Some(learned) = self.learned.contexts.get(&self.device).filter(|c| **c > 0) {
            return learned + MARGIN;
        }
        match self.facts.values().filter_map(|f| f.context).max() {
            Some(measured) if measured > 0 => 2 * measured + MARGIN,
            _ => UNMEASURED_CONTEXT,
        }
    }

    /// The request shape `plan` runs next.
    pub fn set_shape(&mut self, plan: &str, shape: &str) {
        self.shapes.insert(plan.into(), shape.into());
    }

    /// `plan`'s activation growth for its next shape as learned; before its shape is known
    /// (a load) or for a shape never measured, the largest any of its shapes measured, here
    /// or in an earlier run.
    pub fn activation(&self, plan: &str) -> Option<u64> {
        let shaped = self
            .shapes
            .get(plan)
            .and_then(|shape| self.learned.shape(plan, shape))
            .map(super::learned::Shape::bytes)
            .filter(|peak| *peak > 0);
        shaped.or_else(|| {
            [self.facts(plan).activation, self.learned.peak(plan)]
                .into_iter()
                .flatten()
                .max()
        })
    }

    /// Per-method activation growth learned for `plan`'s next shape: the executor's seeds.
    pub fn seeds(&self, plan: &str) -> BTreeMap<String, u64> {
        self.shapes
            .get(plan)
            .and_then(|shape| self.learned.shape(plan, shape))
            .map(|measured| measured.methods.clone())
            .unwrap_or_default()
    }

    /// Holdings `plan`'s executor maps (Degree 2): weights already on the GPU, counted once
    /// among the resident allocations, never again in what it asks for.
    fn attached(&self, plan: &str) -> u64 {
        let Some(pid) = self.tenant(plan).map(|t| t.pid).filter(|pid| *pid != 0) else {
            return 0;
        };
        self.holdings
            .iter()
            .filter(|h| h.readers.contains(&pid))
            .map(|h| h.bytes)
            .sum()
    }

    /// Holdings `plan`'s next executor attaches although another tenant reads them: named by
    /// an earlier executor of it (learned), held now, and not mapped by its own process. They
    /// are resident already, so its load adds none of their bytes.
    fn shared(&self, plan: &str) -> u64 {
        let Some(names) = self
            .learned
            .plans
            .get(plan)
            .map(|learned| &learned.holdings)
        else {
            return 0;
        };
        let own = self.tenant(plan).map_or(0, |t| t.pid);
        self.holdings
            .iter()
            .filter(|h| {
                !h.revoking
                    && !h.readers.is_empty()
                    && !h.readers.contains(&own)
                    && names.contains(holding_name(&h.id))
            })
            .map(|h| h.bytes)
            .sum()
    }

    /// Whether `plan`'s whole construction and its activations fit beside the other tenants
    /// (Degree 2 keeps every component resident). Holdings nobody reads are reclaimable or its
    /// own to attach, and those another tenant reads of its own weight sets it attaches too.
    /// False while its weights or activations were never measured.
    pub fn fits_resident(&self, plan: &str, sample: &Sample) -> bool {
        let unread: u64 = self
            .holdings
            .iter()
            .filter(|h| h.readers.is_empty())
            .map(|h| h.bytes)
            .sum();
        self.want(plan)
            .is_some_and(|want| self.room(plan, sample) + unread + self.shared(plan) >= want)
    }

    /// Everything resident beside what it already maps: context, every weight byte,
    /// activations. None: not known yet.
    pub fn want(&self, plan: &str) -> Option<u64> {
        let facts = self.facts(plan);
        Some(
            facts.context.unwrap_or(self.context_estimate())
                + facts.weights?.saturating_sub(self.attached(plan))
                + self.activation(plan)?
                + MARGIN,
        )
    }

    /// The lowest rung: context, the weights' floor beside what it maps, activations.
    pub fn need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        facts.context.unwrap_or(self.context_estimate())
            + facts
                .weights_floor
                .unwrap_or(0)
                .saturating_sub(self.attached(plan))
            + self.activation(plan).unwrap_or(0)
            + MARGIN
    }

    /// What a spawn of `plan` reserves: a context and its first working set when known.
    pub fn spawn_need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        self.context_estimate()
            + facts.weights_floor.unwrap_or(0)
            + self.activation(plan).unwrap_or(0)
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

    /// The least recently used idle tenant other than `plan`: what the host gives up first.
    pub fn lru_idle(&self, plan: &str) -> Option<String> {
        self.tenants
            .iter()
            .filter(|(other, t)| other.as_str() != plan && t.phase == Phase::Idle)
            .min_by_key(|(_, t)| t.last_used)
            .map(|(other, _)| other.clone())
    }

    /// Every live tenant as `(plan, weights, pinned now)`, `first` ahead and then most
    /// recently used first: the order host pinned budgets are shared in.
    pub fn pinned_order(&self, first: &str) -> Vec<(String, u64, u64)> {
        let mut rows: Vec<_> = self.tenants.iter().collect();
        rows.sort_by_key(|(plan, t)| (plan.as_str() != first, std::cmp::Reverse(t.last_used)));
        let mut order: Vec<_> = rows
            .into_iter()
            .map(|(plan, _)| {
                let facts = self.facts(plan);
                (
                    plan.clone(),
                    facts.weights.unwrap_or(0),
                    facts.pinned.unwrap_or(0),
                )
            })
            .collect();
        if !self.tenants.contains_key(first) {
            let facts = self.facts(first);
            order.insert(0, (first.into(), facts.weights.unwrap_or(0), 0));
        }
        order
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
                ..Facts::default()
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
    fn weights_a_plan_already_maps_are_counted_once() {
        // 24 GiB card, Degree 2: SDXL (7 GiB) and Anima (6 GiB) both kept by custody, each
        // read by its own idle executor (0.5 GiB of process each).
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB / 2);
        loaded(&mut gpu, "anima", 11, 6 * GIB, 3 * GIB);
        let held = |id: &str, bytes, reader| Holding {
            id: id.into(),
            bytes,
            readers: vec![reader],
            idle_ms: 1_000,
            revoking: false,
        };
        gpu.holdings = vec![held("sdxl#1", 7 * GIB, 10), held("anima#1", 6 * GIB, 11)];
        let s = sample(
            24 * GIB,
            24 * GIB - 14 * GIB,
            &[(10, GIB / 2), (11, GIB / 2)],
        );
        // Anima asks for its context and activations only: its weights are resident.
        assert_eq!(gpu.want("anima"), Some(GIB / 2 + 3 * GIB + MARGIN));
        let mut round = Round::default();
        let room = gpu.room("anima", &s);
        assert_eq!(
            gpu.decide(
                "anima",
                gpu.want("anima"),
                gpu.need("anima"),
                &s,
                &mut round
            ),
            Decision::Go(room),
            "nothing of SDXL's is touched"
        );
        // Its cap still leaves every holding in place: room counts them all.
        assert_eq!(room, 24 * GIB - HEADLESS_FLOOR - GIB / 2 - 13 * GIB);
    }

    #[test]
    fn a_weight_set_another_tenant_reads_is_not_counted_again_at_the_fit() {
        // 16 GiB card: an SDXL checkpoint (7 GiB) held and read by one package's idle executor.
        // A second package binds the same checkpoint; its earlier executor named that set.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "txt2img", 10, 7 * GIB, 3 * GIB);
        gpu.holdings = vec![Holding {
            id: "GPU-1/sha256:sdxl#4".into(),
            bytes: 7 * GIB,
            readers: vec![10],
            idle_ms: 1_000,
            revoking: false,
        }];
        gpu.learned.load("img2img", 7 * GIB, GIB);
        gpu.learned.call(
            "img2img",
            "height=1024,width=1024",
            3 * GIB,
            &BTreeMap::new(),
        );
        let s = sample(16 * GIB, 16 * GIB - 7 * GIB - GIB / 2, &[(10, GIB / 2)]);
        // 8.25 GiB of room; context, 7 GiB of weights and 3 GiB of activations would not fit.
        assert!(gpu.want("img2img").unwrap() > gpu.room("img2img", &s));
        assert!(!gpu.fits_resident("img2img", &s));
        // It attaches the held set: only its context and activations are new bytes.
        assert!(gpu
            .learned
            .holding("img2img", holding_name("GPU-1/sha256:sdxl#3")));
        assert!(gpu.fits_resident("img2img", &s));
        // A set being revoked, or one of another checkpoint, is no help.
        gpu.holdings[0].revoking = true;
        assert!(!gpu.fits_resident("img2img", &s));
        gpu.holdings[0].revoking = false;
        gpu.holdings[0].id = "GPU-1/sha256:other#1".into();
        assert!(!gpu.fits_resident("img2img", &s));
    }

    #[test]
    fn a_spawn_reservation_never_hides_foreign_memory() {
        // rtx-3070 with a 6 GiB pool: 2 GiB of ballast, no per-process NVML in the container.
        // Anima (5.4 GiB of weights, 1.4 GiB of activations) was measured in an earlier run.
        let mut gpu = Gpu::default();
        gpu.learned.load("anima", 5 * GIB + 2 * GIB / 5, 300 * MIB);
        gpu.learned
            .call("anima", "height=1024,width=1024", 7 * GIB / 5, &BTreeMap::new());
        let reserved = gpu.spawn_need("anima");
        gpu.starting("anima", reserved);
        let s = Sample {
            total: 8 * GIB,
            free: 6 * GIB,
            display: false,
            processes: None,
        };
        assert!(reserved > GIB);
        assert_eq!(gpu.external(&s), 2 * GIB);
        assert_eq!(gpu.room("anima", &s), 6 * GIB - HEADLESS_FLOOR);
        assert!(!gpu.fits_resident("anima", &s));
        // Another tenant still sees the spawn's reservation as taken.
        assert_eq!(gpu.room("sdxl", &s), 6 * GIB - HEADLESS_FLOOR - reserved);
    }

    #[test]
    fn a_learned_peak_for_the_shape_makes_room_before_the_call() {
        let mut gpu = Gpu {
            device: "GPU-1/580".into(),
            ..Gpu::default()
        };
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB / 4);
        gpu.active("sdxl", 15 * GIB);
        gpu.idle("sdxl");
        loaded(&mut gpu, "anima", 11, 6 * GIB, GIB / 4);
        // Anima's 1536 shape was measured at 3 GiB in an earlier run; this run knows 256 MiB.
        let methods = BTreeMap::from([("transformer".to_string(), 3 * GIB)]);
        gpu.learned
            .call("anima", "height=1536,width=1536", 3 * GIB, &methods);
        gpu.learned.context("GPU-1/580", 300 * MIB);
        gpu.set_shape("anima", "height=1536,width=1536");
        assert_eq!(gpu.activation("anima"), Some(3 * GIB));
        assert_eq!(gpu.seeds("anima"), methods);
        assert_eq!(gpu.context_estimate(), 300 * MIB + MARGIN);
        // SDXL idle holds 8 GiB of a 16 GiB card: 7.75 GiB of room is short of Anima's want.
        let s = sample(16 * GIB, 7 * GIB + GIB / 2, &[(10, 8 * GIB), (11, GIB / 2)]);
        let mut round = Round::default();
        assert_eq!(
            gpu.decide(
                "anima",
                gpu.want("anima"),
                gpu.need("anima"),
                &s,
                &mut round
            ),
            Decision::Step(Step::Unmap("sdxl".into()))
        );
    }

    #[test]
    fn a_restarted_machine_knows_a_plans_whole_want_before_its_first_load() {
        let mut gpu = Gpu {
            device: "GPU-1/580".into(),
            ..Gpu::default()
        };
        assert_eq!(gpu.want("sdxl"), None);
        // An earlier run loaded SDXL (7 GiB) and ran two shapes; no shape is known at a load.
        // At 1024 its call peaked at 0.5 GiB, but its decode reserved 3 GiB.
        gpu.learned.load("sdxl", 7 * GIB, 2 * GIB);
        let decode = BTreeMap::from([("decode".to_string(), 3 * GIB)]);
        gpu.learned
            .call("sdxl", "height=1024,width=1024", GIB / 2, &decode);
        gpu.learned
            .call("sdxl", "height=512,width=512", GIB / 4, &BTreeMap::new());
        gpu.learned.context("GPU-1/580", 300 * MIB);
        assert_eq!(gpu.activation("sdxl"), Some(3 * GIB));
        assert_eq!(
            gpu.want("sdxl"),
            Some(300 * MIB + MARGIN + 7 * GIB + 3 * GIB + MARGIN)
        );
        assert_eq!(
            gpu.need("sdxl"),
            300 * MIB + MARGIN + 2 * GIB + 3 * GIB + MARGIN
        );
        // The next shape, once known, takes its own measurement.
        gpu.set_shape("sdxl", "height=512,width=512");
        assert_eq!(gpu.activation("sdxl"), Some(GIB / 4));
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
