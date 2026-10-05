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
    /// Revoke only enough of a holding's regions for these bytes; the rest stays attachable.
    Trim(String, u64),
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
    /// Plans in a caller's warm set: among idle tenants, the last to give anything up.
    pub members: BTreeSet<String>,
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

    /// What `plan`'s process holds now plus everything free above the floor: what a running
    /// call that asked for room may grow into although the ledger's charges for idle tenants
    /// (their last reports) say less is free. The floor watchdog still guards the card.
    pub fn physical_cap(&self, plan: &str, sample: &Sample) -> u64 {
        let held = self
            .tenant(plan)
            .map_or(0, |tenant| self.actual(plan, tenant, sample));
        held + sample.free.saturating_sub(self.floor(sample))
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
        if let Some(learned) = self
            .learned
            .process_contexts
            .get(&self.device)
            .filter(|c| **c > 0)
        {
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
            .map(|shape| shape.bytes())
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
        self.shape(plan).map(|shape| shape.methods).unwrap_or_default()
    }

    /// Rooms known to squeeze `plan`'s stage methods at its next shape: the executor's seeds.
    pub fn squeezed(&self, plan: &str) -> BTreeMap<String, u64> {
        self.shape(plan)
            .map(|shape| shape.squeezed)
            .unwrap_or_default()
    }

    /// What was learned for `plan`'s next shape, measured or estimated (`estimated_from`).
    pub fn shape(&self, plan: &str) -> Option<super::learned::Shape> {
        self.shapes
            .get(plan)
            .and_then(|shape| self.learned.shape(plan, shape))
    }

    /// Whether `holding` is `plan`'s own: mapped by its process, or, while its executor has
    /// not loaded yet, named by an earlier executor of it and held now (it attaches them at
    /// its load). Its bytes are on the GPU already: counted once among the resident
    /// allocations, never again in what `plan` asks for, and never revoked for `plan`'s own
    /// call (phil, 12 GB: each Anima grant revoked 1.79 GB of Anima's own idle holdings, then
    /// loaded them again). A loaded executor that let a holding go never maps it again: then
    /// it is not its own (huaisang: counted as its own, it priced Anima's want at 1.1 GB).
    fn own(&self, plan: &str, holding: &Holding) -> bool {
        let tenant = self.tenant(plan);
        let pid = tenant.map_or(0, |t| t.pid);
        (pid != 0 && holding.readers.contains(&pid))
            || (!holding.revoking
                && tenant.is_none_or(|t| t.phase == Phase::Starting)
                && self
                    .learned
                    .plans
                    .get(plan)
                    .is_some_and(|learned| learned.holdings.contains(holding_name(&holding.id))))
    }

    /// Whether a grant for `plan` may revoke or trim `holding`: another plan's, read by no
    /// call in progress.
    fn reclaimable(&self, plan: &str, holding: &Holding) -> bool {
        !holding.revoking
            && !self.own(plan, holding)
            && !holding.readers.iter().any(|pid| {
                self.tenants
                    .iter()
                    .any(|(other, t)| other != plan && t.pid == *pid && t.phase != Phase::Idle)
            })
    }

    fn own_bytes(&self, plan: &str) -> u64 {
        self.holdings
            .iter()
            .filter(|h| self.own(plan, h))
            .map(|h| h.bytes)
            .sum()
    }

    /// Whether `plan`'s whole construction and its activations fit beside the other tenants
    /// (Degree 2 keeps every component resident). Its own holdings are in `want` already;
    /// what its grant may revoke or trim (`reclaimable`: idle tenants' too) is room. Counting
    /// only holdings nobody read, two models that do not both fit on a 12 GB card each
    /// detached in turn and reloaded whole at every switch (huaisang). False while its
    /// weights or activations were never measured.
    pub fn fits_resident(&self, plan: &str, sample: &Sample) -> bool {
        let reclaimable: u64 = self
            .holdings
            .iter()
            .filter(|h| self.reclaimable(plan, h))
            .map(|h| h.bytes)
            .sum();
        self.want(plan)
            .is_some_and(|want| self.room(plan, sample) + reclaimable >= want)
    }

    /// The weights a call of `plan` puts on the GPU: what one kept mapped with nothing evicted
    /// (an entrypoint may use only part of its construction), else every weight it registers.
    fn call_weights(&self, plan: &str) -> Option<u64> {
        let mapped = self.learned.plans.get(plan).map(|learned| learned.mapped);
        mapped.filter(|m| *m > 0).or(self.facts(plan).weights)
    }

    /// Everything resident beside what it already maps: context, the weights its calls map,
    /// activations. None: not known yet.
    pub fn want(&self, plan: &str) -> Option<u64> {
        let facts = self.facts(plan);
        Some(
            facts.context.unwrap_or(self.context_estimate())
                + self.call_weights(plan)?.saturating_sub(self.own_bytes(plan))
                + self.activation(plan)?
                + MARGIN,
        )
    }

    /// What a grant makes room for: `want`, or for a plan whose activations were never
    /// measured, its weights and the largest activations any plan measured on this GPU. An
    /// unmeasured call never empties a card that has room for it (C, A40: a second package's
    /// first call revoked every holding with 38.7 GB free).
    pub fn grant_want(&self, plan: &str) -> Option<u64> {
        self.want(plan).or_else(|| {
            let facts = self.facts(plan);
            let largest = self
                .learned
                .plans
                .keys()
                .filter_map(|other| self.learned.peak(other))
                .chain(self.facts.values().filter_map(|f| f.activation))
                .max()?;
            Some(
                facts.context.unwrap_or(self.context_estimate())
                    + self.call_weights(plan)?.saturating_sub(self.own_bytes(plan))
                    + largest
                    + MARGIN,
            )
        })
    }

    /// The lowest rung: context, the weights' floor beside what it maps, activations.
    pub fn need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        facts.context.unwrap_or(self.context_estimate())
            + facts
                .weights_floor
                .unwrap_or(0)
                .saturating_sub(self.own_bytes(plan))
            + self.activation(plan).unwrap_or(0)
            + MARGIN
    }

    /// What a spawn of `plan` reserves: a context and its first working set when known. The
    /// weights it attaches are resident already, so the spawn asks no room for them (H3 on
    /// an RTX PRO 6000: 79.9 GB held and 21.8 GB free; a replacement's 22.6 GB with its
    /// 1.5 GB floor would have revoked a holding it was about to map).
    pub fn spawn_need(&self, plan: &str) -> u64 {
        let facts = self.facts(plan);
        self.context_estimate()
            + facts
                .weights_floor
                .unwrap_or(0)
                .saturating_sub(self.own_bytes(plan))
            + self.activation(plan).unwrap_or(0)
    }

    /// A holding `plan`'s executor named that nobody maps now, least recently used first:
    /// what it let go to make room comes back before any other tenant gives anything.
    pub fn own_unread(&self, plan: &str, round: &mut Round) -> Option<Step> {
        let names = &self.learned.plans.get(plan)?.holdings;
        let holding = self
            .holdings
            .iter()
            .filter(|h| {
                !h.revoking
                    && h.readers.is_empty()
                    && !round.tried.contains(&h.id)
                    && names.contains(holding_name(&h.id))
            })
            .max_by_key(|h| h.idle_ms)?;
        round.tried.insert(holding.id.clone());
        Some(Step::Revoke(holding.id.clone()))
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
            rows.sort_by_key(|(other, tenant)| (self.members.contains(*other), tenant.last_used));
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
            if let Some(holding) = self
                .holdings
                .iter()
                .filter(|h| !round.tried.contains(&h.id) && self.reclaimable(plan, h))
                .max_by_key(|h| h.idle_ms)
            {
                round.tried.insert(holding.id.clone());
                // Only the shortfall goes: a 12 GB card switching SDXL and Anima revoked the
                // UNet's 5.3 GB for a 1.6 GB one, and loaded it all again at the next switch.
                return Decision::Step(match want.map(|want| want - room) {
                    Some(short) if short < holding.bytes => Step::Trim(holding.id.clone(), short),
                    _ => Step::Revoke(holding.id.clone()),
                });
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

    /// The idle tenant other than `plan` the host gives up first: one outside every warm
    /// set before a member, least recently used first. `spare`: never a member.
    pub fn lru_idle(&self, plan: &str, spare: bool) -> Option<String> {
        self.tenants
            .iter()
            .filter(|(other, t)| other.as_str() != plan && t.phase == Phase::Idle)
            .filter(|(other, _)| !(spare && self.members.contains(*other)))
            .min_by_key(|(other, t)| (self.members.contains(*other), t.last_used))
            .map(|(other, _)| other.clone())
    }

    /// The next step to admit warm set member `plan`'s executor: the room that is free, else
    /// what idle tenants outside every warm set give, least recently used first (their
    /// weights, then their processes). `Wait`: only a member or a running call has more.
    pub fn admit_member(&self, plan: &str, sample: &Sample, round: &mut Round) -> Decision {
        let room = self.room(plan, sample);
        if room >= self.spawn_need(plan) {
            return Decision::Go(room);
        }
        let mut idle: Vec<_> = self
            .tenants
            .iter()
            .filter(|(other, t)| other.as_str() != plan && t.phase == Phase::Idle)
            .filter(|(other, _)| !self.members.contains(*other))
            .collect();
        idle.sort_by_key(|(_, tenant)| tenant.last_used);
        for (other, tenant) in &idle {
            if tenant.mapped && round.tried.insert(other.to_string()) {
                return Decision::Step(Step::Unmap(other.to_string()));
            }
        }
        for (other, _) in idle {
            if round.tried.insert(format!("end:{other}")) {
                return Decision::Step(Step::End(other.clone()));
            }
        }
        Decision::Wait
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

    /// Two idle tenants on a 4 GiB card, "member" the less recently used, and a third
    /// model whose lowest rung does not fit beside both contexts.
    fn two_idle_and_a_third(members: &[&str]) -> (Gpu, Sample) {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "member", 10, 2 * GIB, GIB);
        loaded(&mut gpu, "other", 11, 2 * GIB, GIB);
        gpu.members = members.iter().map(|plan| plan.to_string()).collect();
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
        (gpu, s)
    }

    #[test]
    fn a_request_takes_from_tenants_outside_the_warm_set_before_a_member() {
        let (mut gpu, s) = two_idle_and_a_third(&["member"]);
        let decide = |gpu: &Gpu| {
            gpu.decide("c", gpu.want("c"), gpu.need("c"), &s, &mut Round::default())
        };
        // By the clock alone the member, the older of the two, would go first.
        assert_eq!(decide(&gpu), Decision::Step(Step::End("other".into())));
        gpu.observe("member", Facts::default(), Some(true));
        gpu.observe("other", Facts::default(), Some(true));
        assert_eq!(decide(&gpu), Decision::Step(Step::Unmap("other".into())));
        // Work is never refused: with nothing else idle, the member gives its room too.
        gpu.ended("other");
        assert_eq!(decide(&gpu), Decision::Step(Step::Unmap("member".into())));
        assert_eq!(gpu.lru_idle("c", false), Some("member".into()));
        assert_eq!(gpu.lru_idle("c", true), None);
    }

    #[test]
    fn a_member_is_admitted_into_free_room_and_the_room_of_tenants_outside_the_set_only() {
        // "c" joins the set: the idle tenant outside it gives its weights, then its process.
        let (mut gpu, s) = two_idle_and_a_third(&["member", "c"]);
        gpu.observe("other", Facts::default(), Some(true));
        let mut round = Round::default();
        assert_eq!(gpu.admit_member("c", &s, &mut round), Decision::Step(Step::Unmap("other".into())));
        gpu.unmapped("other");
        assert_eq!(gpu.admit_member("c", &s, &mut round), Decision::Step(Step::End("other".into())));
        // Another member is never touched for it: it waits at a lower level.
        gpu.ended("other");
        assert_eq!(gpu.admit_member("c", &s, &mut Round::default()), Decision::Wait);
        // With the room free it is simply admitted.
        gpu.ended("member");
        let free = sample(8 * GIB, 8 * GIB, &[]);
        assert!(matches!(gpu.admit_member("c", &free, &mut Round::default()), Decision::Go(_)));
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
        let short = gpu.want("anima").unwrap() - gpu.room("anima", &s);
        assert_eq!(
            gpu.decide(
                "anima",
                gpu.want("anima"),
                gpu.need("anima"),
                &s,
                &mut round
            ),
            Decision::Step(Step::Trim("sdxl#1".into(), short))
        );
    }

    #[test]
    fn a_grant_never_revokes_what_the_planned_plans_executor_maps() {
        // phil, 12 GB, SDXL and Anima alternating: a grant revoked the planned plan's own
        // holdings and it loaded them again. What its executor maps is counted once and stays;
        // one it let go (huaisang: detached when Degree 2 did not fit) it never maps again,
        // so it is no part of its want and may go like any other.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB);
        loaded(&mut gpu, "anima", 11, 6 * GIB, GIB);
        gpu.learned.holding("anima", "GPU-1/sha256:qwen");
        gpu.learned.holding("anima", "GPU-1/sha256:dit");
        let held = |id: &str, bytes, readers: &[u32], idle_ms| Holding {
            id: id.into(),
            bytes,
            readers: readers.to_vec(),
            idle_ms,
            revoking: false,
        };
        gpu.holdings = vec![
            held("GPU-1/sha256:dit#7", 4 * GIB, &[11], 95_000),
            held("GPU-1/sha256:qwen#6", GIB, &[], 90_000),
            held("GPU-1/sha256:unet#5", 5 * GIB, &[10], 30_000),
        ];
        let s = sample(12 * GIB, GIB, &[(10, GIB / 2), (11, GIB / 2)]);
        let want = gpu.want("anima").unwrap();
        assert_eq!(want, GIB / 2 + 2 * GIB + GIB + MARGIN, "only what it maps is not asked again");
        let mut round = Round::default();
        assert_eq!(
            gpu.decide("anima", Some(want), gpu.need("anima"), &s, &mut round),
            Decision::Step(Step::Revoke("GPU-1/sha256:qwen#6".into())),
            "the holding it let go goes first, as the most idle"
        );
        gpu.holdings.remove(1);
        let s = sample(12 * GIB, 2 * GIB, &[(10, GIB / 2), (11, GIB / 2)]);
        let short = want - gpu.room("anima", &s);
        assert_eq!(
            gpu.decide("anima", Some(want), gpu.need("anima"), &s, &mut round),
            Decision::Step(Step::Trim("GPU-1/sha256:unet#5".into(), short)),
            "its own transformer, though more idle, stays"
        );
    }

    #[test]
    fn a_model_that_fits_once_an_idle_tenants_holdings_are_trimmed_shares() {
        // huaisang, 12 GB: SDXL (7 GiB) and Anima (5.5 GiB) do not both fit. Counting only
        // holdings nobody read, each found it did not fit beside the other's idle holdings,
        // detached and reloaded whole at every switch. The grant trims an idle tenant's
        // holdings, so they count as room at the fit.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB);
        loaded(&mut gpu, "anima", 11, 11 * GIB / 2, GIB / 2);
        gpu.holdings = vec![Holding {
            id: "GPU-1/sha256:dit#3".into(),
            bytes: 11 * GIB / 2,
            readers: vec![11],
            idle_ms: 1_000,
            revoking: false,
        }];
        let s = sample(12 * GIB, 6 * GIB - GIB / 2, &[(10, GIB / 2), (11, GIB / 2)]);
        assert!(gpu.room("sdxl", &s) < gpu.want("sdxl").unwrap());
        assert!(gpu.fits_resident("sdxl", &s));
        // Not while Anima's call reads them.
        gpu.active("anima", GIB / 2);
        assert!(!gpu.fits_resident("sdxl", &s));
    }

    #[test]
    fn a_holding_larger_than_the_shortfall_is_trimmed_by_the_shortfall() {
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "anima", 11, 6 * GIB, GIB);
        let held = |id: &str, bytes| Holding {
            id: id.into(),
            bytes,
            readers: vec![10],
            idle_ms: 5_000,
            revoking: false,
        };
        gpu.holdings = vec![held("GPU-1/sha256:unet#5", 5 * GIB)];
        let s = sample(12 * GIB, 5 * GIB, &[(11, GIB / 2)]);
        let want = gpu.want("anima").unwrap();
        let short = want - gpu.room("anima", &s);
        assert!(0 < short && short < 5 * GIB);
        assert_eq!(
            gpu.decide("anima", Some(want), gpu.need("anima"), &s, &mut Round::default()),
            Decision::Step(Step::Trim("GPU-1/sha256:unet#5".into(), short))
        );
        // A holding the shortfall exceeds goes whole.
        gpu.holdings = vec![held("GPU-1/sha256:te#4", GIB)];
        assert_eq!(
            gpu.decide("anima", Some(want), gpu.need("anima"), &s, &mut Round::default()),
            Decision::Step(Step::Revoke("GPU-1/sha256:te#4".into()))
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
        // 16 GiB card: an SDXL checkpoint (7 GiB) held and read by one package's running call.
        // A second package binds the same checkpoint; its earlier executor named that set.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "txt2img", 10, 7 * GIB, 3 * GIB);
        gpu.active("txt2img", GIB / 2); // its call reads the set: no grant may take it
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
    fn a_replacement_spawns_beside_the_holdings_it_attaches() {
        // 16 GiB card. An executor of SDXL (7 GiB of weights, 1 GiB floor, 7 GiB of activations)
        // was killed: custody keeps its weights, nobody reads them, 8.75 GiB are free.
        let mut gpu = Gpu::default();
        gpu.learned.load("sdxl", 7 * GIB, GIB);
        gpu.learned
            .call("sdxl", "height=1024,width=1024", 7 * GIB, &BTreeMap::new());
        gpu.learned.holding("sdxl", "GPU-1/sha256:sdxl");
        let held = |id: &str| Holding {
            id: id.into(),
            bytes: 7 * GIB,
            readers: vec![],
            idle_ms: 1_000,
            revoking: false,
        };
        gpu.holdings = vec![held("GPU-1/sha256:sdxl#1")];
        let s = sample(16 * GIB, 9 * GIB, &[]);
        // Its floor is part of what is held: the spawn asks for a context and activations.
        let need = gpu.spawn_need("sdxl");
        assert_eq!(need, gpu.context_estimate() + 7 * GIB);
        let mut round = Round::default();
        let room = gpu.room("sdxl", &s);
        assert_eq!(
            gpu.decide("sdxl", Some(need), need, &s, &mut round),
            Decision::Go(room),
            "the holding it maps next is not revoked for its spawn"
        );
        // Another checkpoint's holding is no part of its floor: it makes room as before.
        gpu.holdings = vec![held("GPU-1/sha256:other#1")];
        let need = gpu.spawn_need("sdxl");
        assert_eq!(need, gpu.context_estimate() + GIB + 7 * GIB);
        let short = need - gpu.room("sdxl", &s);
        assert_eq!(
            gpu.decide("sdxl", Some(need), need, &s, &mut round),
            Decision::Step(Step::Trim("GPU-1/sha256:other#1".into(), short))
        );
    }

    #[test]
    fn what_an_executor_let_go_comes_back_before_another_tenant_gives_anything() {
        // H3 runs; SDXL's idle executor has its weights mapped. H3's executor unmapped its
        // VAE to make room and asks for it: custody holds the VAE and H3's text encoder.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB);
        gpu.active("sdxl", 8 * GIB);
        gpu.idle("sdxl");
        loaded(&mut gpu, "h3", 11, 60 * GIB, 20 * GIB);
        for name in ["GPU-1/sha256:vae", "GPU-1/sha256:text"] {
            gpu.learned.holding("h3", name);
        }
        let held = |id: &str, readers: &[u32], idle_ms| Holding {
            id: id.into(),
            bytes: 6 * GIB,
            readers: readers.to_vec(),
            idle_ms,
            revoking: false,
        };
        gpu.holdings = vec![
            held("GPU-1/sha256:vae#3", &[], 10),
            held("GPU-1/sha256:text#2", &[11], 5_000),
            held("GPU-1/sha256:other#1", &[], 9_000),
        ];
        // The ladder alone would unmap SDXL first.
        let s = sample(96 * GIB, 2 * GIB, &[]);
        let ladder = gpu.decide("h3", None, u64::MAX, &s, &mut Round::default());
        assert_eq!(ladder, Decision::Step(Step::Unmap("sdxl".into())));
        // Its own unread holding goes first; one it still maps and another plan's never do.
        let mut round = Round::default();
        let own = gpu.own_unread("h3", &mut round);
        assert_eq!(own, Some(Step::Revoke("GPU-1/sha256:vae#3".into())));
        assert_eq!(gpu.own_unread("h3", &mut round), None);
        assert_eq!(gpu.own_unread("sdxl", &mut Round::default()), None);
    }

    #[test]
    fn a_running_call_asking_for_room_is_charged_what_it_holds_now() {
        // H3's replacement on a 96 GiB card: custody holds 74 GiB, the call runs with a
        // 21 GiB cap and holds 20 GiB, but its last report is its load's (1 GiB). It gave
        // back a 6 GiB holding and asks for room; 1 GiB of the card is foreign.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "h3", 11, 80 * GIB, 20 * GIB);
        gpu.observe(
            "h3",
            Facts {
                process: Some(GIB),
                ..Facts::default()
            },
            None,
        );
        gpu.active("h3", 21 * GIB);
        gpu.holdings = vec![Holding {
            id: "GPU-1/sha256:dit#1".into(),
            bytes: 68 * GIB,
            readers: vec![11],
            idle_ms: 0,
            revoking: false,
        }];
        let s = Sample {
            processes: None,
            ..sample(96 * GIB, 96 * GIB - 68 * GIB - 20 * GIB - GIB, &[])
        };
        // Stale: its 19 GiB since its load read as foreign, and the room is 19 GiB short.
        assert_eq!(gpu.external(&s), 20 * GIB);
        let room = 96 * GIB - HEADLESS_FLOOR - 68 * GIB - GIB;
        assert_eq!(gpu.room("h3", &s), room - 19 * GIB);
        // Its own count with the ask: the room is the card beside custody and the foreign GiB.
        gpu.observe(
            "h3",
            Facts {
                process: Some(20 * GIB),
                ..Facts::default()
            },
            None,
        );
        assert_eq!((gpu.external(&s), gpu.room("h3", &s)), (GIB, room));
    }

    #[test]
    fn a_report_from_before_an_export_hides_external_memory() {
        // 16 GiB card with 1 GiB of foreign memory. SDXL's idle executor holds a 0.5 GiB
        // context; its 7 GiB of weights are custody's since it exported them.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, 3 * GIB);
        gpu.holdings = vec![Holding {
            id: "GPU-1/sha256:sdxl#1".into(),
            bytes: 7 * GIB,
            readers: vec![10],
            idle_ms: 0,
            revoking: false,
        }];
        let s = Sample {
            processes: None,
            ..sample(16 * GIB, 16 * GIB - 7 * GIB - GIB / 2 - GIB, &[])
        };
        let room = 16 * GIB - HEADLESS_FLOOR - 7 * GIB - GIB;
        // Its report from before the export counts the weights again: the foreign GiB vanishes.
        let stale = Facts {
            process: Some(7 * GIB + GIB / 2),
            ..Facts::default()
        };
        gpu.observe("sdxl", stale, None);
        assert_eq!(gpu.external(&s), 0);
        assert_eq!(gpu.room("sdxl", &s), room + GIB);
        // The export's own reply says what the process still holds.
        let fresh = Facts {
            process: Some(GIB / 2),
            ..Facts::default()
        };
        gpu.observe("sdxl", fresh, None);
        assert_eq!((gpu.external(&s), gpu.room("sdxl", &s)), (GIB, room));
    }

    #[test]
    fn a_spawn_reservation_never_hides_foreign_memory() {
        // rtx-3070 with a 6 GiB pool: 2 GiB of ballast, no per-process NVML in the container.
        // Anima (5.4 GiB of weights, 1.4 GiB of activations) was measured in an earlier run.
        let mut gpu = Gpu::default();
        gpu.learned.load("anima", 5 * GIB + 2 * GIB / 5, 300 * MIB);
        gpu.learned.call(
            "anima",
            "height=1024,width=1024",
            7 * GIB / 5,
            &BTreeMap::new(),
        );
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
    fn an_unmeasured_call_leaves_a_card_with_room_for_it_alone() {
        // A40: package A's SDXL idle with 7 GiB held; package B's first call, loaded, its
        // activations never measured. 38 GiB free: nothing of A's needs to leave.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "a", 10, 7 * GIB, 3 * GIB);
        gpu.learned
            .call("a", "height=1024,width=1024", 3 * GIB, &BTreeMap::new());
        gpu.starting("b", 0);
        gpu.spawned("b", 11);
        gpu.observe(
            "b",
            Facts {
                context: Some(GIB / 4),
                weights: Some(7 * GIB),
                ..Facts::default()
            },
            Some(false),
        );
        let s = sample(46 * GIB, 38 * GIB, &[(10, 8 * GIB)]);
        assert_eq!(gpu.want("b"), None);
        let want = gpu.grant_want("b");
        assert_eq!(want, Some(GIB / 4 + 7 * GIB + 3 * GIB + MARGIN));
        let mut round = Round::default();
        let room = gpu.room("b", &s);
        assert_eq!(
            gpu.decide("b", want, gpu.need("b"), &s, &mut round),
            Decision::Go(room)
        );
    }

    #[test]
    fn a_running_call_that_asks_for_room_may_take_what_is_physically_free() {
        // An idle tenant last reported 2 GiB (a decode's cache, since given back); the driver
        // shows the card freer than the ledger's charges say.
        let mut gpu = Gpu::default();
        loaded(&mut gpu, "sdxl", 10, 7 * GIB, GIB / 2);
        gpu.observe(
            "sdxl",
            Facts {
                process: Some(2 * GIB),
                ..Facts::default()
            },
            None,
        );
        gpu.idle("sdxl");
        loaded(&mut gpu, "anima", 11, 6 * GIB, GIB);
        gpu.active("anima", GIB);
        let s = Sample {
            total: 4 * GIB,
            free: 2 * GIB,
            display: false,
            processes: None,
        };
        assert!(gpu.room("anima", &s) < gpu.physical_cap("anima", &s));
        assert_eq!(gpu.physical_cap("anima", &s), GIB / 2 + 2 * GIB - HEADLESS_FLOOR);
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
    fn a_plan_is_priced_by_the_weights_its_calls_map() {
        // H3 on one RTX PRO 6000 (run 4285): the construction registers two denoisers and a
        // call maps one. Priced whole, Degree 2 never fits; priced as mapped, it does.
        let mut gpu = Gpu {
            device: "GPU-1/595".into(),
            ..Gpu::default()
        };
        gpu.learned.load("h3", 102 * GIB, 2 * GIB);
        gpu.learned
            .call("h3", "frames=362", 20 * GIB, &BTreeMap::new());
        gpu.learned.context("GPU-1/595", 700 * MIB);
        gpu.set_shape("h3", "frames=362");
        let free = sample(102 * GIB, 102 * GIB, &[]);
        assert_eq!(
            gpu.want("h3"),
            Some(700 * MIB + MARGIN + 102 * GIB + 20 * GIB + MARGIN)
        );
        assert!(!gpu.fits_resident("h3", &free));
        gpu.learned.mapped("h3", 80 * GIB);
        assert_eq!(
            gpu.want("h3"),
            Some(700 * MIB + MARGIN + 80 * GIB + 20 * GIB + MARGIN)
        );
        assert!(gpu.fits_resident("h3", &free));
        // The lowest rung is unchanged: it never counted every weight.
        assert_eq!(gpu.need("h3"), 700 * MIB + MARGIN + 2 * GIB + 20 * GIB + MARGIN);
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
        // A shape larger than any measured: the executor's seeds and the activation are the
        // largest measured shape's, scaled by its tokens (4x the pixels of 1024).
        gpu.set_shape("sdxl", "height=2048,width=2048");
        assert_eq!(gpu.seeds("sdxl")["decode"], 12 * GIB);
        assert_eq!(gpu.activation("sdxl"), Some(12 * GIB));
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
