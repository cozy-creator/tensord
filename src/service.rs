//! The machine's sole CPU dispatch policy; the engine only journals/supervises.
use crate::{
    catalog::Catalog,
    execution::{process_ended, Engine},
    journal::{Execution, ProcessBirth, State, SubmissionContext},
};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io,
    path::Path,
    sync::{Arc, Mutex},
};

/// What one public submission calls on its generation.
#[derive(Default)]
pub struct Call {
    pub entrypoint: String,
    pub input: Value,
    pub attention_kernel: String,
    pub inputs: Vec<crate::journal::InputFile>,
    pub job: bool,
    /// A child run's parent execution id.
    pub parent: String,
}

pub struct Service {
    pub engine: Arc<Engine>,
    pub catalog: Catalog,
    parallelism: usize,
    stopped: Mutex<bool>,
    retained: Mutex<HashMap<String, Arc<File>>>,
    gpu: Mutex<Option<Arc<crate::gpu_service::GpuPool>>>,
    /// Jobs and model-less child runs, in deviceless executors.
    jobs: Mutex<Option<Arc<crate::jobs::Jobs>>>,
    /// The TensorFS store's caches (`Publisher::reclaim`), swept with the machine's.
    publisher: Mutex<Option<Arc<crate::published::Publisher>>>,
    startup_gpu_births: Mutex<Vec<ProcessBirth>>,
    /// Package/model preparations in flight: work a rental's idle release must wait for.
    preparing: std::sync::atomic::AtomicUsize,
    /// This machine's birth: a fenced leader's group members born before it are its
    /// previous incarnation's followers.
    started_ticks: u64,
    swept: Mutex<std::time::Instant>,
}
/// Held while one preparation runs; the machine is not idle meanwhile.
pub struct Preparing(Arc<Service>);
impl Drop for Preparing {
    fn drop(&mut self) {
        self.0
            .preparing
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        self.0.engine.notify_activity();
    }
}

const SWEEP_EVERY: std::time::Duration = crate::reclaim::IDLE;
impl Service {
    pub fn open(root: &Path, generations: &Path, parallelism: usize) -> io::Result<Arc<Self>> {
        if parallelism == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU parallelism must be positive",
            ));
        }
        let engine = Engine::open(&root.join("execution"))?;
        // A completed request does not prove its retained CUDA context ended. A prior
        // machine's executor cannot be adopted (nothing can reach it), so it is ended now
        // and GPU admission stays fenced until its exit is observed.
        let mut cursor = 0;
        let mut seen = HashSet::new();
        let mut startup_gpu_births = vec![];
        let started_ticks = crate::process::own_start_ticks()?;
        loop {
            let page = engine.gpu_births_after(cursor, 256)?;
            if page.is_empty() {
                break;
            }
            for (id, birth) in page {
                cursor = id;
                let alive = !process_ended(&birth).unwrap_or(false)
                    || crate::process::group_outlives(&birth, started_ticks);
                if alive && seen.insert((birth.pid, birth.boot_id.clone(), birth.start_ticks)) {
                    // A follower left behind by a leader that already ended is ended too.
                    for member in crate::process::group_members(&birth) {
                        if member.start_ticks < started_ticks {
                            if let Ok(Some(exact)) = crate::process::Exact::open(&member) {
                                let _ = exact.kill();
                            }
                        }
                    }
                    if let Err(error) = engine.end_orphan(birth.clone()) {
                        eprintln!("GPU startup birth remains fenced: {error}");
                    }
                    startup_gpu_births.push(birth);
                }
            }
        }
        // Descendants of those executors (a setsid daemon, a compile worker) live on in the
        // executors' own scopes; every such scope of this machine is ended now.
        match crate::scope::Scope::sweep(&crate::scope::namespace(root)) {
            Ok(0) => (),
            Ok(held) => eprintln!("ended {held} process(es) left in earlier executor scopes"),
            Err(error) => eprintln!("earlier executor scopes remain: {error}"),
        }
        let service = Arc::new(Self {
            engine,
            catalog: Catalog::new(generations)?,
            parallelism,
            stopped: Mutex::new(false),
            retained: Mutex::new(HashMap::new()),
            gpu: Mutex::new(None),
            jobs: Mutex::new(None),
            publisher: Mutex::new(None),
            startup_gpu_births: Mutex::new(startup_gpu_births),
            preparing: std::sync::atomic::AtomicUsize::new(0),
            started_ticks,
            swept: Mutex::new(std::time::Instant::now()),
        });
        service.engine.reconcile()?;
        service.reclaim();
        // Keep queued and paused runs' generations alive, including a prior boot's.
        let paused = service.engine.paused(usize::MAX)?;
        for record in service.engine.nonterminal(usize::MAX)?.into_iter().chain(paused) {
            if let Ok(held) = service.catalog.resolve(&record.invocation.generation) {
                service
                    .retained
                    .lock()
                    .unwrap()
                    .insert(record.id.clone(), held.retention());
            }
            if let Some(birth) = record.process {
                service.engine.end_orphan(birth)?;
            }
        }
        let owner = service.clone();
        std::thread::Builder::new()
            .name("machine-dispatch".into())
            .spawn(move || owner.dispatch_loop())?;
        Ok(service)
    }
    pub fn gpu(&self) -> Option<Arc<crate::gpu_service::GpuPool>> {
        self.gpu.lock().unwrap().clone()
    }
    pub fn jobs(&self) -> Option<Arc<crate::jobs::Jobs>> {
        self.jobs.lock().unwrap().clone()
    }
    pub fn configure_publisher(&self, publisher: Arc<crate::published::Publisher>) {
        *self.publisher.lock().unwrap() = Some(publisher);
    }
    pub fn configure_jobs(&self, jobs: Arc<crate::jobs::Jobs>) {
        *self.jobs.lock().unwrap() = Some(jobs);
        self.engine.notify_activity();
    }
    /// Observation only. Unknown births stay reserved; CPU dispatch stays available.
    pub fn gpu_startup_fences(&self) -> usize {
        let mut births = self.startup_gpu_births.lock().unwrap();
        // A group's followers die with their leader (one group kill); each exit is observed.
        births.retain(|birth| {
            !process_ended(birth).unwrap_or(false)
                || crate::process::group_outlives(birth, self.started_ticks)
        });
        births.len()
    }
    pub fn configure_gpu(&self, gpu: Arc<crate::gpu_service::GpuPool>) -> io::Result<()> {
        let mut current = self.gpu.lock().unwrap();
        if current.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "GPU service is already configured",
            ));
        }
        *current = Some(gpu.clone());
        drop(current);
        // Parents of the most recently used generations first, while the host has room.
        let plans = self.recent_gpu_plans(&gpu);
        let recent: Vec<_> = plans.iter().map(|(held, _)| held.clone()).collect();
        let others = self.catalog.installed().into_iter().filter(|held| {
            !recent
                .iter()
                .any(|r| r.record.identity == held.record.identity)
        });
        for held in recent.clone().into_iter().chain(others) {
            gpu.kernel_boot(held.clone());
            gpu.prespawn(held);
        }
        // A previous run's executors still exiting fence it, as they fence requests.
        let fence = self.startup_gpu_births.lock().unwrap().clone();
        gpu.prewarm(&self.engine, plans, fence);
        self.changed_environment()
    }
    /// Each installed GPU generation's most recently used construction, newest first.
    fn recent_gpu_plans(
        &self,
        gpu: &crate::gpu_service::GpuPool,
    ) -> Vec<(crate::catalog::HeldGeneration, crate::gpu_service::GpuPlan)> {
        let mut seen = std::collections::BTreeSet::new();
        let mut plans = vec![];
        for preparation in self.engine.recent_preparations(64).unwrap_or_default() {
            let Ok(plan) = gpu.plan(&preparation) else {
                continue;
            };
            if !seen.insert(plan.generation.clone()) {
                continue;
            }
            if let Ok(held) = self.catalog.resolve(&plan.generation) {
                plans.push((held, plan));
            }
        }
        plans
    }

    pub fn submit(
        &self,
        key: &str,
        generation: &str,
        entrypoint: &str,
        input: Value,
    ) -> io::Result<Execution> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Err(io::Error::other("machine is stopping"));
        }
        let held = self.catalog.resolve(generation)?;
        let invocation = held.invocation(entrypoint, input)?;
        let record = self.engine.submit(key, invocation)?;
        self.retain(&record, held.retention());
        Ok(record)
    }
    pub fn submit_public(
        &self,
        context: SubmissionContext,
        generation: &str,
        call: Call,
        boot: &str,
    ) -> io::Result<Execution> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Err(io::Error::other("machine is stopping"));
        }
        let held = self.catalog.resolve(generation)?;
        let mut invocation = held.invocation(&call.entrypoint, call.input)?;
        invocation.attention_kernel = call.attention_kernel;
        invocation.inputs = call.inputs;
        let record = self
            .engine
            .submit_public_on_boot(context, invocation, boot)?;
        self.retain(&record, held.retention());
        Ok(record)
    }
    /// A run prepared inside itself (`runs`) names its generation, call and model plan.
    pub fn bind_prepared(
        &self,
        id: &str,
        generation: &str,
        call: Call,
        preparation: &str,
    ) -> io::Result<Execution> {
        let held = self.catalog.resolve(generation)?;
        let mut invocation = held.invocation(&call.entrypoint, call.input)?;
        invocation.attention_kernel = call.attention_kernel;
        invocation.inputs = call.inputs;
        invocation.job = call.job;
        invocation.parent = call.parent;
        let record = self.engine.bind_prepared(id, invocation, preparation)?;
        self.retain(&record, held.retention());
        Ok(record)
    }
    fn retain(&self, record: &Execution, hold: Arc<File>) {
        if !record.state.terminal() {
            self.retained
                .lock()
                .unwrap()
                .insert(record.id.clone(), hold);
        }
    }
    pub fn changed_environment(&self) -> io::Result<()> {
        // Explicit install/update completion, not an identical failed retry loop.
        for record in self.engine.nonterminal(usize::MAX)? {
            if record.state == State::Queued
                && record.waiting_reason.is_some()
                && self.catalog.resolve(&record.invocation.generation).is_ok()
            {
                self.engine.wait_for_environment(&record.id, None)?;
            }
        }
        self.engine.notify_activity();
        Ok(())
    }
    pub fn idle(&self) -> io::Result<bool> {
        Ok(
            self.preparing.load(std::sync::atomic::Ordering::Acquire) == 0
                && self.engine.nonterminal(1)?.is_empty(),
        )
    }
    pub fn preparing(self: &Arc<Self>) -> Preparing {
        self.preparing
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Preparing(self.clone())
    }
    pub fn stop(&self) -> io::Result<bool> {
        let mut stopped = self.stopped.lock().unwrap();
        if !self.idle()? {
            return Ok(false);
        }
        *stopped = true;
        if let Some(gpu) = self.gpu() {
            gpu.stop()?;
        }
        self.engine.notify_activity();
        Ok(true)
    }
    fn dispatch_loop(self: Arc<Self>) {
        loop {
            let epoch = self.engine.activity_epoch();
            if *self.stopped.lock().unwrap() {
                return;
            }
            if let Err(error) = self.engine.reconcile() {
                eprintln!("executor birth observation: {error}");
            }
            self.retained.lock().unwrap().retain(|id, _| {
                self.engine
                    .get(id)
                    .map(|r| !r.state.terminal())
                    .unwrap_or(true)
            });
            if let Err(error) = self.dispatch_ready() {
                eprintln!("dispatch observation: {error}");
            }
            if self.swept.lock().unwrap().elapsed() >= SWEEP_EVERY {
                self.reclaim();
            }
            // The wait bounds only how long caches go unswept; it ends nothing.
            self.engine.wait_activity(epoch, Some(SWEEP_EVERY));
        }
    }
    /// Caches manage themselves (`reclaim`): TTL and storage pressure, never a purge verb.
    pub fn reclaim(&self) -> crate::reclaim::Swept {
        *self.swept.lock().unwrap() = std::time::Instant::now();
        // Released Write roots first, so the store's relief can collect their bytes.
        if let Some(jobs) = self.jobs() {
            if let Err(error) = jobs.sweep_objects() {
                eprintln!("reclaim object roots: {error}");
            }
        }
        let bound = self.engine.bound_generations().map(|mut bound| {
            if let Some(gpu) = self.gpu() {
                bound.extend(gpu.config().packages.iter().map(|p| p.generation.clone()));
            }
            bound
        });
        let kernels = self.gpu().map(|gpu| gpu.kernel_caches());
        match bound.and_then(|bound| {
            crate::reclaim::sweep(&self.engine, &self.catalog, &bound, kernels.as_ref())
        }) {
            Ok(mut swept) => {
                if let Some(jobs) = self.jobs() {
                    swept.scratch = jobs.sweep_scratch(&self.engine);
                }
                if let Some(gpu) = self.gpu() {
                    let disk = || crate::reclaim::Disk::measure(&self.engine.root);
                    let pressure = disk().is_ok_and(|disk| disk.pressure());
                    let relieved = || disk().is_ok_and(|disk| disk.relieved());
                    swept.memo = gpu.memo().sweep(pressure, relieved);
                }
                let publisher = self.publisher.lock().unwrap().clone();
                match publisher.map(|publisher| publisher.reclaim(self)) {
                    Some(Ok((views, bytes))) => {
                        swept.adapter_views = views;
                        swept.store_bytes = bytes;
                    }
                    Some(Err(error)) => eprintln!("reclaim store: {error}"),
                    None => {}
                }
                if swept != crate::reclaim::Swept::default() {
                    eprintln!("reclaimed {swept:?}");
                }
                swept
            }
            Err(error) => {
                eprintln!("reclaim: {error}");
                crate::reclaim::Swept::default()
            }
        }
    }
    fn dispatch_ready(&self) -> io::Result<()> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Ok(());
        }
        // Orphan exact process births remain reservations; a new owner does not
        // pretend they are free merely because it has no local supervisor.
        let active = self.engine.active(usize::MAX)?;
        let is_gpu = |record: &Execution| {
            record
                .submission
                .as_ref()
                .is_some_and(|s| !s.preparation_id.is_empty())
        };
        let mut gpu_active = self.gpu_startup_fences() != 0 || active.iter().any(is_gpu);
        // A job mostly waits on its children; only they take CPU room.
        let mut room = self.parallelism.saturating_sub(
            active
                .iter()
                .filter(|r| !is_gpu(r) && !r.invocation.job)
                .count(),
        );
        let jobs = self.jobs.lock().unwrap().clone();
        // A queued job needs no room, so with jobs every ready record is looked at.
        if room == 0 && (gpu_active || self.gpu().is_none()) && jobs.is_none() {
            return Ok(());
        }
        let mut cursor = 0;
        loop {
            let page = self.engine.ready_after(cursor, 256)?;
            if page.is_empty() {
                break;
            }
            for record in page {
                cursor = record.id.parse().map_err(io::Error::other)?;
                if record.state != State::Queued || record.waiting_reason.is_some() {
                    continue;
                }
                let held = match self.catalog.resolve(&record.invocation.generation) {
                    Ok(held) => held,
                    Err(error) => {
                        self.engine.wait_for_environment(
                            &record.id,
                            Some(format!("held generation unavailable: {error}")),
                        )?;
                        continue;
                    }
                };
                if record.invocation.package != held.record.package
                    || record.invocation.module != held.record.application
                {
                    self.engine.wait_for_environment(
                        &record.id,
                        Some("held package identity differs from accepted invocation".into()),
                    )?;
                    continue;
                }
                if crate::jobs::Jobs::takes(&record) {
                    let Some(jobs) = &jobs else {
                        self.engine.wait_for_environment(
                            &record.id,
                            Some("CPU execution is not configured on this machine".into()),
                        )?;
                        continue;
                    };
                    if record.invocation.job {
                        jobs.dispatch(&self.engine, &record, held)?;
                    } else if room > 0 && jobs.dispatch(&self.engine, &record, held)? {
                        room -= 1;
                    }
                } else if let Some(context) = record
                    .submission
                    .as_ref()
                    .filter(|s| !s.preparation_id.is_empty())
                {
                    if gpu_active {
                        continue;
                    }
                    let Some(gpu) = self.gpu() else {
                        self.engine.wait_for_environment(
                            &record.id,
                            Some("GPU execution operation is not configured".into()),
                        )?;
                        continue;
                    };
                    let preparation = self
                        .engine
                        .preparation(&context.actor, &context.preparation_id)?
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "accepted GPU preparation is absent",
                            )
                        })?;
                    gpu_active =
                        gpu.dispatch(&self.engine, &record, held, gpu.plan(&preparation)?)?;
                }
                if room == 0 && gpu_active && jobs.is_none() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod preparing_tests {
    use super::*;

    #[test]
    fn a_preparation_in_flight_keeps_the_machine_busy() {
        let root = std::env::temp_dir().join(format!("service-preparing-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root, &root.join("generations"), 1).unwrap();
        assert!(service.idle().unwrap());
        let preparing = service.preparing();
        assert!(
            !service.idle().unwrap(),
            "a rental must not release itself mid-install"
        );
        drop(preparing);
        assert!(service.idle().unwrap());
        assert!(service.stop().unwrap());
        let _ = std::fs::remove_dir_all(root);
    }
}
