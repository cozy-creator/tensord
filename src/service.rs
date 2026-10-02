//! The machine's sole CPU dispatch policy; the engine only journals/supervises.
use crate::{
    catalog::Catalog,
    execution::{process_ended, Engine},
    journal::{Execution, State, SubmissionContext},
};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io,
    os::fd::{AsRawFd, FromRawFd},
    path::Path,
    sync::{Arc, Mutex},
};

pub struct Service {
    pub engine: Arc<Engine>,
    pub catalog: Catalog,
    parallelism: usize,
    stopped: Mutex<bool>,
    retained: Mutex<HashMap<String, Arc<File>>>,
}
impl Service {
    pub fn open(root: &Path, generations: &Path, parallelism: usize) -> io::Result<Arc<Self>> {
        if parallelism == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU parallelism must be positive",
            ));
        }
        let service = Arc::new(Self {
            engine: Engine::open(&root.join("execution"))?,
            catalog: Catalog::new(generations)?,
            parallelism,
            stopped: Mutex::new(false),
            retained: Mutex::new(HashMap::new()),
        });
        service.engine.reconcile()?;
        // Keep queued generations alive, including accepted work from a prior boot.
        for record in service.engine.nonterminal(usize::MAX)? {
            if let Ok(held) = service.catalog.resolve(&record.invocation.generation) {
                service
                    .retained
                    .lock()
                    .unwrap()
                    .insert(record.id.clone(), held.retention());
            }
            if let Some(birth) = record.process {
                service.watch_orphan(birth)?;
            }
        }
        let owner = service.clone();
        std::thread::Builder::new()
            .name("machine-dispatch".into())
            .spawn(move || owner.dispatch_loop())?;
        Ok(service)
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
        entrypoint: &str,
        input: Value,
        boot: &str,
    ) -> io::Result<Execution> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Err(io::Error::other("machine is stopping"));
        }
        let held = self.catalog.resolve(generation)?;
        let record = self.engine.submit_public_on_boot(
            context,
            held.invocation(entrypoint, input)?,
            boot,
        )?;
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
        Ok(self.engine.nonterminal(1)?.is_empty())
    }
    pub fn stop(&self) -> io::Result<bool> {
        let mut stopped = self.stopped.lock().unwrap();
        if !self.idle()? {
            return Ok(false);
        }
        *stopped = true;
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
            self.engine.wait_activity(epoch, None);
        }
    }
    fn dispatch_ready(&self) -> io::Result<()> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Ok(());
        }
        // Orphan exact process births remain reservations; a new owner does not
        // pretend they are free merely because it has no local supervisor.
        let active = self.engine.active(self.parallelism)?;
        let room = self.parallelism.saturating_sub(active.len());
        if room == 0 {
            return Ok(());
        }
        for record in self.engine.ready(room)? {
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
            self.engine.dispatch(&record.id, held.runner())?;
        }
        Ok(())
    }
    fn watch_orphan(&self, birth: crate::journal::ProcessBirth) -> io::Result<()> {
        if process_ended(&birth)? {
            return Ok(());
        }
        // Read birth on both sides of pidfd_open so PID reuse cannot watch or
        // reclaim a different process. A kernel exit event only wakes reconciliation.
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, birth.pid, 0) } as i32;
        if raw < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                self.engine.notify_activity();
                return Ok(());
            }
            return Err(error);
        }
        let pidfd = unsafe { File::from_raw_fd(raw) };
        if process_ended(&birth)? {
            self.engine.notify_activity();
            return Ok(());
        }
        let engine = self.engine.clone();
        std::thread::Builder::new()
            .name(format!("orphan-{}", birth.pid))
            .spawn(move || {
                let mut item = libc::pollfd {
                    fd: pidfd.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                loop {
                    let result = unsafe { libc::poll(&mut item, 1, -1) };
                    if result >= 0 {
                        engine.notify_activity();
                        return;
                    }
                    let error = io::Error::last_os_error();
                    if error.kind() != io::ErrorKind::Interrupted {
                        eprintln!("orphan kernel observation: {error}");
                        return;
                    }
                }
            })?;
        Ok(())
    }
}
