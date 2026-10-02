//! The machine's sole CPU dispatch policy; the engine only journals/supervises.
use crate::{
    catalog::Catalog,
    execution::Engine,
    journal::{Execution, State},
};
use serde_json::Value;
use std::{
    io,
    path::Path,
    sync::{Arc, Mutex},
};

pub struct Service {
    pub engine: Arc<Engine>,
    pub catalog: Catalog,
    parallelism: usize,
    stopped: Mutex<bool>,
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
        });
        service.engine.reconcile()?;
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
        self.engine.submit(key, invocation)
    }
    pub fn changed_environment(&self) -> io::Result<()> {
        // Explicit install/update completion, not an identical failed retry loop.
        self.engine.notify_activity();
        Ok(())
    }
    pub fn idle(&self) -> io::Result<bool> {
        Ok(self.engine.active(1)?.is_empty())
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
                    eprintln!(
                        "execution {} waiting for held generation: {error}",
                        record.id
                    );
                    continue;
                }
            };
            if record.invocation.package != held.record.package
                || record.invocation.module != held.record.application
            {
                eprintln!(
                    "execution {} has an incompatible held package identity",
                    record.id
                );
                continue;
            }
            self.engine.dispatch(&record.id, held.runner())?;
        }
        Ok(())
    }
}
