//! Independent native checkpoint custody for accepted work. Cache repositories and
//! caller-local `keep` snapshots are not execution obligations.
use crate::{execution::Engine, gpu_service::GpuPlan, journal::Preparation};
use prost::Message;
use std::{
    io,
    sync::{Arc, Mutex},
};
use tensorfs_core::{catalog::WriterGuard, checkpoint_root, ids::ObjectRef, store::Store};

pub struct Models {
    store: Arc<Store>,
    /// Unreadable obligations remove destructive-GC authority for every native caller.
    blocked: Mutex<Option<WriterGuard>>,
}

impl Models {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            store,
            blocked: Mutex::new(None),
        }
    }
    pub(crate) fn guard(&self) -> io::Result<WriterGuard> {
        WriterGuard::acquire(self.store.root()).map_err(io::Error::other)
    }
    pub(crate) fn block(&self, error: &io::Error) -> io::Result<()> {
        let mut blocked = self.blocked.lock().unwrap();
        if blocked.is_none() {
            *blocked = Some(self.guard()?);
            eprintln!("model custody cannot be proved; destructive store GC disabled: {error}");
        }
        Ok(())
    }

    /// Caller holds the engine custody mutex and native writer exclusion. Reserve the
    /// policy row first, install the native root next, reference it before acknowledgment.
    pub(crate) fn retain(
        &self,
        engine: &Engine,
        execution: Option<&str>,
        repository: &str,
        manifest: &ObjectRef,
    ) -> io::Result<()> {
        let repository = if repository.is_empty() {
            #[derive(serde::Deserialize)]
            struct Alias {
                org: String,
                name: String,
                manifest_sha256: String,
                manifest_length: u64,
            }
            let rows = tensorfs_core::storage::repository_release_lines(self.store.root())
                .map_err(io::Error::other)?;
            rows.into_iter()
                .map(|row| serde_json::from_slice::<Alias>(&row).map_err(io::Error::other))
                .collect::<io::Result<Vec<_>>>()?
                .into_iter()
                .find(|row| {
                    row.manifest_sha256 == manifest.sha256 && row.manifest_length == manifest.length
                })
                .map(|row| format!("{}/{}", row.org, row.name))
                .ok_or_else(|| io::Error::other("model input has no native source repository"))?
        } else {
            repository.into()
        };
        let mut root = engine.with_journal(|j| j.reserve_model(&repository, manifest))?;
        if checkpoint_root::read(&self.store, &root.owner)
            .map_err(io::Error::other)?
            .is_some_and(|held| held.released)
        {
            // A crash after native release but before its journal mark is recoverable.
            engine.with_journal(|j| j.release_model(&manifest.sha256))?;
            root = engine.with_journal(|j| j.reserve_model(&repository, manifest))?;
        }
        let standing = checkpoint_root::read(&self.store, &root.owner).map_err(io::Error::other)?;
        match standing {
            Some(held) if held.manifest == *manifest && !held.released => {}
            Some(_) => {
                return Err(io::Error::other(
                    "native model root differs from journal custody",
                ))
            }
            None => {
                checkpoint_root::retain(
                    &self.store,
                    &root.owner,
                    &root.repository,
                    manifest.clone(),
                )
                .map_err(io::Error::other)?;
            }
        }
        if let Some(execution) = execution {
            engine.with_journal(|j| j.reference_model(execution, &manifest.sha256))?;
        }
        Ok(())
    }

    pub(crate) fn preparation(
        &self,
        engine: &Engine,
        execution: Option<&str>,
        preparation: &Preparation,
    ) -> io::Result<()> {
        let document: serde_json::Value =
            serde_json::from_slice(&preparation.document).map_err(io::Error::other)?;
        if document.get("slots").is_none() {
            // CPU preparations have no model slots. An unreadable GPU record is not that.
            if preparation.id.starts_with("gpu-") {
                return Err(io::Error::other("GPU preparation has no model slots"));
            }
            return Ok(());
        }
        let plan: GpuPlan = serde_json::from_value(document).map_err(io::Error::other)?;
        if plan.id != preparation.id || plan.installation != preparation.installation {
            return Err(io::Error::other(
                "GPU preparation differs from journal subject",
            ));
        }
        for slot in plan.slots {
            let sha256 = slot
                .binding
                .snapshot
                .strip_prefix("sha256:")
                .unwrap_or(&slot.binding.snapshot);
            tensorfs_core::ids::hex64("model custody", sha256).map_err(io::Error::other)?;
            let length = std::fs::metadata(self.store.manifest_path(sha256))?.len();
            let repository = slot.binding.model.split('@').next().unwrap_or_default();
            self.retain(
                engine,
                execution,
                repository,
                &ObjectRef {
                    sha256: sha256.into(),
                    length,
                },
            )?;
        }
        Ok(())
    }

    /// Restore legacy accepted/prepared dependencies before the first publisher or GC.
    /// Existing native roots survive independently of this reconstruction.
    pub(crate) fn restore(&self, engine: &Engine) -> io::Result<()> {
        let _writer = self.guard()?;
        for (execution, preparation) in engine.with_journal(|j| j.model_preparations())? {
            self.preparation(engine, Some(&execution), &preparation)?;
        }
        #[derive(serde::Deserialize)]
        struct Inputs {
            #[serde(default)]
            inputs: std::collections::BTreeMap<String, (String, String, u64)>,
            #[serde(default)]
            models: Vec<Vec<u8>>,
        }
        for (execution, document) in engine.with_journal(|j| j.model_job_inputs())? {
            let input: Inputs = serde_json::from_slice(&document).map_err(io::Error::other)?;
            let choices = input
                .models
                .iter()
                .map(|bytes| {
                    crate::api::pb::ModelChoice::decode(bytes.as_slice()).map_err(io::Error::other)
                })
                .collect::<io::Result<Vec<_>>>()?;
            for (_, (_, sha256, length)) in input.inputs {
                let repository = choices
                    .iter()
                    .find(|choice| {
                        choice.manifest.as_ref().is_some_and(|reference| {
                            tensorfs_core::sha256::hex(&reference.digest) == sha256
                        })
                    })
                    .map(|choice| choice.repository.as_str())
                    .unwrap_or_default();
                self.retain(
                    engine,
                    Some(&execution),
                    repository,
                    &ObjectRef { sha256, length },
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn sweep(&self, engine: &Engine) -> io::Result<usize> {
        if self.blocked.lock().unwrap().is_some() {
            return Ok(0);
        }
        let _writer = self.guard()?;
        let mut released = 0;
        for root in engine.with_journal(|j| j.unused_models(crate::reclaim::IDLE))? {
            checkpoint_root::release(
                &self.store,
                &root.owner,
                &root.repository,
                ObjectRef {
                    sha256: root.sha256.clone(),
                    length: root.length,
                },
            )
            .map_err(io::Error::other)?;
            engine.with_journal(|j| j.release_model(&root.sha256))?;
            released += 1;
        }
        Ok(released)
    }
}
