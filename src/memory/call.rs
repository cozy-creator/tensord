//! One call's device memory as its executor measured it: the run's `memory` measurements,
//! which `cozy run show` prints. Peaks are the call's own; holdings are its process's at the
//! call's end; counts are what its plane and torch did since the executor's previous call (since
//! it started, for its first).
use crate::device_executor::{Frame, PlaneFacts, Shortfall};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct CallMemory {
    /// torch's high-water marks: live tensors, and the segments it reserved for them.
    pub peak_allocated_bytes: Option<u64>,
    pub peak_reserved_bytes: Option<u64>,
    /// The peak above the call's start (activations, derived weights), and each stage method's.
    pub activation_peak_bytes: Option<u64>,
    pub stage_peaks: BTreeMap<String, u64>,
    /// The process on its GPU at the end: CUDA context + torch reserved + the plane's maps.
    pub process_bytes: Option<u64>,
    pub context_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    /// Weights mapped on the GPU (its own and custody's), the plane's budget, the process cap.
    pub weights_bytes: Option<u64>,
    pub budget_bytes: Option<u64>,
    pub cap_bytes: Option<u64>,
    /// Weight bytes copied to the GPU; of them, read from disk and bounced through staging.
    pub paged_in_bytes: Option<u64>,
    pub disk_bytes: Option<u64>,
    pub bounce_bytes: Option<u64>,
    pub register_failed: Option<u64>,
    /// Weight regions evicted, and components given back whole to make torch room.
    pub evictions: Option<u64>,
    pub paged_out: Option<u64>,
    pub paged_out_bytes: Option<u64>,
    /// torch short of room: whole-cache drops to place one allocation, OOMs it raised, the
    /// executor's retries after one, and its own cache releases.
    pub alloc_retries: Option<u64>,
    pub alloc_ooms: Option<u64>,
    pub oom_retries: Option<u64>,
    pub cache_releases: Option<u64>,
    pub cache_released_bytes: Option<u64>,
    /// How each stage ran (decode mode, CFG layout), and what a refused stage lacked.
    pub modes: BTreeMap<String, String>,
    pub shortfall: Option<Shortfall>,
}

impl CallMemory {
    /// `since` is the plane's counts at the executor's previous call. None: nothing measured.
    pub fn of(reply: &Frame, since: &PlaneFacts) -> Option<Self> {
        if reply.metrics.is_none() && reply.plane.is_none() && reply.shortfall.is_none() {
            return None;
        }
        let metrics = reply.metrics.clone().unwrap_or_default();
        let plane = reply.plane.clone().unwrap_or_default();
        let moved = |of: fn(&PlaneFacts) -> Option<u64>| {
            of(&plane).map(|now| now.saturating_sub(of(since).unwrap_or(0)))
        };
        Some(Self {
            peak_allocated_bytes: known(metrics.peak_vram_bytes),
            peak_reserved_bytes: known(metrics.peak_reserved_bytes),
            activation_peak_bytes: known(plane.activation_peak_bytes)
                .or(known(metrics.working_peak_vram_bytes)),
            stage_peaks: metrics
                .activation_peaks
                .iter()
                .filter_map(|(stage, bytes)| Some((stage.clone(), known(Some(*bytes))?)))
                .collect(),
            process_bytes: known(plane.process_bytes),
            context_bytes: known(plane.context_bytes),
            reserved_bytes: known(plane.reserved_bytes),
            weights_bytes: known(plane.committed_bytes)
                .map(|own| own + known(plane.shared_bytes).unwrap_or(0)),
            budget_bytes: known(plane.budget_bytes),
            cap_bytes: known(plane.cap_bytes),
            paged_in_bytes: moved(|p| p.h2d_bytes),
            disk_bytes: moved(|p| p.disk_copy_bytes),
            bounce_bytes: moved(|p| p.bounce_copy_bytes),
            register_failed: moved(|p| p.register_failed),
            evictions: moved(|p| p.evictions),
            paged_out: moved(|p| p.paged_out),
            paged_out_bytes: moved(|p| p.paged_out_bytes),
            alloc_retries: moved(|p| p.alloc_retries),
            alloc_ooms: moved(|p| p.alloc_ooms),
            oom_retries: moved(|p| p.oom_retries),
            cache_releases: moved(|p| p.cache_releases),
            cache_released_bytes: moved(|p| p.cache_released_bytes),
            modes: plane.modes,
            shortfall: reply.shortfall.clone(),
        })
    }
}

/// The executor's signed byte counts: -1 is unreadable.
fn known(value: Option<i64>) -> Option<u64> {
    value.and_then(|v| u64::try_from(v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_counts_what_moved_since_the_executors_previous_call() {
        let reply: Frame = serde_json::from_value(serde_json::json!({
            "metrics": {"peak_vram_bytes": 900, "peak_reserved_bytes": 1000,
                "activation_peaks": {"vae.decode": 300}},
            "plane": {"h2d_bytes": 5000, "evictions": 7, "committed_bytes": 400,
                "shared_bytes": 100, "context_bytes": -1, "activation_peak_bytes": 350},
            "shortfall": {"resource": "vram", "needed_bytes": 1200, "available_bytes": 1150,
                "request_shape": "vae.decode"}
        }))
        .unwrap();
        let before = PlaneFacts { h2d_bytes: Some(3000), evictions: Some(2), ..Default::default() };
        let memory = CallMemory::of(&reply, &before).unwrap();
        assert_eq!(memory.peak_reserved_bytes, Some(1000));
        assert_eq!(memory.activation_peak_bytes, Some(350));
        assert_eq!(memory.stage_peaks["vae.decode"], 300);
        assert_eq!((memory.paged_in_bytes, memory.evictions), (Some(2000), Some(5)));
        assert_eq!((memory.weights_bytes, memory.context_bytes), (Some(500), None));
        assert_eq!(memory.paged_out, None);
        assert_eq!(memory.shortfall.unwrap().needed_bytes, 1200);
        assert!(CallMemory::of(&Frame::default(), &before).is_none());
    }
}
