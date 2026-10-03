//! What executors measured, kept across executors and machine restarts: the CUDA context per
//! GPU and driver, and each plan's activation growth per request shape (per stage method and
//! for the whole call). Estimates come from these, not from constants; a plan unused for
//! `TTL` is forgotten.
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const TTL: Duration = Duration::from_secs(30 * 24 * 3600);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Shape {
    /// The call's peak activation growth.
    pub peak: u64,
    /// Each stage method's.
    pub methods: BTreeMap<String, u64>,
}
impl Shape {
    /// What the call grows by at its worst: its own peak or a stage's (a stage counts the
    /// segments torch reserved for it), whichever is more.
    pub fn bytes(&self) -> u64 {
        self.methods.values().copied().fold(self.peak, u64::max)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Plan {
    pub shapes: BTreeMap<String, Shape>,
    /// The executor's private host bytes (PSS less shared memory) after a call.
    pub host_bytes: u64,
    /// Its weights as stages count them at load (decoded copies included) and their floor.
    pub weights: u64,
    pub weights_floor: u64,
    pub used_ms: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Learned {
    /// GPU UUID and driver version -> the largest context an executor measured there.
    pub contexts: BTreeMap<String, u64>,
    pub plans: BTreeMap<String, Plan>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// A shape cell's axes (`plan.shape_cell`: `axis=value,...`), or None when one is not a count.
fn axes(cell: &str) -> Option<BTreeMap<&str, i64>> {
    cell.split(',')
        .filter(|part| *part != "-" && !part.is_empty())
        .map(|part| {
            let (axis, value) = part.split_once('=')?;
            Some((axis, value.parse().ok()?))
        })
        .collect()
}

/// The request's shape cell from PrepareRequest `features` (Runtime `worker.plan.shape_cell`).
pub fn shape_cell(features: &BTreeMap<String, serde_json::Value>) -> String {
    let cell: Vec<String> = features
        .iter()
        .map(|(axis, value)| format!("{axis}={value}"))
        .collect();
    if cell.is_empty() {
        "-".into()
    } else {
        cell.join(",")
    }
}

impl Learned {
    /// Read what earlier runs learned at `path`; a missing or unreadable file starts empty.
    pub fn open(path: &Path) -> Self {
        let mut learned: Self = fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        let horizon = now_ms().saturating_sub(TTL.as_millis() as u64);
        learned.plans.retain(|_, plan| plan.used_ms >= horizon);
        learned.path = Some(path.to_path_buf());
        learned
    }

    pub fn save(&self) -> io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let temporary = path.with_extension("json.new");
        fs::write(&temporary, serde_json::to_vec(self)?)?;
        fs::rename(temporary, path)
    }

    pub fn context(&mut self, device: &str, measured: u64) {
        let entry = self.contexts.entry(device.into()).or_default();
        *entry = (*entry).max(measured);
    }

    /// One call's measurements for `plan` at `shape`; estimates only grow.
    pub fn call(&mut self, plan: &str, shape: &str, peak: u64, methods: &BTreeMap<String, u64>) {
        let row = self.plans.entry(plan.into()).or_default();
        row.used_ms = now_ms();
        let cell = row.shapes.entry(shape.into()).or_default();
        cell.peak = cell.peak.max(peak);
        for (method, bytes) in methods {
            let entry = cell.methods.entry(method.clone()).or_default();
            *entry = (*entry).max(*bytes);
        }
    }

    /// `plan`'s weights at its last load: what the next load plans before it starts.
    pub fn load(&mut self, plan: &str, weights: u64, floor: u64) {
        let row = self.plans.entry(plan.into()).or_default();
        row.used_ms = now_ms();
        (row.weights, row.weights_floor) = (weights, floor);
    }

    /// The largest activation growth any shape of `plan` measured.
    pub fn peak(&self, plan: &str) -> Option<u64> {
        let plan = self.plans.get(plan)?;
        plan.shapes
            .values()
            .map(Shape::bytes)
            .max()
            .filter(|peak| *peak > 0)
    }

    pub fn host(&mut self, plan: &str, bytes: u64) {
        let row = self.plans.entry(plan.into()).or_default();
        row.host_bytes = row.host_bytes.max(bytes);
    }

    /// `plan`'s measurements for `shape`: exactly that shape, else the least among measured
    /// shapes at least as large on every axis (memory does not shrink as a shape grows).
    pub fn shape(&self, plan: &str, shape: &str) -> Option<&Shape> {
        let plan = self.plans.get(plan)?;
        if let Some(exact) = plan.shapes.get(shape) {
            return Some(exact);
        }
        let mine = axes(shape)?;
        plan.shapes
            .iter()
            .filter(|(other, _)| {
                axes(other).is_some_and(|theirs| {
                    theirs.len() == mine.len()
                        && mine
                            .iter()
                            .all(|(axis, value)| theirs.get(axis).is_some_and(|t| t >= value))
                })
            })
            .map(|(_, measured)| measured)
            .min_by_key(|measured| measured.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shape_never_seen_takes_the_least_larger_one_and_survives_a_restart() {
        let path = std::env::temp_dir().join(format!("learned-{}.json", uuid::Uuid::new_v4()));
        let mut learned = Learned::open(&path);
        let methods = BTreeMap::from([("transformer".to_string(), 3 << 30)]);
        learned.call("anima", "height=1536,width=1536", 3 << 30, &methods);
        learned.call("anima", "height=2048,width=2048", 5 << 30, &methods);
        learned.context("GPU-1/580", 220 << 20);
        learned.load("anima", 6 << 30, 2 << 30);
        learned.save().unwrap();
        let learned = Learned::open(&path);
        let anima = &learned.plans["anima"];
        assert_eq!((anima.weights, anima.weights_floor), (6 << 30, 2 << 30));
        assert_eq!(learned.peak("anima"), Some(5 << 30));
        assert_eq!(
            learned
                .shape("anima", "height=1024,width=1024")
                .unwrap()
                .peak,
            3 << 30
        );
        assert_eq!(
            learned
                .shape("anima", "height=1536,width=1536")
                .unwrap()
                .methods,
            methods
        );
        assert!(learned.shape("anima", "height=4096,width=4096").is_none());
        assert_eq!(learned.contexts["GPU-1/580"], 220 << 20);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn features_spell_the_runtime_shape_cell() {
        let features = BTreeMap::from([
            ("width".to_string(), serde_json::json!(1024)),
            ("height".to_string(), serde_json::json!(1024)),
        ]);
        assert_eq!(shape_cell(&features), "height=1024,width=1024");
        assert_eq!(shape_cell(&BTreeMap::new()), "-");
    }
}
