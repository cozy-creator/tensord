//! What executors measured, kept across executors and machine restarts: the CUDA context per
//! GPU and driver, and each plan's activation growth per request shape (per stage method and
//! for the whole call). Estimates come from these, not from constants; a plan unused for
//! `TTL` is forgotten.
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const TTL: Duration = Duration::from_secs(30 * 24 * 3600);

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Shape {
    /// The call's peak activation growth.
    pub peak: u64,
    /// Each stage method's.
    pub methods: BTreeMap<String, u64>,
    /// An estimate, never a measurement: the measured shape it was scaled from, and the ratio
    /// of their token counts. Never written back (`call` stores measurements only).
    #[serde(skip)]
    pub estimated_from: Option<(String, f64)>,
}
impl Shape {
    /// What the call grows by at its worst: its own peak or a stage's (a stage counts the
    /// segments torch reserved for it), whichever is more.
    pub fn bytes(&self) -> u64 {
        self.methods.values().copied().fold(self.peak, u64::max)
    }

    /// This shape's peaks scaled by `ratio`, as an estimate from `cell`.
    fn scaled(&self, cell: &str, ratio: f64) -> Self {
        let scale = |bytes: u64| (bytes as f64 * ratio).ceil() as u64;
        Self {
            peak: scale(self.peak),
            methods: self
                .methods
                .iter()
                .map(|(method, bytes)| (method.clone(), scale(*bytes)))
                .collect(),
            estimated_from: Some((cell.into(), ratio)),
        }
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
    /// Custody's names (without the generation) of the weight sets its executor asked the
    /// machine for or offered it: what a new executor of it attaches when they are held.
    pub holdings: BTreeSet<String>,
    pub used_ms: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Learned {
    /// GPU UUID and driver version -> the largest context an executor measured there.
    pub contexts: BTreeMap<String, u64>,
    /// Provenance for context estimates; legacy device-wide deltas must not train admission.
    pub context_measurements: BTreeMap<String, String>,
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

/// The tokens a shape puts through a model: frames times pixels (else width times height) times
/// listed inputs. Steps change time, not memory, so they are not counted.
fn tokens(axes: &BTreeMap<&str, i64>) -> f64 {
    let axis = |name| axes.get(name).copied().unwrap_or(1).max(1) as f64;
    let pixels = match axes.get("pixels") {
        Some(_) => axis("pixels"),
        None => axis("width") * axis("height"),
    };
    axis("frames") * pixels * axis("assets")
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
        learned.contexts.retain(|device, _| {
            learned.context_measurements.get(device).map(String::as_str) == Some("process_driver")
        });
        learned
            .context_measurements
            .retain(|device, _| learned.contexts.contains_key(device));
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
        self.context_measurements
            .insert(device.into(), "process_driver".into());
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

    /// `plan`'s executor named weight set `holding` to custody; whether that is news.
    pub fn holding(&mut self, plan: &str, holding: &str) -> bool {
        let row = self.plans.entry(plan.into()).or_default();
        row.used_ms = now_ms();
        row.holdings.insert(holding.into())
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
        row.used_ms = now_ms();
        row.host_bytes = row.host_bytes.max(bytes);
    }

    /// `plan`'s measurements for `shape`: exactly that shape, else the least among measured
    /// shapes at least as large on every axis (memory does not shrink as a shape grows), else
    /// an estimate: the largest measured shape no larger on any axis, its peaks scaled by the
    /// ratio of token counts (segment 2 of a long-form holds 384 frames where segment 1 held
    /// 362, run 4074). Linear in tokens; a kernel that grows faster is caught by recovery, and
    /// its first measurement replaces the estimate.
    pub fn shape(&self, plan: &str, shape: &str) -> Option<Shape> {
        let plan = self.plans.get(plan)?;
        if let Some(exact) = plan.shapes.get(shape) {
            return Some(exact.clone());
        }
        let mine = axes(shape)?;
        let measured = || {
            plan.shapes.iter().filter_map(|(cell, measured)| {
                let theirs = axes(cell)?;
                (theirs.len() == mine.len() && mine.keys().all(|axis| theirs.contains_key(axis)))
                    .then_some((cell, theirs, measured))
            })
        };
        let larger = measured()
            .filter(|(_, theirs, _)| mine.iter().all(|(axis, value)| theirs[axis] >= *value))
            .map(|(_, _, measured)| measured)
            .min_by_key(|measured| measured.bytes());
        if let Some(larger) = larger {
            return Some(larger.clone());
        }
        let (cell, theirs, measured) = measured()
            .filter(|(_, theirs, _)| mine.iter().all(|(axis, value)| theirs[axis] <= *value))
            .max_by(|a, b| tokens(&a.1).total_cmp(&tokens(&b.1)))?;
        Some(measured.scaled(cell, tokens(&mine) / tokens(&theirs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_contexts_leave_without_discarding_valid_plan_or_attributed_context_data() {
        let path = std::env::temp_dir().join(format!("learned-{}.json", uuid::Uuid::new_v4()));
        let mut learned = Learned::open(&path);
        learned.call("anima", "height=512,width=512", 400, &BTreeMap::from([("render".into(), 500)]));
        learned.load("anima", 600, 100);
        learned.context("trusted/driver", 200);
        learned.contexts.insert("legacy/driver".into(), 7000);
        learned.save().unwrap();
        let restored = Learned::open(&path);
        assert_eq!(restored.contexts, BTreeMap::from([("trusted/driver".into(), 200)]));
        assert_eq!(restored.plans["anima"].weights, 600);
        assert_eq!(restored.plans["anima"].weights_floor, 100);
        assert_eq!(restored.peak("anima"), Some(500));
        assert_eq!(restored.shape("anima", "height=512,width=512").unwrap().peak, 400);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_shape_never_seen_takes_the_least_larger_one_and_survives_a_restart() {
        let path = std::env::temp_dir().join(format!("learned-{}.json", uuid::Uuid::new_v4()));
        let mut learned = Learned::open(&path);
        let methods = BTreeMap::from([("transformer".to_string(), 3 << 30)]);
        learned.call("anima", "height=1536,width=1536", 3 << 30, &methods);
        learned.call("anima", "height=2048,width=2048", 5 << 30, &methods);
        learned.context("GPU-1/580", 220 << 20);
        learned.load("anima", 6 << 30, 2 << 30);
        // A row that only ever measured host bytes (an import-only parent) is kept too.
        learned.host("parent:abc", 600 << 20);
        learned.save().unwrap();
        let learned = Learned::open(&path);
        assert_eq!(learned.plans["parent:abc"].host_bytes, 600 << 20);
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
        // Larger than anything measured: the largest shape below it, scaled by its pixels.
        let estimate = learned.shape("anima", "height=4096,width=4096").unwrap();
        assert_eq!(estimate.peak, 20 << 30);
        assert_eq!(estimate.methods["transformer"], 12 << 30);
        assert_eq!(
            estimate.estimated_from,
            Some(("height=2048,width=2048".to_string(), 4.0))
        );
        // A measured shape is never an estimate.
        assert!(learned
            .shape("anima", "height=2048,width=2048")
            .unwrap()
            .estimated_from
            .is_none());
        assert_eq!(learned.contexts["GPU-1/580"], 220 << 20);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_longer_segment_scales_by_frames_and_never_by_steps() {
        let mut learned = Learned::default();
        let methods = BTreeMap::from([("base_model.sample_ref2va_turbo".to_string(), 362)]);
        learned.call("h3", "assets=1,frames=362,steps=8", 100, &methods);
        let estimate = learned.shape("h3", "assets=1,frames=384,steps=8").unwrap();
        assert_eq!(estimate.methods["base_model.sample_ref2va_turbo"], 384);
        assert_eq!(estimate.peak, 107); // ceil(100 * 384 / 362)
        // More steps alone: tokens are equal, so the estimate is the measurement's own size.
        let steps = learned.shape("h3", "assets=1,frames=362,steps=30").unwrap();
        assert_eq!((steps.peak, steps.estimated_from.unwrap().1), (100, 1.0));
        // A smaller shape takes the measurement itself, never a scaled-down estimate.
        let smaller = learned.shape("h3", "assets=1,frames=300,steps=8").unwrap();
        assert_eq!((smaller.peak, smaller.estimated_from), (100, None));
        // Larger on one axis and smaller on another, or other axes: no estimate.
        assert!(learned.shape("h3", "assets=2,frames=384,steps=4").is_none());
        assert!(learned.shape("h3", "frames=384").is_none());
        // Estimates are read-side only: nothing new was stored.
        assert_eq!(learned.plans["h3"].shapes.len(), 1);
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
