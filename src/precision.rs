//! The lane rule (xs-042, design `precision-tiers.md` §2.2, §2.7, §2.8): a model slot's lane
//! from its GPUs' native formats, the request's bits, the slot's declared tolerances and the
//! memory budget. Nothing is keyed by a card's name.
use serde_json::Value;

/// The lanes the rule picks by name; any other is a release's only lane, served silently.
const VOCABULARY: [&str; 5] = ["bf16", "fp16", "int8", "fp8", "nvfp4"];

/// The GPUs a function runs on as the rule reads them (design §2.1, §2.7): the native formats
/// every one runs (`mma`: cc 8.0 or later, where `nvfp4` serves at least weight-only; 10.3 has
/// no int8; `bf16`: unless a GPU is known below cc 8.0) and how a warning names them (`sm90`, `sm86+sm89`).
#[derive(Clone, Debug, PartialEq)]
pub struct Gpus {
    pub mma: bool,
    pub bf16: bool,
    pub int8: bool,
    pub fp8: bool,
    pub name: String,
}

/// No GPU: the CPU, which runs bf16 and no encoded format.
impl Default for Gpus {
    fn default() -> Self {
        Self::of(&[])
    }
}

impl Gpus {
    /// From each GPU's compute capability (`"9.0"`); one below 8.0 or unread runs none.
    pub fn of(ccs: &[&str]) -> Self {
        let number = |cc: &str| cc.split_once('.').and_then(|(major, minor)| Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?));
        let all = |runs: fn(u32) -> bool| !ccs.is_empty() && ccs.iter().all(|cc| runs(number(cc).unwrap_or(0)));
        let mut names: Vec<String> = ccs.iter().filter(|cc| !cc.is_empty()).map(|cc| format!("sm{}", cc.replace('.', ""))).collect();
        names.sort();
        names.dedup();
        let name = if names.is_empty() { "no GPU".into() } else { names.join("+") };
        // 12.1 (GB10) runs no fp8 until a card proves it (owner, 2026-10-16).
        Self {
            mma: all(|cc| cc >= 80),
            // fp16 first only where a GPU is known to run no bf16: an unread cc, or no GPU, runs bf16.
            bf16: !ccs.iter().any(|cc| matches!(number(cc), Some(n) if n < 80)),
            int8: all(|cc| cc >= 80 && cc != 103),
            fp8: all(|cc| cc >= 89 && cc != 121),
            name,
        }
    }
}

/// A slot's declared tolerances (package interface): `min_bits` (8 when omitted), its phase,
/// whether its class takes encoded leaves, and whether it is raw weights (`Weights[M]`, read as
/// data: always 16 bits, whatever its class declares).
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub min_bits: u64,
    pub decode: bool,
    pub encoded: bool,
    pub raw: bool,
}

impl Slot {
    pub fn of(slot: &Value) -> Self {
        Self {
            min_bits: slot["min_bits"].as_u64().unwrap_or(8),
            decode: slot["phase"] == "decode",
            encoded: slot["encoded_leaves"] == "accept",
            raw: slot["weights"] == true,
        }
    }

    /// Its bits (§2.2): the request's; else 16 at `min_bits=16`; else 4 at `min_bits=4` for a
    /// decode slot or a function whose 8-bit weights exceed the budget (`short`); else 8.
    pub fn bits(&self, asked: u32, short: bool) -> u32 {
        match asked {
            _ if self.raw => 16,
            4 | 8 | 16 => asked,
            _ if self.min_bits == 16 => 16,
            _ if self.min_bits == 4 && (self.decode || short) => 4,
            _ => 8,
        }
    }

    /// The lanes to ask for at `bits`, in order (§2.2): the width's native formats (4 bits: `nvfp4`
    /// first, then the 8-bit order), then 16 bits: `bf16`, and `fp16` first only where the GPUs
    /// run no bf16 (owner, 2026-10-16). Encoded lanes go when the slot declines them.
    pub fn lanes(&self, bits: u32, gpus: &Gpus) -> Vec<&'static str> {
        let eight = [("int8", gpus.int8), ("fp8", gpus.fp8)];
        let native: Vec<_> = match bits {
            16 => vec![],
            4 => [("nvfp4", gpus.mma)].into_iter().chain(eight).collect(),
            _ => eight.to_vec(),
        };
        let native = native.into_iter().filter(|(_, runs)| *runs && self.encoded).map(|(lane, _)| lane);
        native.chain(if gpus.bf16 { ["bf16", "fp16"] } else { ["fp16", "bf16"] }).collect()
    }

    /// The run's warnings for `model` served in `served` (§2.8): one when it declined encoded
    /// lanes the GPUs run, one naming every lane asked before the one served, and fp16 served
    /// because the GPUs run no bf16 says so.
    pub fn warnings(&self, bits: u32, gpus: &Gpus, model: &str, served: &str) -> Vec<String> {
        if self.raw || !VOCABULARY.contains(&served) {
            return vec![];
        }
        let declined = !self.encoded && Self { encoded: true, ..*self }.lanes(bits, gpus).len() > 2;
        let declined = declined.then(|| format!("precision {bits}: {model} declines encoded leaves; served {served}"));
        let mut missing: Vec<&str> = self.lanes(bits, gpus).into_iter().take_while(|lane| *lane != served).collect();
        let missing = missing.pop().map(|last| match missing.is_empty() {
            true => last.to_string(),
            false => format!("{} or {last}", missing.join(", ")),
        });
        let missing = missing.map(|lanes| format!("precision {bits}: {model} has no {lanes} lane for {}; served {served}", gpus.name));
        let no_bf16 = (served == "fp16" && !gpus.bf16).then(|| format!("precision {bits}: {} runs no bf16; served fp16", gpus.name));
        declined.into_iter().chain(missing).chain(no_bf16).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Each row's lane for a compute slot, a `min_bits=4` decode slot, `--bits=4` and
    /// `--bits=16` (design §2.2's derived table).
    #[test]
    fn one_lane_per_row() {
        let compute = Slot::of(&json!({"encoded_leaves": "accept"}));
        let decode = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 4, "phase": "decode"}));
        let first = |slot: &Slot, asked, cc: &[&str]| slot.lanes(slot.bits(asked, false), &Gpus::of(cc))[0];
        for (cc, eight) in [("8.0", "int8"), ("8.6", "int8"), ("8.9", "int8"), ("9.0", "int8"), ("10.0", "int8"),
            ("10.3", "fp8"), ("12.0", "int8"), ("12.1", "int8"), ("11.0", "int8")] {
            assert_eq!(first(&compute, 0, &[cc]), eight, "{cc}");
            assert_eq!(first(&decode, 0, &[cc]), "nvfp4", "{cc}");
            assert_eq!(first(&compute, 4, &[cc]), "nvfp4", "{cc}");
            assert_eq!(first(&compute, 16, &[cc]), "bf16", "{cc}");
        }
        // bf16 everywhere it runs (cc 8.0 or later, or no GPU); fp16 first only below 8.0.
        for (none, sixteen) in [(&["7.5"][..], "fp16"), (&[""], "bf16"), (&[], "bf16")] {
            for (slot, asked) in [(&compute, 0), (&decode, 0), (&compute, 4), (&compute, 16)] {
                assert_eq!(first(slot, asked, none), sixteen, "{none:?}");
            }
        }
        assert_eq!(compute.lanes(8, &Gpus::of(&["9.0"])), ["int8", "fp8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, &Gpus::of(&["8.6"])), ["nvfp4", "int8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, &Gpus::of(&["12.0"])), ["nvfp4", "int8", "fp8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, &Gpus::of(&["12.1"])), ["nvfp4", "int8", "bf16", "fp16"], "12.1 runs no fp8 yet");
        let exact = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 16}));
        assert_eq!((exact.bits(0, true), exact.bits(8, false)), (16, 8), "the request's bits win");
        let raw = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 4, "phase": "decode", "weights": true}));
        assert_eq!((raw.bits(4, true), raw.lanes(raw.bits(8, false), &Gpus::of(&["10.0"]))), (16, vec!["bf16", "fp16"]), "raw weights are data");
        assert!(raw.warnings(16, &Gpus::of(&["9.0"]), "x@1", "fp16").is_empty());
    }

    /// A set's row is each column ANDed: a mixed local machine runs only what every GPU runs.
    #[test]
    fn a_mixed_set_runs_what_every_gpu_runs() {
        let slot = Slot::of(&json!({"encoded_leaves": "accept"}));
        let ampere_ada = Gpus::of(&["8.6", "8.9"]);
        assert_eq!(ampere_ada.name, "sm86+sm89");
        assert_eq!(slot.lanes(8, &ampere_ada), ["int8", "bf16", "fp16"]);
        assert_eq!(slot.lanes(8, &Gpus::of(&["10.0", "10.3"])), ["fp8", "bf16", "fp16"]);
        assert_eq!(slot.lanes(8, &Gpus::of(&["9.0", "7.5"])), ["fp16", "bf16"]);
        assert_eq!(Gpus::of(&["9.0", "9.0"]).name, "sm90");
    }

    /// A fallback warns once, naming every lane it lacked; a slot declining encoded leaves warns
    /// once; a served lane outside the vocabulary and an exact slot say nothing.
    #[test]
    fn steps_and_declines_warn() {
        let (hopper, ada) = (Gpus::of(&["9.0"]), Gpus::of(&["8.9"]));
        let slot = Slot::of(&json!({"encoded_leaves": "accept"}));
        assert_eq!(slot.warnings(4, &ada, "fidika/sdxl@1.0.0", "int8"),
            ["precision 4: fidika/sdxl@1.0.0 has no nvfp4 lane for sm89; served int8"]);
        assert_eq!(slot.warnings(8, &hopper, "x@1", "bf16"), ["precision 8: x@1 has no int8 or fp8 lane for sm90; served bf16"]);
        assert_eq!(slot.warnings(4, &ada, "x@1", "fp16"), ["precision 4: x@1 has no nvfp4, int8, fp8 or bf16 lane for sm89; served fp16"]);
        assert!(slot.warnings(8, &hopper, "x@1", "int8").is_empty());
        assert!(slot.warnings(8, &hopper, "x@1", "pdd8").is_empty());
        let declines = Slot::of(&json!({}));
        assert_eq!(declines.lanes(8, &hopper), ["bf16", "fp16"]);
        assert_eq!(declines.warnings(8, &hopper, "x@1", "bf16"), ["precision 8: x@1 declines encoded leaves; served bf16"]);
        assert!(declines.warnings(16, &hopper, "x@1", "bf16").is_empty());
        assert!(declines.warnings(8, &Gpus::of(&[]), "x@1", "bf16").is_empty());
        assert_eq!(Gpus::default(), Gpus::of(&[]), "no GPU service is the CPU: bf16 first");
        // fp16 says why: the GPUs run no bf16, or the release has no bf16 lane.
        let turing = Gpus::of(&["7.5"]);
        assert_eq!(slot.warnings(8, &turing, "x@1", "fp16"), ["precision 8: sm75 runs no bf16; served fp16"]);
        assert_eq!(slot.warnings(8, &turing, "x@1", "bf16"), ["precision 8: x@1 has no fp16 lane for sm75; served bf16"]);
        assert_eq!(slot.warnings(16, &hopper, "sdxl@1", "fp16"), ["precision 16: sdxl@1 has no bf16 lane for sm90; served fp16"]);
    }
}
