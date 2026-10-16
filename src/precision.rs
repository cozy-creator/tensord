//! The lane rule (xs-042, design `precision-tiers.md` §2.2, §2.7, §2.8): a model slot's lane
//! from its GPUs' native formats, the request's bits, the slot's declared tolerances and the
//! memory budget. Nothing is keyed by a card's name.
use serde_json::Value;

/// The lanes the rule picks by name; any other is a release's only lane, served silently.
const VOCABULARY: [&str; 5] = ["bf16", "fp16", "int8", "fp8", "nvfp4"];

/// The native formats of a set of GPUs (design §2.1): `mma` is cc 8.0 or later, where `nvfp4`
/// serves at least weight-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Row {
    pub mma: bool,
    pub int8: bool,
    pub fp8: bool,
}

impl Row {
    /// One GPU's row from its compute capability (`"9.0"`); none below 8.0 or unread. 10.3 has
    /// no int8 (`tcgen05.mma.kind::i8` is not in its target).
    pub fn of(cc: &str) -> Self {
        let parsed = cc.split_once('.').and_then(|(major, minor)| Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?));
        let cc = parsed.unwrap_or(0);
        Self { mma: cc >= 80, int8: cc >= 80 && cc != 103, fp8: cc >= 89 }
    }
}

/// The GPUs a function runs on, as the rule reads them: their row, each column ANDed (§2.7),
/// and how a warning names them (`sm90`, `sm86+sm89`).
#[derive(Clone, Debug, PartialEq)]
pub struct Gpus {
    pub row: Row,
    pub name: String,
}

impl Gpus {
    pub fn of(ccs: &[&str]) -> Self {
        let row = ccs.iter().map(|cc| Row::of(cc)).reduce(|a, b| Row { mma: a.mma && b.mma, int8: a.int8 && b.int8, fp8: a.fp8 && b.fp8 });
        let mut names: Vec<String> = ccs.iter().filter(|cc| !cc.is_empty()).map(|cc| format!("sm{}", cc.replace('.', ""))).collect();
        names.sort();
        names.dedup();
        Self { row: row.unwrap_or_default(), name: if names.is_empty() { "no GPU".into() } else { names.join("+") } }
    }
}

/// A slot's declared tolerances (package interface): `min_bits` (8 when omitted), its phase and
/// whether its class takes encoded leaves.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub min_bits: u64,
    pub decode: bool,
    pub encoded: bool,
}

impl Slot {
    pub fn of(slot: &Value) -> Self {
        Self {
            min_bits: slot["min_bits"].as_u64().unwrap_or(8),
            decode: slot["phase"] == "decode",
            encoded: slot["encoded_leaves"] == "accept",
        }
    }

    /// Its bits (§2.2): the request's; else 16 at `min_bits=16`; else 4 at `min_bits=4` for a
    /// decode slot or a function whose 8-bit weights exceed the budget (`short`); else 8.
    pub fn bits(&self, asked: u32, short: bool) -> u32 {
        match asked {
            4 | 8 | 16 => asked,
            _ if self.min_bits == 16 => 16,
            _ if self.min_bits == 4 && (self.decode || short) => 4,
            _ => 8,
        }
    }

    /// The lanes to ask for at `bits`, in order (§2.2): the width's native formats (4 bits: `nvfp4`
    /// first, then the 8-bit order), then 16 bits. Encoded lanes go when the slot declines them.
    pub fn lanes(&self, bits: u32, row: Row) -> Vec<&'static str> {
        let eight = [("int8", row.int8), ("fp8", row.fp8)];
        let native: Vec<_> = match bits {
            16 => vec![],
            4 => [("nvfp4", row.mma)].into_iter().chain(eight).collect(),
            _ => eight.to_vec(),
        };
        let native = native.into_iter().filter(|(_, runs)| *runs && self.encoded).map(|(lane, _)| lane);
        native.chain(["bf16", "fp16"]).collect()
    }

    /// The run's warnings for `model` served in `served` (§2.8): one when it declined encoded
    /// lanes the row runs, and one per lane asked before the one served.
    pub fn warnings(&self, bits: u32, gpus: &Gpus, model: &str, served: &str) -> Vec<String> {
        if !VOCABULARY.contains(&served) {
            return vec![];
        }
        let declined = !self.encoded && Self { encoded: true, ..*self }.lanes(bits, gpus.row).len() > 2;
        let declined = declined.then(|| format!("precision {bits}: {model} declines encoded leaves; served {served}"));
        let asked = self.lanes(bits, gpus.row);
        let missing = asked.into_iter().take_while(|lane| *lane != served);
        let missing = missing.map(|lane| format!("precision {bits}: {model} has no {lane} lane for {}; served {served}", gpus.name));
        declined.into_iter().chain(missing).collect()
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
        let first = |slot: &Slot, asked, cc: &[&str]| slot.lanes(slot.bits(asked, false), Gpus::of(cc).row)[0];
        for (cc, eight) in [("8.0", "int8"), ("8.6", "int8"), ("8.9", "int8"), ("9.0", "int8"), ("10.0", "int8"),
            ("10.3", "fp8"), ("12.0", "int8"), ("12.1", "int8"), ("11.0", "int8")] {
            assert_eq!(first(&compute, 0, &[cc]), eight, "{cc}");
            assert_eq!(first(&decode, 0, &[cc]), "nvfp4", "{cc}");
            assert_eq!(first(&compute, 4, &[cc]), "nvfp4", "{cc}");
            assert_eq!(first(&compute, 16, &[cc]), "bf16", "{cc}");
        }
        for none in [&["7.5"][..], &[""], &[]] {
            for (slot, asked) in [(&compute, 0), (&decode, 0), (&compute, 4), (&compute, 16)] {
                assert_eq!(first(slot, asked, none), "bf16", "{none:?}");
            }
        }
        assert_eq!(compute.lanes(8, Row::of("9.0")), ["int8", "fp8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, Row::of("8.6")), ["nvfp4", "int8", "bf16", "fp16"]);
        let exact = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 16}));
        assert_eq!((exact.bits(0, true), exact.bits(8, false)), (16, 8), "the request's bits win");
    }

    /// A set's row is each column ANDed: a mixed local machine runs only what every GPU runs.
    #[test]
    fn a_mixed_set_runs_what_every_gpu_runs() {
        let slot = Slot::of(&json!({"encoded_leaves": "accept"}));
        let ampere_ada = Gpus::of(&["8.6", "8.9"]);
        assert_eq!(ampere_ada.name, "sm86+sm89");
        assert_eq!(slot.lanes(8, ampere_ada.row), ["int8", "bf16", "fp16"]);
        assert_eq!(slot.lanes(8, Gpus::of(&["10.0", "10.3"]).row), ["fp8", "bf16", "fp16"]);
        assert_eq!(slot.lanes(8, Gpus::of(&["9.0", "7.5"]).row), ["bf16", "fp16"]);
        assert_eq!(Gpus::of(&["9.0", "9.0"]).name, "sm90");
    }

    /// The fallback steps warn once each; a slot declining encoded leaves warns once; a served
    /// lane outside the vocabulary and an exact slot say nothing.
    #[test]
    fn steps_and_declines_warn() {
        let (hopper, ada) = (Gpus::of(&["9.0"]), Gpus::of(&["8.9"]));
        let slot = Slot::of(&json!({"encoded_leaves": "accept"}));
        assert_eq!(slot.warnings(4, &ada, "fidika/sdxl@1.0.0", "int8"),
            ["precision 4: fidika/sdxl@1.0.0 has no nvfp4 lane for sm89; served int8"]);
        assert_eq!(slot.warnings(8, &hopper, "x@1", "bf16").len(), 2);
        assert!(slot.warnings(8, &hopper, "x@1", "int8").is_empty());
        assert!(slot.warnings(8, &hopper, "x@1", "pdd8").is_empty());
        let declines = Slot::of(&json!({}));
        assert_eq!(declines.lanes(8, hopper.row), ["bf16", "fp16"]);
        assert_eq!(declines.warnings(8, &hopper, "x@1", "bf16"), ["precision 8: x@1 declines encoded leaves; served bf16"]);
        assert!(declines.warnings(16, &hopper, "x@1", "bf16").is_empty());
        assert!(declines.warnings(8, &Gpus::of(&[]), "x@1", "bf16").is_empty());
    }
}
