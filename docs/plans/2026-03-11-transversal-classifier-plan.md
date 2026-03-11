# Transversal Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an 8-sound calibration mode that reconstructs all 16 Q8 nibbles using orthogonal Latin square transversals, with accuracy benchmarks comparing it to the existing 16-sound mode.

**Architecture:** A new `TransversalClassifier` in `stq8-core/src/transversal.rs` computes cosine similarity against 8 calibration centroids (two phrases of 4 syllables), takes the Top-2 from each phrase, and intersects consonant/vowel sets to reconstruct the intended nibble. `Pipeline` dispatches to either the existing `NearestCentroid` or `TransversalClassifier` based on a `CalibrationMode` enum in `UserProfile`. WASM bindings expose the new calibration methods.

**Tech Stack:** Rust (stq8-core crate, no_std-compatible), wasm-bindgen (stq8-web crate), serde for serialization

---

### Task 1: Make `cosine_similarity` crate-visible

**Files:**
- Modify: `stq8-core/src/classifier.rs:76`

**Step 1: Change visibility**

Change line 76 from:
```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
```
to:
```rust
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
```

**Step 2: Verify existing tests still pass**

Run: `cargo test -p stq8-core`
Expected: All tests pass (no public API change, only crate-internal visibility)

**Step 3: Commit**

```bash
git add stq8-core/src/classifier.rs
git commit -m "refactor(classifier): make cosine_similarity pub(crate) for transversal reuse"
```

---

### Task 2: Add `CalibrationMode` to profile

**Files:**
- Modify: `stq8-core/src/profile.rs`
- Modify: `stq8-core/src/lib.rs`

**Step 1: Write the failing test**

Add to `stq8-core/src/profile.rs` in the `tests` module:

```rust
#[test]
fn calibration_mode_default_is_full() {
    let profile = UserProfile {
        version: 1,
        centroids: Vec::new(),
        thresholds: Thresholds::default(),
        custom_map: Vec::new(),
        created_epoch_secs: 0,
        calibration_mode: CalibrationMode::default(),
    };
    assert_eq!(profile.calibration_mode, CalibrationMode::Full);
}

#[test]
fn calibration_mode_serialization_roundtrip() {
    let profile = UserProfile {
        version: 1,
        centroids: Vec::new(),
        thresholds: Thresholds::default(),
        custom_map: Vec::new(),
        created_epoch_secs: 0,
        calibration_mode: CalibrationMode::Transversal,
    };
    let json = serde_json::to_string(&profile).expect("serialize");
    let restored: UserProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.calibration_mode, CalibrationMode::Transversal);
}

#[test]
fn calibration_mode_defaults_to_full_when_missing_in_json() {
    // Backward compat: old profiles without calibration_mode should deserialize as Full
    let json = r#"{"version":1,"centroids":[],"thresholds":{"auto_accept":0.85,"reject":0.40},"custom_map":[],"created_epoch_secs":0}"#;
    let profile: UserProfile = serde_json::from_str(json).expect("deserialize old format");
    assert_eq!(profile.calibration_mode, CalibrationMode::Full);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p stq8-core -- calibration_mode`
Expected: FAIL — `CalibrationMode` doesn't exist yet

**Step 3: Implement CalibrationMode and add to UserProfile**

In `stq8-core/src/profile.rs`, add before the `Thresholds` struct:

```rust
/// Which calibration strategy was used to train this profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMode {
    /// 16-sound: all CV pairs calibrated directly
    Full,
    /// 8-sound: two orthogonal transversal phrases, reconstruct via intersection
    Transversal,
}

impl Default for CalibrationMode {
    fn default() -> Self {
        Self::Full
    }
}
```

Add the field to `UserProfile`:

```rust
pub struct UserProfile {
    pub version: u8,
    pub centroids: Vec<(Syllable, Vec<f32>)>,
    pub thresholds: Thresholds,
    pub custom_map: Vec<(Phoneme, Phoneme)>,
    pub created_epoch_secs: u64,
    #[serde(default)]
    pub calibration_mode: CalibrationMode,
}
```

The `#[serde(default)]` attribute makes old JSON profiles (without this field) deserialize as `CalibrationMode::Full`.

**Step 4: Fix compilation errors**

The `Pipeline::new()` in `pipeline.rs` constructs a `UserProfile` directly. Add the new field:

```rust
calibration_mode: CalibrationMode::default(),
```

The `sample_profile()` helper in `profile.rs` tests also constructs `UserProfile` directly — add the field there too.

**Step 5: Run tests to verify they pass**

Run: `cargo test -p stq8-core`
Expected: All tests pass, including the three new ones

**Step 6: Commit**

```bash
git add stq8-core/src/profile.rs stq8-core/src/pipeline.rs
git commit -m "feat(profile): add CalibrationMode enum (Full/Transversal) with backward compat"
```

---

### Task 3: Create `transversal.rs` with phrase constants and reconstruction logic

**Files:**
- Create: `stq8-core/src/transversal.rs`
- Modify: `stq8-core/src/lib.rs`

This is the core algorithmic task. The transversal classifier uses two orthogonal phrases (each covering all 4 consonants and all 4 vowels exactly once) to reconstruct any of the 16 possible nibbles.

**Step 1: Write failing tests**

Create `stq8-core/src/transversal.rs` with tests first:

```rust
//! Transversal classifier: 8-sound calibration using orthogonal Latin square transversals.
//!
//! Two 4-syllable phrases each cover all 4 consonants and all 4 vowels exactly once.
//! At inference time, Top-2-per-phrase intersection reconstructs any of the 16 nibbles.

use crate::classifier::cosine_similarity;
use crate::mfcc::FEATURE_DIM;
use crate::q8::{Consonant, Syllable, Vowel};
use serde::{Deserialize, Serialize};

/// Phrase 1 (diagonal): 'O(0), JU(5), KE(10), VI(15)
pub const PHRASE_1: [Syllable; 4] = [
    Syllable { consonant: Consonant::GlottalStop, vowel: Vowel::O },
    Syllable { consonant: Consonant::J, vowel: Vowel::U },
    Syllable { consonant: Consonant::K, vowel: Vowel::E },
    Syllable { consonant: Consonant::V, vowel: Vowel::I },
];

/// Phrase 2 (anti-diagonal): 'I(3), JE(6), KO(8), VU(13)
pub const PHRASE_2: [Syllable; 4] = [
    Syllable { consonant: Consonant::GlottalStop, vowel: Vowel::I },
    Syllable { consonant: Consonant::J, vowel: Vowel::E },
    Syllable { consonant: Consonant::K, vowel: Vowel::O },
    Syllable { consonant: Consonant::V, vowel: Vowel::U },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(syllable_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; FEATURE_DIM];
        let base = (syllable_idx * 3) % FEATURE_DIM;
        v[base] = 1.0;
        v[(base + 1) % FEATURE_DIM] = 0.5;
        v[(base + 2) % FEATURE_DIM] = 0.8;
        v
    }

    fn train_transversal() -> TransversalClassifier {
        let mut tc = TransversalClassifier::new();
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

        let mut samples = Vec::new();
        for &idx in &phrase1_indices {
            samples.push((0u8, Syllable::from_nibble(idx), make_features(idx as usize)));
        }
        for &idx in &phrase2_indices {
            samples.push((1u8, Syllable::from_nibble(idx), make_features(idx as usize)));
        }
        tc.train(&samples);
        tc
    }

    #[test]
    fn phrases_cover_all_consonants_and_vowels() {
        for phrase in [&PHRASE_1, &PHRASE_2] {
            let consonants: std::collections::HashSet<_> = phrase.iter().map(|s| s.consonant).collect();
            let vowels: std::collections::HashSet<_> = phrase.iter().map(|s| s.vowel).collect();
            assert_eq!(consonants.len(), 4, "phrase must cover all 4 consonants");
            assert_eq!(vowels.len(), 4, "phrase must cover all 4 vowels");
        }
    }

    #[test]
    fn phrases_are_orthogonal() {
        // No syllable appears in both phrases
        let p1_set: std::collections::HashSet<_> = PHRASE_1.iter().collect();
        let p2_set: std::collections::HashSet<_> = PHRASE_2.iter().collect();
        assert!(p1_set.is_disjoint(&p2_set), "phrases must not share any syllable");
    }

    #[test]
    fn untrained_returns_none() {
        let tc = TransversalClassifier::new();
        let features = vec![0.0; FEATURE_DIM];
        assert!(tc.classify(&features).is_none());
    }

    #[test]
    fn reconstructs_all_16_nibbles() {
        let tc = train_transversal();

        for nibble in 0..16u8 {
            let features = make_features(nibble as usize);
            let result = tc.classify(&features).unwrap_or_else(|| {
                panic!("nibble {nibble} should classify")
            });
            assert_eq!(
                result.syllable.to_nibble(), nibble,
                "nibble {nibble} ({}) should reconstruct correctly, got {} ({})",
                Syllable::from_nibble(nibble), result.syllable, result.syllable.to_nibble()
            );
        }
    }

    #[test]
    fn calibrated_syllables_have_high_confidence() {
        let tc = train_transversal();
        // Calibrated syllables (exact centroid match) should have high confidence
        for &idx in &[0u8, 5, 10, 15, 3, 6, 8, 13] {
            let features = make_features(idx as usize);
            let result = tc.classify(&features).expect("should classify");
            assert!(
                result.confidence > 0.3,
                "calibrated nibble {idx} confidence {:.3} should be > 0.3",
                result.confidence
            );
        }
    }

    #[test]
    fn wrong_feature_dim_returns_none() {
        let tc = train_transversal();
        assert!(tc.classify(&vec![1.0; FEATURE_DIM - 1]).is_none());
        assert!(tc.classify(&vec![1.0; FEATURE_DIM + 1]).is_none());
        assert!(tc.classify(&Vec::<f32>::new()).is_none());
    }

    #[test]
    fn train_rejects_invalid_phrase_index() {
        let mut tc = TransversalClassifier::new();
        let samples = vec![
            (2u8, Syllable::from_nibble(0), make_features(0)), // phrase 2 is invalid
        ];
        tc.train(&samples);
        assert!(!tc.is_trained(), "should reject phrase index > 1");
    }

    #[test]
    fn train_rejects_wrong_syllable_for_phrase() {
        let mut tc = TransversalClassifier::new();
        // JO (nibble 4) is NOT in phrase 1 (which has 'O, JU, KE, VI)
        let samples = vec![
            (0u8, Syllable::from_nibble(4), make_features(4)),
        ];
        tc.train(&samples);
        assert!(!tc.is_trained(), "should reject syllable not belonging to its phrase");
    }
}
```

Add the module to `stq8-core/src/lib.rs`:

```rust
pub mod transversal;
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core -- transversal`
Expected: FAIL — `TransversalClassifier` doesn't exist yet

**Step 3: Implement TransversalClassifier**

Add above the `#[cfg(test)]` module in `stq8-core/src/transversal.rs`:

```rust
/// Result of transversal classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransversalClassification {
    /// The reconstructed syllable (may not be one of the 8 calibration syllables).
    pub syllable: Syllable,
    /// Margin-based confidence: how cleanly the Top-2 separated from the bottom 2.
    pub confidence: f32,
    /// Diagnostics: Top-2 matches from phrase 1 with their similarities.
    pub phrase1_top2: [(Syllable, f32); 2],
    /// Diagnostics: Top-2 matches from phrase 2 with their similarities.
    pub phrase2_top2: [(Syllable, f32); 2],
}

/// Transversal classifier: 8 centroids across two orthogonal phrases.
pub struct TransversalClassifier {
    /// Centroids grouped by phrase index (0 or 1). Each phrase has exactly 4.
    phrase_centroids: [Vec<(Syllable, Vec<f32>)>; 2],
    trained: bool,
}

impl TransversalClassifier {
    pub fn new() -> Self {
        Self {
            phrase_centroids: [Vec::new(), Vec::new()],
            trained: false,
        }
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Access centroids for a given phrase (0 or 1).
    pub fn phrase_centroids(&self, phrase: u8) -> &[(Syllable, Vec<f32>)] {
        &self.phrase_centroids[phrase as usize]
    }

    /// Train from tagged samples: (phrase_index, syllable, features).
    ///
    /// Validates that:
    /// - phrase_index is 0 or 1
    /// - syllable belongs to the correct phrase
    /// - Each phrase has at least one sample per syllable (4 syllables)
    ///
    /// Multiple samples per syllable are averaged into one centroid.
    pub fn train(&mut self, samples: &[(u8, Syllable, Vec<f32>)]) {
        use std::collections::HashMap;

        let phrases: [&[Syllable; 4]; 2] = [&PHRASE_1, &PHRASE_2];
        let mut groups: [HashMap<Syllable, Vec<&Vec<f32>>>; 2] =
            [HashMap::new(), HashMap::new()];

        for (phrase_idx, syllable, features) in samples {
            if *phrase_idx > 1 {
                self.trained = false;
                return;
            }
            let phrase = phrases[*phrase_idx as usize];
            if !phrase.contains(syllable) {
                self.trained = false;
                return;
            }
            groups[*phrase_idx as usize]
                .entry(*syllable)
                .or_default()
                .push(features);
        }

        // Each phrase must have all 4 syllables represented
        for (i, phrase) in phrases.iter().enumerate() {
            for syllable in phrase.iter() {
                if !groups[i].contains_key(syllable) {
                    self.trained = false;
                    return;
                }
            }
        }

        // Compute centroids (mean of samples per syllable)
        for (i, group) in groups.iter().enumerate() {
            self.phrase_centroids[i].clear();
            for (syllable, vectors) in group {
                let dim = vectors[0].len();
                let n = vectors.len() as f32;
                let mut mean = vec![0.0f32; dim];
                for v in vectors {
                    for (j, &val) in v.iter().enumerate() {
                        mean[j] += val;
                    }
                }
                for v in &mut mean {
                    *v /= n;
                }
                self.phrase_centroids[i].push((*syllable, mean));
            }
        }

        self.trained = true;
    }

    /// Load pre-computed centroids directly (for profile import).
    pub fn load_centroids(&mut self, phrase0: Vec<(Syllable, Vec<f32>)>, phrase1: Vec<(Syllable, Vec<f32>)>) {
        self.phrase_centroids[0] = phrase0;
        self.phrase_centroids[1] = phrase1;
        self.trained = self.phrase_centroids[0].len() == 4 && self.phrase_centroids[1].len() == 4;
    }

    /// Classify a feature vector by Top-2-per-phrase intersection.
    ///
    /// Returns `None` if untrained or feature dimension is wrong.
    pub fn classify(&self, features: &[f32]) -> Option<TransversalClassification> {
        if !self.trained || features.len() != FEATURE_DIM {
            return None;
        }

        let mut phrase_top2 = [[(Syllable::from_nibble(0), 0.0f32); 2]; 2];
        let mut phrase_margins = [0.0f32; 2];

        for (phrase_idx, centroids) in self.phrase_centroids.iter().enumerate() {
            let mut sims: Vec<(Syllable, f32)> = centroids
                .iter()
                .filter(|(_, c)| c.len() == FEATURE_DIM)
                .map(|(s, c)| (*s, cosine_similarity(features, c)))
                .collect();

            if sims.len() < 4 {
                return None;
            }

            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            phrase_top2[phrase_idx] = [(sims[0].0, sims[0].1), (sims[1].0, sims[1].1)];

            // Margin: mean(top2) - mean(bottom2)
            let top2_mean = (sims[0].1 + sims[1].1) / 2.0;
            let bottom2_mean = (sims[2].1 + sims[3].1) / 2.0;
            phrase_margins[phrase_idx] = top2_mean - bottom2_mean;
        }

        // Intersect consonant sets across phrases
        let p1_consonants: Vec<Consonant> = phrase_top2[0].iter().map(|(s, _)| s.consonant).collect();
        let p2_consonants: Vec<Consonant> = phrase_top2[1].iter().map(|(s, _)| s.consonant).collect();
        let consonant = p1_consonants.iter().find(|c| p2_consonants.contains(c))?;

        // Intersect vowel sets across phrases
        let p1_vowels: Vec<Vowel> = phrase_top2[0].iter().map(|(s, _)| s.vowel).collect();
        let p2_vowels: Vec<Vowel> = phrase_top2[1].iter().map(|(s, _)| s.vowel).collect();
        let vowel = p1_vowels.iter().find(|v| p2_vowels.contains(v))?;

        // Confidence: average margin, normalized to [0, 1]
        // Cosine similarity range is [-1, 1], so max margin is 2.0
        // In practice margins are much smaller; clamp to [0, 1]
        let raw_confidence = (phrase_margins[0] + phrase_margins[1]) / 2.0;
        let confidence = raw_confidence.clamp(0.0, 1.0);

        Some(TransversalClassification {
            syllable: Syllable::new(*consonant, *vowel),
            confidence,
            phrase1_top2: phrase_top2[0],
            phrase2_top2: phrase_top2[1],
        })
    }
}

impl Default for TransversalClassifier {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core -- transversal`
Expected: All 8 transversal tests pass

**Step 5: Commit**

```bash
git add stq8-core/src/transversal.rs stq8-core/src/lib.rs
git commit -m "feat(transversal): add TransversalClassifier with Top-2-per-phrase intersection"
```

---

### Task 4: Integrate transversal calibration into Pipeline

**Files:**
- Modify: `stq8-core/src/pipeline.rs`

**Step 1: Write failing tests**

Add to `stq8-core/src/pipeline.rs` in the `tests` module:

```rust
#[test]
fn pipeline_transversal_calibrate() {
    let mut pipeline = Pipeline::new();
    let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
    let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

    for &idx in &phrase1_indices {
        let features = make_features(idx as usize);
        pipeline.add_transversal_sample(0, Syllable::from_nibble(idx), features);
    }
    for &idx in &phrase2_indices {
        let features = make_features(idx as usize);
        pipeline.add_transversal_sample(1, Syllable::from_nibble(idx), features);
    }

    assert!(!pipeline.is_calibrated());
    pipeline.finalize_transversal_calibration();
    assert!(pipeline.is_calibrated());
}

#[test]
fn pipeline_transversal_sets_calibration_mode() {
    use crate::profile::CalibrationMode;

    let mut pipeline = Pipeline::new();
    let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
    let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

    for &idx in &phrase1_indices {
        pipeline.add_transversal_sample(0, Syllable::from_nibble(idx), make_features(idx as usize));
    }
    for &idx in &phrase2_indices {
        pipeline.add_transversal_sample(1, Syllable::from_nibble(idx), make_features(idx as usize));
    }

    pipeline.finalize_transversal_calibration();
    assert_eq!(pipeline.profile.calibration_mode, CalibrationMode::Transversal);
}

#[test]
fn pipeline_transversal_export_import_roundtrip() {
    use crate::profile::CalibrationMode;

    let mut pipeline = Pipeline::new();
    let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
    let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

    for &idx in &phrase1_indices {
        pipeline.add_transversal_sample(0, Syllable::from_nibble(idx), make_features(idx as usize));
    }
    for &idx in &phrase2_indices {
        pipeline.add_transversal_sample(1, Syllable::from_nibble(idx), make_features(idx as usize));
    }

    pipeline.finalize_transversal_calibration();

    let json = pipeline.export_profile_json().expect("export");

    let mut new_pipeline = Pipeline::new();
    new_pipeline.import_profile_json(&json).expect("import");
    assert!(new_pipeline.is_calibrated());
    assert_eq!(new_pipeline.profile.calibration_mode, CalibrationMode::Transversal);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core -- pipeline_transversal`
Expected: FAIL — `add_transversal_sample` and `finalize_transversal_calibration` don't exist

**Step 3: Implement transversal pipeline integration**

Add to `Pipeline` struct:

```rust
use crate::transversal::TransversalClassifier;

pub struct Pipeline {
    classifier: NearestCentroid,
    transversal_classifier: TransversalClassifier,
    profile: UserProfile,
    segmenter_config: SegmenterConfig,
    calibration_samples: Vec<(Syllable, Vec<f32>)>,
    transversal_samples: Vec<(u8, Syllable, Vec<f32>)>,
    frame_processor: FrameProcessor,
    carry_syllable: Option<Syllable>,
}
```

Update `Pipeline::new()`:

```rust
transversal_classifier: TransversalClassifier::new(),
transversal_samples: Vec::new(),
```

Add new methods:

```rust
/// Accumulate a transversal calibration sample tagged with phrase index (0 or 1).
pub fn add_transversal_sample(&mut self, phrase_index: u8, syllable: Syllable, features: Vec<f32>) {
    self.transversal_samples.push((phrase_index, syllable, features));
}

/// Train the transversal classifier from accumulated samples and store
/// centroids in the user profile with `CalibrationMode::Transversal`.
pub fn finalize_transversal_calibration(&mut self) {
    if self.transversal_samples.is_empty() {
        return;
    }
    self.transversal_classifier.train(&self.transversal_samples);
    if self.transversal_classifier.is_trained() {
        // Store centroids in profile tagged by phrase
        self.profile.centroids.clear();
        for phrase_idx in 0..2u8 {
            for (syllable, centroid) in self.transversal_classifier.phrase_centroids(phrase_idx) {
                self.profile.centroids.push((*syllable, centroid.clone()));
            }
        }
        self.profile.calibration_mode = CalibrationMode::Transversal;
    }
    self.transversal_samples.clear();
    self.carry_syllable = None;
}
```

Update `is_calibrated()`:

```rust
pub fn is_calibrated(&self) -> bool {
    self.classifier.is_trained() || self.transversal_classifier.is_trained()
}
```

Update `process()` to dispatch based on calibration mode:

```rust
// Step 3: Classify
let classification = match self.profile.calibration_mode {
    CalibrationMode::Transversal => {
        match self.transversal_classifier.classify(&features) {
            Some(tc) => Classification {
                syllable: tc.syllable,
                confidence: tc.confidence,
            },
            None => continue,
        }
    }
    CalibrationMode::Full => {
        match self.classifier.classify(&features) {
            Some(c) => c,
            None => continue,
        }
    }
};
```

Update `import_profile_json()` to restore the right classifier:

```rust
pub fn import_profile_json(&mut self, json: &str) -> Result<(), serde_json::Error> {
    let profile: UserProfile = serde_json::from_str(json)?;
    if profile.version != 1 {
        return Err(serde::de::Error::custom(format!(
            "unsupported profile version {}",
            profile.version
        )));
    }
    match profile.calibration_mode {
        CalibrationMode::Transversal => {
            // First 4 centroids = phrase 0, last 4 = phrase 1
            let (p0, p1) = if profile.centroids.len() == 8 {
                (profile.centroids[..4].to_vec(), profile.centroids[4..].to_vec())
            } else {
                (Vec::new(), Vec::new())
            };
            self.transversal_classifier.load_centroids(p0, p1);
        }
        CalibrationMode::Full => {
            self.classifier.load_centroids(profile.centroids.clone());
        }
    }
    self.calibration_samples.clear();
    self.transversal_samples.clear();
    self.carry_syllable = None;
    self.profile = profile;
    Ok(())
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core`
Expected: All tests pass, including the 3 new pipeline_transversal tests

**Step 5: Commit**

```bash
git add stq8-core/src/pipeline.rs
git commit -m "feat(pipeline): integrate TransversalClassifier with calibration and inference dispatch"
```

---

### Task 5: Add WASM bindings for transversal calibration

**Files:**
- Modify: `stq8-web/src/lib.rs`

**Step 1: Add new WASM methods**

Add to the `#[wasm_bindgen] impl WasmPipeline` block:

```rust
/// Feed a transversal calibration sample.
/// phrase_index: 0 or 1 (phrase 1 = diagonal, phrase 2 = anti-diagonal).
/// syllable_index: 0-15 (must be a valid syllable for the given phrase).
/// pcm: raw f32 samples at 16kHz.
pub fn add_transversal_sample(
    &mut self,
    phrase_index: u8,
    syllable_index: u8,
    pcm: &[f32],
) -> Result<(), JsError> {
    if phrase_index > 1 {
        return Err(JsError::new(&format!(
            "phrase_index {phrase_index} out of range (0–1)"
        )));
    }
    if syllable_index > 15 {
        return Err(JsError::new(&format!(
            "syllable_index {syllable_index} out of range (0–15)"
        )));
    }
    let syllable = Syllable::from_nibble(syllable_index);
    let features = mfcc::extract_features(pcm);
    self.inner
        .add_transversal_sample(phrase_index, syllable, features);
    Ok(())
}

pub fn finalize_transversal_calibration(&mut self) {
    self.inner.finalize_transversal_calibration();
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p stq8-web`
Expected: Compiles cleanly

**Step 3: Commit**

```bash
git add stq8-web/src/lib.rs
git commit -m "feat(wasm): add transversal calibration bindings"
```

---

### Task 6: Accuracy comparison benchmarks

**Files:**
- Create: `stq8-core/tests/transversal_accuracy.rs`

This is an integration test that trains both classifiers on the same synthetic data and compares accuracy.

**Step 1: Write the benchmark test**

Create `stq8-core/tests/transversal_accuracy.rs`:

```rust
//! Accuracy comparison: 16-sound (NearestCentroid) vs 8-sound (TransversalClassifier).
//!
//! Uses identical synthetic MFCC data to compare:
//! - Accuracy (% correct nibble prediction)
//! - Confidence distributions
//! - Noise robustness

use stq8_core::classifier::{Classifier, NearestCentroid};
use stq8_core::mfcc::FEATURE_DIM;
use stq8_core::q8::Syllable;
use stq8_core::transversal::TransversalClassifier;

/// Deterministic synthetic features: each syllable gets a unique region.
fn make_features(syllable_idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; FEATURE_DIM];
    let base = (syllable_idx * 3) % FEATURE_DIM;
    v[base] = 1.0;
    v[(base + 1) % FEATURE_DIM] = 0.5;
    v[(base + 2) % FEATURE_DIM] = 0.8;
    v
}

/// Add Gaussian-like noise (deterministic pseudo-random based on seed).
fn add_noise(features: &[f32], noise_level: f32, seed: u32) -> Vec<f32> {
    features
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            // Simple deterministic "noise" using hash-like mixing
            let hash = ((seed as u64).wrapping_mul(6364136223846793005) + i as u64) as f32;
            let noise = (hash / u32::MAX as f32) * 2.0 - 1.0; // [-1, 1]
            v + noise * noise_level
        })
        .collect()
}

fn train_full() -> NearestCentroid {
    let mut nc = NearestCentroid::new();
    let samples: Vec<(Syllable, Vec<f32>)> = (0..16)
        .map(|i| (Syllable::from_nibble(i), make_features(i as usize)))
        .collect();
    nc.train(&samples);
    nc
}

fn train_transversal() -> TransversalClassifier {
    let mut tc = TransversalClassifier::new();
    let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
    let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

    let mut samples = Vec::new();
    for &idx in &phrase1_indices {
        samples.push((0u8, Syllable::from_nibble(idx), make_features(idx as usize)));
    }
    for &idx in &phrase2_indices {
        samples.push((1u8, Syllable::from_nibble(idx), make_features(idx as usize)));
    }
    tc.train(&samples);
    tc
}

#[test]
fn clean_accuracy_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    let mut nc_correct = 0u32;
    let mut tc_correct = 0u32;

    for nibble in 0..16u8 {
        let features = make_features(nibble as usize);
        let expected = Syllable::from_nibble(nibble);

        if let Some(c) = nc.classify(&features) {
            if c.syllable == expected {
                nc_correct += 1;
            }
        }
        if let Some(c) = tc.classify(&features) {
            if c.syllable == expected {
                tc_correct += 1;
            }
        }
    }

    println!("=== Clean Accuracy ===");
    println!("Full (16-sound):       {nc_correct}/16 ({:.0}%)", nc_correct as f32 / 16.0 * 100.0);
    println!("Transversal (8-sound): {tc_correct}/16 ({:.0}%)", tc_correct as f32 / 16.0 * 100.0);

    // Both should be 100% on clean data
    assert_eq!(nc_correct, 16, "Full classifier should be 100% on clean data");
    assert_eq!(tc_correct, 16, "Transversal classifier should be 100% on clean data");
}

#[test]
fn noisy_accuracy_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    let noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30];
    let trials_per_level = 50;

    println!("\n=== Noisy Accuracy (avg over {trials_per_level} trials) ===");
    println!("{:>10} {:>12} {:>12}", "Noise", "Full", "Transversal");

    for &noise in &noise_levels {
        let mut nc_correct = 0u32;
        let mut tc_correct = 0u32;
        let total = 16 * trials_per_level;

        for nibble in 0..16u8 {
            let expected = Syllable::from_nibble(nibble);
            for trial in 0..trials_per_level {
                let seed = (nibble as u32) * 1000 + trial;
                let features = add_noise(&make_features(nibble as usize), noise, seed);

                if let Some(c) = nc.classify(&features) {
                    if c.syllable == expected {
                        nc_correct += 1;
                    }
                }
                if let Some(c) = tc.classify(&features) {
                    if c.syllable == expected {
                        tc_correct += 1;
                    }
                }
            }
        }

        let nc_pct = nc_correct as f32 / total as f32 * 100.0;
        let tc_pct = tc_correct as f32 / total as f32 * 100.0;
        println!("{noise:>10.2} {nc_pct:>11.1}% {tc_pct:>11.1}%");
    }
}

#[test]
fn confidence_distribution_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    println!("\n=== Confidence Distribution (clean data) ===");
    println!("{:>7} {:>12} {:>12}", "Nibble", "Full conf", "Trans conf");

    let mut nc_total = 0.0f32;
    let mut tc_total = 0.0f32;

    for nibble in 0..16u8 {
        let features = make_features(nibble as usize);

        let nc_conf = nc.classify(&features).map(|c| c.confidence).unwrap_or(0.0);
        let tc_conf = tc.classify(&features).map(|c| c.confidence).unwrap_or(0.0);

        nc_total += nc_conf;
        tc_total += tc_conf;

        println!("{nibble:>7} {nc_conf:>11.3} {tc_conf:>11.3}");
    }

    println!("{:>7} {:.3} {:.3}", "Mean", nc_total / 16.0, tc_total / 16.0);
}

#[test]
fn inference_cost_comparison() {
    // Just verify the computational difference is as expected:
    // Full: 16 cosine similarity ops per classify
    // Transversal: 8 cosine similarity ops + intersection logic

    let nc = train_full();
    let tc = train_transversal();

    // Measure by counting successful classifications
    let features = make_features(7); // arbitrary nibble

    let nc_result = nc.classify(&features);
    let tc_result = tc.classify(&features);

    assert!(nc_result.is_some(), "Full classifier should classify");
    assert!(tc_result.is_some(), "Transversal classifier should classify");

    // Both should get the right answer
    assert_eq!(nc_result.unwrap().syllable.to_nibble(), 7);
    assert_eq!(tc_result.unwrap().syllable.to_nibble(), 7);

    println!("\n=== Inference Cost ===");
    println!("Full: 16 centroid comparisons");
    println!("Transversal: 8 centroid comparisons + set intersection");
    println!("Expected speedup: ~2x on similarity computation");
}
```

**Step 2: Run the benchmarks**

Run: `cargo test -p stq8-core --test transversal_accuracy -- --nocapture`
Expected: All 4 tests pass, with printed comparison tables

**Step 3: Commit**

```bash
git add stq8-core/tests/transversal_accuracy.rs
git commit -m "test(transversal): add accuracy comparison benchmarks vs 16-sound classifier"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Make `cosine_similarity` pub(crate) | `classifier.rs` |
| 2 | Add `CalibrationMode` to profile | `profile.rs`, `pipeline.rs` |
| 3 | TransversalClassifier with reconstruction | `transversal.rs`, `lib.rs` |
| 4 | Pipeline integration (calibrate + inference dispatch) | `pipeline.rs` |
| 5 | WASM bindings | `stq8-web/src/lib.rs` |
| 6 | Accuracy comparison benchmarks | `tests/transversal_accuracy.rs` |

Total: ~6 commits, TDD throughout, frequent validation.
