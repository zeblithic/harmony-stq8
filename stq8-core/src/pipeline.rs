//! Pipeline integration: wires segmenter + MFCC + classifier + profile into a
//! single processing pipeline.
//!
//! The [`Pipeline`] is the main API surface for STQ8. It accepts raw PCM audio
//! from a PTT utterance and returns an [`UtteranceResult`] containing per-syllable
//! classification decisions and Q8-encoded bytes for auto-accepted syllables.

use crate::classifier::{Classification, Classifier, NearestCentroid};
use crate::mfcc::{self, FrameProcessor};
use crate::profile::{CalibrationMode, Decision, Thresholds, UserProfile};
use crate::q8::Syllable;
use crate::segmenter::{self, SegmenterConfig};
use crate::transversal::{TrainError, TransversalClassifier};
use serde::{Deserialize, Serialize};

/// Result of processing a single syllable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableResult {
    pub decision: Decision,
    pub raw_classification: Classification,
}

/// Result of processing a full PTT utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceResult {
    /// Per-syllable classification decisions.
    pub syllables: Vec<SyllableResult>,
    /// Q8 bytes emitted only from auto-accepted syllables.
    pub q8_bytes: Vec<u8>,
    /// An accepted syllable that couldn't be paired into a complete byte.
    /// The UI should prompt the user to speak one more syllable.
    pub pending_syllable: Option<Syllable>,
    /// The carry syllable from the *previous* call that was consumed and
    /// paired with the first accepted syllable of this call. `None` if
    /// no carry was consumed.
    pub consumed_carry: Option<Syllable>,
}

/// Main processing pipeline: segment -> MFCC -> classify -> decide -> Q8 encode.
pub struct Pipeline {
    classifier: NearestCentroid,
    transversal_classifier: TransversalClassifier,
    profile: UserProfile,
    segmenter_config: SegmenterConfig,
    calibration_samples: Vec<(Syllable, Vec<f32>)>,
    transversal_samples: Vec<(u8, Syllable, Vec<f32>)>,
    frame_processor: FrameProcessor,
    /// Syllable carried over from a previous `process()` call when an odd
    /// number of syllables were accepted. Prepended to the next call's
    /// accepted syllables so the pair completes a byte.
    carry_syllable: Option<Syllable>,
}

impl Pipeline {
    /// Create a new uncalibrated pipeline with default config and profile.
    pub fn new() -> Self {
        Self {
            classifier: NearestCentroid::new(),
            transversal_classifier: TransversalClassifier::new(),
            profile: UserProfile {
                version: 1,
                centroids: Vec::new(),
                thresholds: Thresholds::default(),
                custom_map: Vec::new(),
                calibration_mode: CalibrationMode::default(),
                created_epoch_secs: 0,
            },
            segmenter_config: SegmenterConfig::default(),
            calibration_samples: Vec::new(),
            transversal_samples: Vec::new(),
            frame_processor: FrameProcessor::new(),
            carry_syllable: None,
        }
    }

    /// Accumulate a single calibration sample for batch training.
    pub fn add_calibration_sample(&mut self, syllable: Syllable, features: Vec<f32>) {
        self.calibration_samples.push((syllable, features));
    }

    /// Train the classifier from all accumulated calibration samples and
    /// store the resulting centroids in the user profile.
    ///
    /// No-op if no samples have been added since the last calibration or import.
    pub fn finalize_calibration(&mut self) {
        if self.calibration_samples.is_empty() {
            return;
        }
        self.classifier.train(&self.calibration_samples);
        self.profile.centroids = self.classifier.centroids().to_vec();
        self.profile.calibration_mode = CalibrationMode::Full;
        // Reset stale transversal classifier so is_calibrated() and process()
        // reflect only the active mode.
        self.transversal_classifier = TransversalClassifier::new();
        self.calibration_samples.clear();
        self.carry_syllable = None;
    }

    /// Accumulate a transversal calibration sample tagged with phrase index (0 or 1).
    pub fn add_transversal_sample(
        &mut self,
        phrase_index: u8,
        syllable: Syllable,
        features: Vec<f32>,
    ) {
        self.transversal_samples
            .push((phrase_index, syllable, features));
    }

    /// Train the transversal classifier from accumulated samples and store
    /// centroids in the user profile with CalibrationMode::Transversal.
    ///
    /// Returns an error if any sample has an invalid phrase index, a syllable
    /// that doesn't belong to its phrase, or if a phrase is missing samples.
    pub fn finalize_transversal_calibration(&mut self) -> Result<(), TrainError> {
        if self.transversal_samples.is_empty() {
            return Err(TrainError::NoSamples);
        }
        self.transversal_classifier
            .train(&self.transversal_samples)?;
        // Store centroids in profile tagged by phrase
        self.profile.centroids.clear();
        for phrase_idx in 0..2u8 {
            if let Some(centroids) = self.transversal_classifier.phrase_centroids(phrase_idx) {
                for (syllable, centroid) in centroids {
                    self.profile.centroids.push((*syllable, centroid.clone()));
                }
            }
        }
        self.profile.calibration_mode = CalibrationMode::Transversal;
        // Reset stale full classifier so is_calibrated() and process()
        // reflect only the active mode.
        self.classifier = NearestCentroid::new();
        self.transversal_samples.clear();
        self.carry_syllable = None;
        Ok(())
    }

    /// Convenience: add all samples and finalize calibration in one call.
    pub fn calibrate(&mut self, samples: &[(Syllable, Vec<f32>)]) {
        for (syllable, features) in samples {
            self.calibration_samples.push((*syllable, features.clone()));
        }
        self.finalize_calibration();
    }

    /// Returns true if either classifier has been trained.
    pub fn is_calibrated(&self) -> bool {
        self.classifier.is_trained() || self.transversal_classifier.is_trained()
    }

    /// Process a PTT utterance and return per-syllable decisions plus Q8 bytes.
    ///
    /// Steps:
    /// 1. Segment the PCM into syllable bounds
    /// 2. Extract MFCC features for each syllable
    /// 3. Classify features into one of 16 syllables
    /// 4. Apply profile remap (phoneme-level component remapping)
    /// 5. Apply threshold decision
    /// 6. Encode auto-accepted syllables into Q8 bytes (2 syllables = 1 byte)
    pub fn process(&mut self, pcm: &[f32]) -> UtteranceResult {
        if !self.is_calibrated() {
            return UtteranceResult {
                syllables: Vec::new(),
                q8_bytes: Vec::new(),
                pending_syllable: None,
                consumed_carry: None,
            };
        }

        // Step 1: Segment
        let bounds = segmenter::segment(pcm, &self.segmenter_config);

        let mut syllable_results = Vec::new();
        let mut accepted_syllables: Vec<Syllable> = Vec::new();

        // Prepend carried-over syllable from the previous call
        let consumed_carry = self.carry_syllable.take();
        let carry_prepended = consumed_carry.is_some();
        if let Some(carry) = consumed_carry {
            accepted_syllables.push(carry);
        }

        for bound in &bounds {
            let segment_pcm = &pcm[bound.start..bound.end];

            // Step 2: Extract features (reuse cached FrameProcessor)
            let features = mfcc::extract_features_with(segment_pcm, &self.frame_processor);

            // Step 3: Classify — dispatch based on calibration mode
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
                CalibrationMode::Full => match self.classifier.classify(&features) {
                    Some(c) => c,
                    None => continue,
                },
            };

            // Step 4: Apply profile remap
            let remapped = self.profile.apply_remap(classification.syllable);

            // Step 5: Apply threshold decision
            let decision = self
                .profile
                .thresholds
                .decide(remapped, classification.confidence);

            // Collect accepted syllables for Q8 encoding
            if let Decision::Accept(syllable, _) = &decision {
                accepted_syllables.push(*syllable);
            }

            syllable_results.push(SyllableResult {
                decision,
                raw_classification: classification,
            });
        }

        // Step 6: Encode accepted syllables into Q8 bytes.
        // Each syllable is one nibble. Two syllables = one byte.
        let q8_bytes = encode_syllables_to_q8(&accepted_syllables);

        // Count newly accepted syllables (excluding carried-over prefix)
        let new_accepted = accepted_syllables.len() - usize::from(carry_prepended);

        let pending_syllable = if accepted_syllables.len() % 2 == 1 {
            let trailing = *accepted_syllables.last().unwrap();
            self.carry_syllable = Some(trailing);
            // Only surface as pending if new syllables arrived this call.
            // If the carry just passed through unchanged, don't re-report it —
            // the caller already knows about it from the previous call.
            if new_accepted > 0 {
                Some(trailing)
            } else {
                None
            }
        } else {
            None
        };

        // Report consumed_carry only if it contributed to a completed byte.
        // If the carry just became the new pending syllable, it wasn't "consumed."
        let consumed_carry = consumed_carry.filter(|_| !q8_bytes.is_empty());

        UtteranceResult {
            syllables: syllable_results,
            q8_bytes,
            pending_syllable,
            consumed_carry,
        }
    }

    /// Set the profile's creation timestamp. The caller (e.g., JS via WASM)
    /// is responsible for supplying the current time since `Pipeline` has no
    /// access to system clocks.
    pub fn set_created_epoch_secs(&mut self, secs: u64) {
        self.profile.created_epoch_secs = secs;
    }

    /// Export the user profile as JSON.
    pub fn export_profile_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.profile)
    }

    /// Import a user profile from JSON, replacing the current profile and
    /// re-training the classifier from the imported centroids.
    ///
    /// Rejects profiles with an unsupported version number.
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
                if profile.centroids.len() != 8 {
                    return Err(serde::de::Error::custom(format!(
                        "transversal profile requires exactly 8 centroids, got {}",
                        profile.centroids.len()
                    )));
                }
                let (p0, p1) = (
                    profile.centroids[..4].to_vec(),
                    profile.centroids[4..].to_vec(),
                );
                self.transversal_classifier
                    .load_centroids(p0, p1)
                    .map_err(serde::de::Error::custom)?;
                // Reset stale Full classifier so is_calibrated() reflects only the active mode
                self.classifier = NearestCentroid::new();
            }
            CalibrationMode::Full => {
                self.classifier.load_centroids(profile.centroids.clone());
                // Reset stale Transversal classifier so is_calibrated() reflects only the active mode
                self.transversal_classifier = TransversalClassifier::new();
            }
        }
        self.calibration_samples.clear();
        self.transversal_samples.clear();
        self.carry_syllable = None;
        self.profile = profile;
        Ok(())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a sequence of accepted syllables into Q8 bytes.
///
/// Each syllable is a nibble (4 bits). Two syllables = one byte
/// (first = high nibble, second = low nibble).
/// Trailing syllables that don't complete a byte are discarded.
fn encode_syllables_to_q8(syllables: &[Syllable]) -> Vec<u8> {
    syllables
        .chunks_exact(2)
        .map(|pair| (pair[0].to_nibble() << 4) | pair[1].to_nibble())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mfcc::{FEATURE_DIM, SAMPLE_RATE};
    use crate::q8::{Consonant, Vowel};
    use std::f32::consts::PI;

    /// Generate a sine wave at a given frequency.
    fn sine_wave(freq: f32, duration_samples: usize) -> Vec<f32> {
        (0..duration_samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    /// Build synthetic MFCC features for a syllable using a deterministic pattern.
    fn make_features(syllable_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; FEATURE_DIM];
        let base = (syllable_idx * 3) % FEATURE_DIM;
        v[base] = 1.0;
        v[(base + 1) % FEATURE_DIM] = 0.5;
        v[(base + 2) % FEATURE_DIM] = 0.8;
        v
    }

    fn all_syllables() -> Vec<Syllable> {
        (0..16).map(Syllable::from_nibble).collect()
    }

    #[test]
    fn pipeline_uncalibrated_returns_empty() {
        let mut pipeline = Pipeline::new();
        assert!(!pipeline.is_calibrated());

        let pcm = sine_wave(440.0, SAMPLE_RATE as usize);
        let result = pipeline.process(&pcm);

        assert!(result.syllables.is_empty());
        assert!(result.q8_bytes.is_empty());
    }

    #[test]
    fn pipeline_calibrate_and_process() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        let mut samples: Vec<(Syllable, Vec<f32>)> = Vec::new();
        for (idx, &syllable) in syllables.iter().enumerate() {
            for rep in 0..5 {
                let mut features = make_features(idx);
                for (i, v) in features.iter_mut().enumerate() {
                    *v += (rep as f32) * 0.01 * ((i % 3) as f32 - 1.0);
                }
                samples.push((syllable, features));
            }
        }

        assert!(!pipeline.is_calibrated());
        pipeline.calibrate(&samples);
        assert!(pipeline.is_calibrated());
        assert!(!pipeline.profile.centroids.is_empty());
    }

    #[test]
    fn pipeline_add_calibration_sample_then_finalize() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        for (idx, &syllable) in syllables.iter().enumerate() {
            for rep in 0..3 {
                let mut features = make_features(idx);
                for (i, v) in features.iter_mut().enumerate() {
                    *v += (rep as f32) * 0.005 * ((i % 3) as f32 - 1.0);
                }
                pipeline.add_calibration_sample(syllable, features);
            }
        }

        assert!(!pipeline.is_calibrated());
        pipeline.finalize_calibration();
        assert!(pipeline.is_calibrated());
    }

    #[test]
    fn pipeline_export_import_profile() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();

        pipeline.calibrate(&samples);

        let json = pipeline
            .export_profile_json()
            .expect("export should succeed");
        assert!(!json.is_empty());
        assert!(json.contains("centroids"));

        let mut new_pipeline = Pipeline::new();
        new_pipeline
            .import_profile_json(&json)
            .expect("import should succeed");
        assert!(new_pipeline.is_calibrated());
        assert_eq!(
            new_pipeline.profile.centroids.len(),
            pipeline.profile.centroids.len()
        );
    }

    #[test]
    fn import_clears_stale_calibration_samples() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        // Add calibration samples but don't finalize
        for (idx, &syllable) in syllables.iter().enumerate() {
            pipeline.add_calibration_sample(syllable, make_features(idx));
        }

        // Calibrate and export a profile from a separate pipeline
        let mut source = Pipeline::new();
        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();
        source.calibrate(&samples);
        let json = source.export_profile_json().expect("export");

        // Import into the pipeline that has stale samples
        pipeline.import_profile_json(&json).expect("import");
        assert!(pipeline.is_calibrated());

        // finalize_calibration should be a no-op (no stale samples to re-train on)
        // — it should NOT overwrite the imported profile
        pipeline.finalize_calibration();
        assert!(
            pipeline.is_calibrated(),
            "finalize_calibration with no pending samples should not destroy imported state"
        );
        assert_eq!(
            pipeline.profile.centroids.len(),
            syllables.len(),
            "imported centroids should survive a no-op finalize_calibration"
        );
    }

    #[test]
    fn pipeline_process_with_real_audio() {
        let mut pipeline = Pipeline::new();

        let syllables = all_syllables();
        let freqs = [
            200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1600.0, 2000.0, 2400.0, 2800.0, 3200.0,
            3600.0, 4000.0, 4800.0, 5600.0, 6400.0,
        ];

        let mut samples = Vec::new();
        for (idx, &syllable) in syllables.iter().enumerate() {
            let wave = sine_wave(freqs[idx], SAMPLE_RATE as usize / 2);
            let features = mfcc::extract_features(&wave);
            samples.push((syllable, features));
        }

        pipeline.calibrate(&samples);
        assert!(pipeline.is_calibrated());

        // Create test audio: silence + a burst + silence
        let sr = SAMPLE_RATE as usize;
        let mut pcm = vec![0.0f32; sr];
        let burst_start = sr / 5;
        let burst_end = burst_start + sr * 2 / 5;
        for i in burst_start..burst_end.min(sr) {
            let t = i as f32 / SAMPLE_RATE as f32;
            pcm[i] = (2.0 * PI * freqs[0] * t).sin() * 0.8;
        }

        let result = pipeline.process(&pcm);
        assert!(
            !result.syllables.is_empty(),
            "pipeline with calibration data should detect at least one syllable in a burst"
        );
    }

    #[test]
    fn encode_syllables_to_q8_basic() {
        // KU + 'E = byte 0x92
        let syllables = vec![
            Syllable::new(Consonant::K, Vowel::U),           // nibble 9
            Syllable::new(Consonant::GlottalStop, Vowel::E), // nibble 2
        ];

        let bytes = encode_syllables_to_q8(&syllables);
        assert_eq!(bytes, vec![0x92]);
    }

    #[test]
    fn encode_syllables_to_q8_trailing_discarded() {
        // 1 syllable can't form a complete byte
        let syllables = vec![Syllable::new(Consonant::J, Vowel::O)];
        let bytes = encode_syllables_to_q8(&syllables);
        assert!(bytes.is_empty(), "1 syllable should produce 0 bytes");
    }

    #[test]
    fn encode_syllables_to_q8_empty() {
        assert!(encode_syllables_to_q8(&[]).is_empty());
    }

    #[test]
    fn encode_syllables_roundtrip_with_q8() {
        // Encode via syllables, verify it matches q8::byte_to_word
        use crate::q8;

        let s1 = Syllable::new(Consonant::V, Vowel::I); // nibble 15 = 0xF
        let s2 = Syllable::new(Consonant::V, Vowel::I); // nibble 15 = 0xF
        let bytes = encode_syllables_to_q8(&[s1, s2]);
        assert_eq!(bytes, vec![0xFF]);
        assert_eq!(q8::byte_to_word(0xFF), "VIVI");
    }

    #[test]
    fn pipeline_default_trait() {
        let pipeline = Pipeline::default();
        assert!(!pipeline.is_calibrated());
    }

    #[test]
    fn syllable_result_serialization() {
        let syllable = Syllable::new(Consonant::K, Vowel::O);
        let result = SyllableResult {
            decision: Decision::Accept(syllable, 0.95),
            raw_classification: Classification {
                syllable,
                confidence: 0.95,
            },
        };

        let json = serde_json::to_string(&result).expect("serialize SyllableResult");
        let restored: SyllableResult =
            serde_json::from_str(&json).expect("deserialize SyllableResult");

        assert_eq!(
            restored.raw_classification.syllable,
            result.raw_classification.syllable
        );
    }

    #[test]
    fn utterance_result_serialization() {
        let syllable = Syllable::new(Consonant::V, Vowel::E);
        let carry = Syllable::new(Consonant::J, Vowel::U);
        let result = UtteranceResult {
            syllables: vec![SyllableResult {
                decision: Decision::Reject(0.2),
                raw_classification: Classification {
                    syllable,
                    confidence: 0.2,
                },
            }],
            q8_bytes: vec![0x92, 0xFF],
            pending_syllable: Some(Syllable::new(Consonant::K, Vowel::O)),
            consumed_carry: Some(carry),
        };

        let json = serde_json::to_string(&result).expect("serialize UtteranceResult");
        let restored: UtteranceResult =
            serde_json::from_str(&json).expect("deserialize UtteranceResult");

        assert_eq!(restored.syllables.len(), 1);
        assert_eq!(restored.q8_bytes, vec![0x92, 0xFF]);
        assert_eq!(
            restored.pending_syllable,
            Some(Syllable::new(Consonant::K, Vowel::O))
        );
        assert_eq!(restored.consumed_carry, Some(carry));
    }

    #[test]
    fn encode_odd_syllables_reports_pending() {
        let syllables = vec![
            Syllable::new(Consonant::K, Vowel::U),           // nibble 9
            Syllable::new(Consonant::GlottalStop, Vowel::E), // nibble 2
            Syllable::new(Consonant::V, Vowel::I),           // nibble 15 — trailing
        ];

        let bytes = encode_syllables_to_q8(&syllables);
        assert_eq!(bytes, vec![0x92], "only complete pairs should encode");

        let pending = if syllables.len() % 2 == 1 {
            syllables.last().copied()
        } else {
            None
        };
        assert_eq!(
            pending,
            Some(Syllable::new(Consonant::V, Vowel::I)),
            "trailing syllable should be reported as pending"
        );
    }

    #[test]
    fn import_rejects_unsupported_version() {
        let mut pipeline = Pipeline::new();
        let json = r#"{"version":99,"centroids":[],"thresholds":{"auto_accept":0.85,"reject":0.40},"custom_map":[],"created_epoch_secs":0}"#;
        let result = pipeline.import_profile_json(json);
        assert!(result.is_err(), "version 99 should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unsupported profile version"),
            "error should mention version: {err}"
        );
    }

    #[test]
    fn recalibration_clears_carry_syllable() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();

        pipeline.calibrate(&samples);

        // Manually set a carry syllable (simulating a previous process() with odd accepted)
        pipeline.carry_syllable = Some(Syllable::from_nibble(5));

        // Recalibrate should clear carry
        pipeline.calibrate(&samples);
        assert!(
            pipeline.carry_syllable.is_none(),
            "recalibration should clear stale carry syllable"
        );
    }

    #[test]
    fn carry_passthrough_does_not_rereport_pending() {
        let mut pipeline = Pipeline::new();

        // Calibrate with minimal data so process() runs
        let syllables = all_syllables();
        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();
        pipeline.calibrate(&samples);

        // Set carry AFTER calibration (calibration clears carry)
        let carry = Syllable::from_nibble(5);
        pipeline.carry_syllable = Some(carry);

        // Process silence — no new syllables accepted
        let silence = vec![0.0f32; SAMPLE_RATE as usize];
        let result = pipeline.process(&silence);

        // The carry should still be held internally
        assert!(
            pipeline.carry_syllable.is_some(),
            "carry should persist internally when nothing new is accepted"
        );

        // But pending_syllable should NOT re-report the unchanged carry
        assert!(
            result.pending_syllable.is_none(),
            "pending_syllable should be None when carry just passes through unchanged"
        );

        // consumed_carry should also be None (carry wasn't consumed into a byte)
        assert!(
            result.consumed_carry.is_none(),
            "consumed_carry should be None when no bytes were produced"
        );
    }

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
        pipeline
            .finalize_transversal_calibration()
            .expect("training should succeed");
        assert!(pipeline.is_calibrated());
    }

    #[test]
    fn pipeline_transversal_sets_calibration_mode() {
        use crate::profile::CalibrationMode;

        let mut pipeline = Pipeline::new();
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

        for &idx in &phrase1_indices {
            pipeline.add_transversal_sample(
                0,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        for &idx in &phrase2_indices {
            pipeline.add_transversal_sample(
                1,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }

        pipeline
            .finalize_transversal_calibration()
            .expect("training should succeed");
        assert_eq!(
            pipeline.profile.calibration_mode,
            CalibrationMode::Transversal
        );
    }

    #[test]
    fn pipeline_transversal_export_import_roundtrip() {
        use crate::profile::CalibrationMode;

        let mut pipeline = Pipeline::new();
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

        for &idx in &phrase1_indices {
            pipeline.add_transversal_sample(
                0,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        for &idx in &phrase2_indices {
            pipeline.add_transversal_sample(
                1,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }

        pipeline
            .finalize_transversal_calibration()
            .expect("training should succeed");

        let json = pipeline.export_profile_json().expect("export");

        let mut new_pipeline = Pipeline::new();
        new_pipeline.import_profile_json(&json).expect("import");
        assert!(new_pipeline.is_calibrated());
        assert_eq!(
            new_pipeline.profile.calibration_mode,
            CalibrationMode::Transversal
        );
    }

    #[test]
    fn set_created_epoch_secs() {
        let mut pipeline = Pipeline::new();
        pipeline.set_created_epoch_secs(1_710_000_000);

        let json = pipeline.export_profile_json().expect("export");
        assert!(
            json.contains("1710000000"),
            "exported profile should contain the timestamp"
        );
    }

    #[test]
    fn import_resets_stale_full_classifier() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        // Step 1: Calibrate with Full mode — classifier is now trained
        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();
        pipeline.calibrate(&samples);
        assert!(pipeline.classifier.is_trained());

        // Step 2: Import a Transversal profile — should reset the Full classifier
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];
        let mut transversal_pipeline = Pipeline::new();
        for &idx in &phrase1_indices {
            transversal_pipeline.add_transversal_sample(
                0,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        for &idx in &phrase2_indices {
            transversal_pipeline.add_transversal_sample(
                1,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        transversal_pipeline
            .finalize_transversal_calibration()
            .expect("training should succeed");
        let json = transversal_pipeline.export_profile_json().expect("export");

        pipeline.import_profile_json(&json).expect("import");
        // The stale Full classifier must be reset
        assert!(
            !pipeline.classifier.is_trained(),
            "importing Transversal profile should reset stale Full classifier"
        );
        assert!(pipeline.transversal_classifier.is_trained());
    }

    #[test]
    fn import_resets_stale_transversal_classifier() {
        let mut pipeline = Pipeline::new();
        let syllables = all_syllables();

        // Step 1: Calibrate with Transversal mode
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];
        for &idx in &phrase1_indices {
            pipeline.add_transversal_sample(
                0,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        for &idx in &phrase2_indices {
            pipeline.add_transversal_sample(
                1,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        pipeline
            .finalize_transversal_calibration()
            .expect("training should succeed");
        assert!(pipeline.transversal_classifier.is_trained());

        // Step 2: Import a Full profile — should reset the Transversal classifier
        let mut full_pipeline = Pipeline::new();
        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();
        full_pipeline.calibrate(&samples);
        let json = full_pipeline.export_profile_json().expect("export");

        pipeline.import_profile_json(&json).expect("import");
        assert!(
            !pipeline.transversal_classifier.is_trained(),
            "importing Full profile should reset stale Transversal classifier"
        );
        assert!(pipeline.classifier.is_trained());
    }

    #[test]
    fn import_rejects_transversal_with_wrong_centroid_count() {
        let mut pipeline = Pipeline::new();
        // Manually construct a profile JSON with Transversal mode but wrong centroid count
        let json = r#"{"version":1,"centroids":[[{"consonant":"GlottalStop","vowel":"O"},[1.0]]],"thresholds":{"auto_accept":0.85,"reject":0.40},"custom_map":[],"calibration_mode":"transversal","created_epoch_secs":0}"#;
        let result = pipeline.import_profile_json(json);
        assert!(result.is_err(), "should reject Transversal profile with != 8 centroids");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("8 centroids"),
            "error should mention centroid count: {err}"
        );
    }

    #[test]
    fn finalize_transversal_propagates_wrong_syllable_error() {
        let mut pipeline = Pipeline::new();
        // Add a syllable that doesn't belong to phrase 0
        // PHRASE_2[0] = 'I (nibble 3), which is not in PHRASE_1
        pipeline.add_transversal_sample(
            0,
            Syllable::from_nibble(3), // 'I — belongs to phrase 2, not phrase 1
            make_features(3),
        );
        let result = pipeline.finalize_transversal_calibration();
        assert!(result.is_err(), "should propagate WrongSyllableForPhrase error");
    }

    #[test]
    fn finalize_calibration_sets_full_mode_and_resets_transversal() {
        use crate::profile::CalibrationMode;

        let mut pipeline = Pipeline::new();
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

        // First: transversal calibration
        for &idx in &phrase1_indices {
            pipeline.add_transversal_sample(
                0,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        for &idx in &phrase2_indices {
            pipeline.add_transversal_sample(
                1,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        pipeline
            .finalize_transversal_calibration()
            .expect("transversal training should succeed");
        assert_eq!(pipeline.profile.calibration_mode, CalibrationMode::Transversal);
        assert!(pipeline.transversal_classifier.is_trained());

        // Now: full 16-sound calibration over the top
        let syllables = all_syllables();
        for (idx, &syllable) in syllables.iter().enumerate() {
            pipeline.add_calibration_sample(syllable, make_features(idx));
        }
        pipeline.finalize_calibration();

        // Mode must switch to Full
        assert_eq!(pipeline.profile.calibration_mode, CalibrationMode::Full);
        assert!(pipeline.classifier.is_trained());
        // Stale transversal classifier must be reset
        assert!(
            !pipeline.transversal_classifier.is_trained(),
            "finalize_calibration should reset stale transversal classifier"
        );
    }

    #[test]
    fn finalize_transversal_errors_on_empty_samples() {
        let mut pipeline = Pipeline::new();
        let result = pipeline.finalize_transversal_calibration();
        assert!(result.is_err(), "empty samples should return error");
        let err = result.unwrap_err();
        assert_eq!(
            err,
            crate::transversal::TrainError::NoSamples,
            "error should be NoSamples variant"
        );
    }

    #[test]
    fn import_rejects_transversal_with_wrong_syllable_membership() {
        // Build a valid transversal profile, then mangle the syllable labels
        let mut pipeline = Pipeline::new();
        let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
        let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

        for &idx in &phrase1_indices {
            pipeline.add_transversal_sample(
                0,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        for &idx in &phrase2_indices {
            pipeline.add_transversal_sample(
                1,
                Syllable::from_nibble(idx),
                make_features(idx as usize),
            );
        }
        pipeline
            .finalize_transversal_calibration()
            .expect("training should succeed");

        // Export, parse, swap phrase 0/1 centroids, re-serialize
        let json = pipeline.export_profile_json().expect("export");
        let mut profile: serde_json::Value =
            serde_json::from_str(&json).expect("parse");
        let centroids = profile["centroids"].as_array_mut().unwrap();
        // Swap first 4 and last 4 centroids (phrase 0 ↔ phrase 1)
        let first_half: Vec<_> = centroids[..4].to_vec();
        let second_half: Vec<_> = centroids[4..].to_vec();
        centroids[..4].clone_from_slice(&second_half);
        centroids[4..].clone_from_slice(&first_half);
        let mangled_json = serde_json::to_string(&profile).unwrap();

        let mut new_pipeline = Pipeline::new();
        let result = new_pipeline.import_profile_json(&mangled_json);
        assert!(
            result.is_err(),
            "importing profile with swapped phrase centroids should fail"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("does not belong to phrase"),
            "error should mention phrase membership: {err}"
        );
    }
}
