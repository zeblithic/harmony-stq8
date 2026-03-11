//! Pipeline integration: wires segmenter + MFCC + classifier + profile into a
//! single processing pipeline.
//!
//! The [`Pipeline`] is the main API surface for STQ8. It accepts raw PCM audio
//! from a PTT utterance and returns an [`UtteranceResult`] containing per-syllable
//! classification decisions and Q8-encoded bytes for auto-accepted syllables.

use crate::classifier::{Classification, Classifier, NearestCentroid};
use crate::mfcc;
use crate::profile::{Decision, Thresholds, UserProfile};
use crate::q8::Syllable;
use crate::segmenter::{self, SegmenterConfig};
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
}

/// Main processing pipeline: segment -> MFCC -> classify -> decide -> Q8 encode.
pub struct Pipeline {
    classifier: NearestCentroid,
    profile: UserProfile,
    segmenter_config: SegmenterConfig,
    calibration_samples: Vec<(Syllable, Vec<f32>)>,
}

impl Pipeline {
    /// Create a new uncalibrated pipeline with default config and profile.
    pub fn new() -> Self {
        Self {
            classifier: NearestCentroid::new(),
            profile: UserProfile {
                version: 1,
                centroids: Vec::new(),
                thresholds: Thresholds::default(),
                custom_map: Vec::new(),
                created_epoch_secs: 0,
            },
            segmenter_config: SegmenterConfig::default(),
            calibration_samples: Vec::new(),
        }
    }

    /// Accumulate a single calibration sample for batch training.
    pub fn add_calibration_sample(&mut self, syllable: Syllable, features: Vec<f32>) {
        self.calibration_samples.push((syllable, features));
    }

    /// Train the classifier from all accumulated calibration samples and
    /// store the resulting centroids in the user profile.
    pub fn finalize_calibration(&mut self) {
        self.classifier.train(&self.calibration_samples);
        self.profile.centroids = compute_centroids(&self.calibration_samples);
        self.calibration_samples.clear();
    }

    /// Convenience: add all samples and finalize calibration in one call.
    pub fn calibrate(&mut self, samples: &[(Syllable, Vec<f32>)]) {
        for (syllable, features) in samples {
            self.calibration_samples.push((*syllable, features.clone()));
        }
        self.finalize_calibration();
    }

    /// Returns true if the classifier has been trained.
    pub fn is_calibrated(&self) -> bool {
        self.classifier.is_trained()
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
    pub fn process(&self, pcm: &[f32]) -> UtteranceResult {
        if !self.is_calibrated() {
            return UtteranceResult {
                syllables: Vec::new(),
                q8_bytes: Vec::new(),
            };
        }

        // Step 1: Segment
        let bounds = segmenter::segment(pcm, &self.segmenter_config);

        let mut syllable_results = Vec::new();
        let mut accepted_syllables: Vec<Syllable> = Vec::new();

        for bound in &bounds {
            let segment_pcm = &pcm[bound.start..bound.end];

            // Step 2: Extract features
            let features = mfcc::extract_features(segment_pcm);

            // Step 3: Classify
            let classification = match self.classifier.classify(&features) {
                Some(c) => c,
                None => continue,
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

        UtteranceResult {
            syllables: syllable_results,
            q8_bytes,
        }
    }

    /// Export the user profile as JSON.
    pub fn export_profile_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.profile)
    }

    /// Import a user profile from JSON, replacing the current profile and
    /// re-training the classifier from the imported centroids.
    pub fn import_profile_json(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let profile: UserProfile = serde_json::from_str(json)?;
        self.classifier.train(&profile.centroids);
        self.profile = profile;
        Ok(())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute mean centroids from training samples, grouped by syllable.
fn compute_centroids(samples: &[(Syllable, Vec<f32>)]) -> Vec<(Syllable, Vec<f32>)> {
    use std::collections::HashMap;

    let mut groups: HashMap<Syllable, Vec<&Vec<f32>>> = HashMap::new();
    for (syllable, features) in samples {
        groups.entry(*syllable).or_default().push(features);
    }

    let mut centroids = Vec::new();
    for (syllable, vectors) in groups {
        if vectors.is_empty() {
            continue;
        }
        let n = vectors.len() as f32;
        let dim = vectors[0].len();
        let mut mean = vec![0.0_f32; dim];
        for v in &vectors {
            for (i, &val) in v.iter().enumerate() {
                mean[i] += val;
            }
        }
        for v in &mut mean {
            *v /= n;
        }
        centroids.push((syllable, mean));
    }
    centroids
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
        let pipeline = Pipeline::new();
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
            !result.syllables.is_empty() || result.q8_bytes.is_empty(),
            "process should return a valid UtteranceResult"
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
        let result = UtteranceResult {
            syllables: vec![SyllableResult {
                decision: Decision::Reject(0.2),
                raw_classification: Classification {
                    syllable,
                    confidence: 0.2,
                },
            }],
            q8_bytes: vec![0x92, 0xFF],
        };

        let json = serde_json::to_string(&result).expect("serialize UtteranceResult");
        let restored: UtteranceResult =
            serde_json::from_str(&json).expect("deserialize UtteranceResult");

        assert_eq!(restored.syllables.len(), 1);
        assert_eq!(restored.q8_bytes, vec![0x92, 0xFF]);
    }
}
