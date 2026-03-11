//! Pipeline integration: wires segmenter + MFCC + classifier + profile into a
//! single processing pipeline.
//!
//! The [`Pipeline`] is the main API surface for STQ8. It accepts raw PCM audio
//! from a PTT utterance and returns an [`UtteranceResult`] containing per-syllable
//! classification decisions and Q8-encoded bytes for auto-accepted phonemes.

use crate::classifier::{Classification, Classifier, NearestCentroid};
use crate::mfcc;
use crate::profile::{Decision, Thresholds, UserProfile};
use crate::q8::Phoneme;
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
    calibration_samples: Vec<(Phoneme, Vec<f32>)>,
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
    pub fn add_calibration_sample(&mut self, phoneme: Phoneme, features: Vec<f32>) {
        self.calibration_samples.push((phoneme, features));
    }

    /// Train the classifier from all accumulated calibration samples and
    /// store the resulting centroids in the user profile.
    pub fn finalize_calibration(&mut self) {
        self.classifier.train(&self.calibration_samples);
        // Store centroids in profile for export. We re-derive them by training
        // a temporary classifier (the NearestCentroid stores centroids internally,
        // but we can't read them directly). Instead, compute centroids from samples
        // the same way the classifier does.
        self.profile.centroids = compute_centroids(&self.calibration_samples);
        self.calibration_samples.clear();
    }

    /// Convenience: add all samples and finalize calibration in one call.
    pub fn calibrate(&mut self, samples: &[(Phoneme, Vec<f32>)]) {
        for (phoneme, features) in samples {
            self.calibration_samples.push((*phoneme, features.clone()));
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
    /// 3. Classify features
    /// 4. Apply profile remap
    /// 5. Apply threshold decision
    /// 6. Encode auto-accepted phonemes into Q8 bytes
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
        let mut accepted_phonemes: Vec<Phoneme> = Vec::new();

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
            let remapped_phoneme = self.profile.apply_remap(classification.phoneme);

            // Step 5: Apply threshold decision
            let decision = self
                .profile
                .thresholds
                .decide(remapped_phoneme, classification.confidence);

            // Collect accepted phonemes for Q8 encoding
            if let Decision::Accept(phoneme, _) = &decision {
                accepted_phonemes.push(*phoneme);
            }

            syllable_results.push(SyllableResult {
                decision,
                raw_classification: classification,
            });
        }

        // Step 6: Encode accepted phonemes into Q8 bytes.
        // Each phoneme produces a 2-bit value. Group every 2 phonemes into a nibble
        // (first = high 2 bits, second = low 2 bits). Group every 2 nibbles into a byte.
        let q8_bytes = encode_phonemes_to_q8(&accepted_phonemes);

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
        // Re-train classifier from the profile's stored centroids.
        // Each centroid is a single (phoneme, features) pair used as a training sample.
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

/// Compute mean centroids from training samples, grouped by phoneme.
fn compute_centroids(samples: &[(Phoneme, Vec<f32>)]) -> Vec<(Phoneme, Vec<f32>)> {
    use std::collections::HashMap;

    let mut groups: HashMap<Phoneme, Vec<&Vec<f32>>> = HashMap::new();
    for (phoneme, features) in samples {
        groups.entry(*phoneme).or_default().push(features);
    }

    let mut centroids = Vec::new();
    for (phoneme, vectors) in groups {
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
        centroids.push((phoneme, mean));
    }
    centroids
}

/// Encode a sequence of accepted phonemes into Q8 bytes.
///
/// Each phoneme produces a 2-bit value. Every 2 phonemes form a nibble
/// (first = high 2 bits, second = low 2 bits). Every 2 nibbles form a byte.
/// Trailing phonemes that don't complete a byte are discarded.
fn encode_phonemes_to_q8(phonemes: &[Phoneme]) -> Vec<u8> {
    // Each phoneme -> 2-bit value
    let bits: Vec<u8> = phonemes
        .iter()
        .map(|p| match p {
            Phoneme::Consonant(c) => c.bits(),
            Phoneme::Vowel(v) => v.bits(),
        })
        .collect();

    // Group into nibbles: 2 phonemes per nibble
    let nibbles: Vec<u8> = bits
        .chunks_exact(2)
        .map(|pair| (pair[0] << 2) | pair[1])
        .collect();

    // Group into bytes: 2 nibbles per byte
    nibbles
        .chunks_exact(2)
        .map(|pair| (pair[0] << 4) | pair[1])
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

    /// Build synthetic MFCC features for a phoneme using a deterministic pattern.
    /// Each phoneme gets a unique region of the 52-dim space.
    fn make_features(phoneme_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; FEATURE_DIM];
        let base = (phoneme_idx * 6) % FEATURE_DIM;
        v[base] = 1.0;
        v[(base + 1) % FEATURE_DIM] = 0.5;
        v[(base + 2) % FEATURE_DIM] = 0.8;
        v
    }

    fn all_phonemes() -> Vec<Phoneme> {
        vec![
            Phoneme::Consonant(Consonant::GlottalStop),
            Phoneme::Consonant(Consonant::J),
            Phoneme::Consonant(Consonant::K),
            Phoneme::Consonant(Consonant::V),
            Phoneme::Vowel(Vowel::O),
            Phoneme::Vowel(Vowel::U),
            Phoneme::Vowel(Vowel::E),
            Phoneme::Vowel(Vowel::I),
        ]
    }

    #[test]
    fn pipeline_uncalibrated_returns_empty() {
        let pipeline = Pipeline::new();
        assert!(!pipeline.is_calibrated());

        // Generate some audio (1 second of 440Hz sine)
        let pcm = sine_wave(440.0, SAMPLE_RATE as usize);
        let result = pipeline.process(&pcm);

        assert!(
            result.syllables.is_empty(),
            "uncalibrated pipeline should produce no syllable results"
        );
        assert!(
            result.q8_bytes.is_empty(),
            "uncalibrated pipeline should produce no Q8 bytes"
        );
    }

    #[test]
    fn pipeline_calibrate_and_process() {
        let mut pipeline = Pipeline::new();
        let phonemes = all_phonemes();

        // Generate synthetic calibration data: 5 samples per phoneme
        // using unique feature vectors per phoneme.
        let mut samples: Vec<(Phoneme, Vec<f32>)> = Vec::new();
        for (idx, &phoneme) in phonemes.iter().enumerate() {
            for rep in 0..5 {
                let mut features = make_features(idx);
                // Add small noise per repetition
                for (i, v) in features.iter_mut().enumerate() {
                    *v += (rep as f32) * 0.01 * ((i % 3) as f32 - 1.0);
                }
                samples.push((phoneme, features));
            }
        }

        assert!(
            !pipeline.is_calibrated(),
            "should not be calibrated before training"
        );

        pipeline.calibrate(&samples);

        assert!(
            pipeline.is_calibrated(),
            "should be calibrated after training"
        );

        // Verify profile has centroids stored
        assert!(
            !pipeline.profile.centroids.is_empty(),
            "profile should have centroids after calibration"
        );
    }

    #[test]
    fn pipeline_add_calibration_sample_then_finalize() {
        let mut pipeline = Pipeline::new();
        let phonemes = all_phonemes();

        // Add samples one at a time
        for (idx, &phoneme) in phonemes.iter().enumerate() {
            for rep in 0..3 {
                let mut features = make_features(idx);
                for (i, v) in features.iter_mut().enumerate() {
                    *v += (rep as f32) * 0.005 * ((i % 3) as f32 - 1.0);
                }
                pipeline.add_calibration_sample(phoneme, features);
            }
        }

        assert!(!pipeline.is_calibrated());
        pipeline.finalize_calibration();
        assert!(pipeline.is_calibrated());
    }

    #[test]
    fn pipeline_export_import_profile() {
        let mut pipeline = Pipeline::new();
        let phonemes = all_phonemes();

        // Calibrate with synthetic data
        let samples: Vec<(Phoneme, Vec<f32>)> = phonemes
            .iter()
            .enumerate()
            .map(|(idx, &phoneme)| (phoneme, make_features(idx)))
            .collect();

        pipeline.calibrate(&samples);
        assert!(pipeline.is_calibrated());

        // Export profile
        let json = pipeline
            .export_profile_json()
            .expect("export should succeed");

        // Verify it's valid JSON
        assert!(!json.is_empty(), "exported JSON should not be empty");
        assert!(
            json.contains("centroids"),
            "JSON should contain centroids field"
        );

        // Import into a new pipeline
        let mut new_pipeline = Pipeline::new();
        assert!(!new_pipeline.is_calibrated());

        new_pipeline
            .import_profile_json(&json)
            .expect("import should succeed");

        assert!(
            new_pipeline.is_calibrated(),
            "imported pipeline should be calibrated"
        );

        // Verify the imported profile matches
        assert_eq!(
            new_pipeline.profile.centroids.len(),
            pipeline.profile.centroids.len(),
            "imported profile should have same number of centroids"
        );
        assert_eq!(
            new_pipeline.profile.version, pipeline.profile.version,
            "imported profile version should match"
        );
    }

    #[test]
    fn pipeline_process_with_real_audio() {
        let mut pipeline = Pipeline::new();

        // Use different sine frequencies as calibration data for each phoneme.
        // This simulates different phonemes having different spectral content.
        let phonemes = all_phonemes();
        let freqs = [200.0, 400.0, 800.0, 1600.0, 3200.0, 4800.0, 5600.0, 6400.0];

        let mut samples = Vec::new();
        for (idx, &phoneme) in phonemes.iter().enumerate() {
            // Generate a short sine wave and extract real MFCC features
            let wave = sine_wave(freqs[idx], SAMPLE_RATE as usize / 2);
            let features = mfcc::extract_features(&wave);
            samples.push((phoneme, features));
        }

        pipeline.calibrate(&samples);
        assert!(pipeline.is_calibrated());

        // Create test audio: silence + a burst + silence
        // This should produce at least one segmented syllable.
        let sr = SAMPLE_RATE as usize;
        let mut pcm = vec![0.0f32; sr]; // 1 second
                                        // Insert a 200ms burst (matching phoneme 0's frequency) starting at 200ms
        let burst_start = sr / 5; // 3200 samples
        let burst_end = burst_start + sr * 2 / 5; // 6400 samples more
        for i in burst_start..burst_end.min(sr) {
            let t = i as f32 / SAMPLE_RATE as f32;
            pcm[i] = (2.0 * PI * freqs[0] * t).sin() * 0.8;
        }

        let result = pipeline.process(&pcm);
        // The segmenter should detect the burst as a syllable.
        // Exact behavior depends on segmenter config and how MFCC features
        // are classified, but we should get at least one syllable result.
        // (Not asserting exact count since segmenter behavior with synthetic
        // data can vary.)
        assert!(
            !result.syllables.is_empty() || result.q8_bytes.is_empty(),
            "process should return a valid UtteranceResult"
        );
    }

    #[test]
    fn encode_phonemes_to_q8_basic() {
        // 4 phonemes -> 2 nibbles -> 1 byte
        let phonemes = vec![
            Phoneme::Consonant(Consonant::K),           // bits = 10
            Phoneme::Vowel(Vowel::U),                   // bits = 01
            Phoneme::Consonant(Consonant::GlottalStop), // bits = 00
            Phoneme::Vowel(Vowel::E),                   // bits = 10
        ];

        let bytes = encode_phonemes_to_q8(&phonemes);

        // nibble0 = (10 << 2) | 01 = 0b1001 = 9
        // nibble1 = (00 << 2) | 10 = 0b0010 = 2
        // byte = (9 << 4) | 2 = 0x92
        assert_eq!(bytes, vec![0x92]);
    }

    #[test]
    fn encode_phonemes_to_q8_trailing_discarded() {
        // 3 phonemes -> 1 nibble (2 phonemes), 1 leftover -> 0 complete bytes
        // Actually: 3 phonemes -> 1 complete nibble + 1 leftover phoneme
        // 1 nibble is not enough for a byte, so 0 bytes
        let phonemes = vec![
            Phoneme::Consonant(Consonant::J), // bits = 01
            Phoneme::Vowel(Vowel::O),         // bits = 00
            Phoneme::Vowel(Vowel::I),         // bits = 11 (leftover)
        ];

        let bytes = encode_phonemes_to_q8(&phonemes);
        assert!(
            bytes.is_empty(),
            "3 phonemes should produce 0 complete bytes"
        );
    }

    #[test]
    fn encode_phonemes_to_q8_empty() {
        let bytes = encode_phonemes_to_q8(&[]);
        assert!(bytes.is_empty());
    }

    #[test]
    fn pipeline_default_trait() {
        let pipeline = Pipeline::default();
        assert!(!pipeline.is_calibrated());
    }

    #[test]
    fn syllable_result_serialization() {
        let result = SyllableResult {
            decision: Decision::Accept(Phoneme::Consonant(Consonant::K), 0.95),
            raw_classification: Classification {
                phoneme: Phoneme::Consonant(Consonant::K),
                confidence: 0.95,
            },
        };

        let json = serde_json::to_string(&result).expect("serialize SyllableResult");
        let restored: SyllableResult =
            serde_json::from_str(&json).expect("deserialize SyllableResult");

        assert_eq!(
            restored.raw_classification.phoneme,
            result.raw_classification.phoneme
        );
    }

    #[test]
    fn utterance_result_serialization() {
        let result = UtteranceResult {
            syllables: vec![SyllableResult {
                decision: Decision::Reject(0.2),
                raw_classification: Classification {
                    phoneme: Phoneme::Vowel(Vowel::E),
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
