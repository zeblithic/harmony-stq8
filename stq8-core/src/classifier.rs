//! Phoneme classifier with nearest-centroid implementation.
//!
//! Takes a 52-dimensional MFCC feature vector and returns the most likely
//! [`Phoneme`] with a confidence score. The [`Classifier`] trait abstracts
//! the inference backend so we can swap in Coral/GPU backends later.

use crate::mfcc::FEATURE_DIM;
use crate::q8::Phoneme;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of classifying a single phoneme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub phoneme: Phoneme,
    pub confidence: f32,
}

/// Classifier trait — abstracts inference backend.
pub trait Classifier {
    fn classify(&self, features: &[f32]) -> Option<Classification>;
    fn train(&mut self, samples: &[(Phoneme, Vec<f32>)]);
    fn is_trained(&self) -> bool;
}

/// Cosine-similarity nearest-centroid classifier.
///
/// Stores one centroid (mean feature vector) per phoneme. Classification
/// finds the centroid with highest cosine similarity to the input.
/// Confidence is based on daylight between the best and second-best match.
pub struct NearestCentroid {
    centroids: Vec<(Phoneme, Vec<f32>)>,
}

impl NearestCentroid {
    /// Create a new untrained classifier.
    pub fn new() -> Self {
        Self {
            centroids: Vec::new(),
        }
    }
}

impl Default for NearestCentroid {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl Classifier for NearestCentroid {
    fn classify(&self, features: &[f32]) -> Option<Classification> {
        if self.centroids.is_empty() || features.len() != FEATURE_DIM {
            return None;
        }

        let mut similarities: Vec<(Phoneme, f32)> = self
            .centroids
            .iter()
            .map(|(phoneme, centroid)| (*phoneme, cosine_similarity(features, centroid)))
            .collect();

        // Sort descending by similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = similarities[0];
        let confidence = if similarities.len() == 1 {
            1.0
        } else {
            let second_best = similarities[1].1;
            // Confidence = 1 - (second_best / best), clamped to [0, 1]
            if best.1 == 0.0 {
                0.0
            } else {
                (1.0 - second_best / best.1).clamp(0.0, 1.0)
            }
        };

        Some(Classification {
            phoneme: best.0,
            confidence,
        })
    }

    fn train(&mut self, samples: &[(Phoneme, Vec<f32>)]) {
        // Group samples by phoneme
        let mut groups: HashMap<Phoneme, Vec<&Vec<f32>>> = HashMap::new();
        for (phoneme, features) in samples {
            groups.entry(*phoneme).or_default().push(features);
        }

        // Compute mean feature vector per phoneme
        self.centroids.clear();
        for (phoneme, vectors) in groups {
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
            self.centroids.push((phoneme, mean));
        }
    }

    fn is_trained(&self) -> bool {
        !self.centroids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q8::{Consonant, Vowel};

    fn make_features(phoneme_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; FEATURE_DIM];
        // Each phoneme gets a unique 6-dimensional block within 52 dims,
        // ensuring zero cosine similarity between different phonemes.
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
    fn untrained_returns_none() {
        let nc = NearestCentroid::new();
        assert!(!nc.is_trained());
        let features = vec![0.0; FEATURE_DIM];
        assert!(nc.classify(&features).is_none());
    }

    #[test]
    fn trained_classifies_correctly() {
        let mut nc = NearestCentroid::new();
        let phonemes = all_phonemes();

        // 8 phonemes x 5 samples each
        let mut samples: Vec<(Phoneme, Vec<f32>)> = Vec::new();
        for (idx, &phoneme) in phonemes.iter().enumerate() {
            for rep in 0..5 {
                let mut features = make_features(idx);
                // Add small noise per repetition so samples aren't identical
                for (i, v) in features.iter_mut().enumerate() {
                    *v += (rep as f32) * 0.01 * ((i % 3) as f32 - 1.0);
                }
                samples.push((phoneme, features));
            }
        }

        nc.train(&samples);
        assert!(nc.is_trained());

        // Classify each phoneme's centroid-like vector
        for (idx, &phoneme) in phonemes.iter().enumerate() {
            let features = make_features(idx);
            let result = nc.classify(&features).expect("should classify");
            assert_eq!(
                result.phoneme, phoneme,
                "phoneme index {idx} should classify correctly"
            );
            assert!(
                result.confidence > 0.5,
                "phoneme index {idx} confidence {:.3} should be > 0.5",
                result.confidence
            );
        }
    }

    #[test]
    fn exact_centroid_high_confidence() {
        let mut nc = NearestCentroid::new();
        let phonemes = all_phonemes();

        let samples: Vec<(Phoneme, Vec<f32>)> = phonemes
            .iter()
            .enumerate()
            .map(|(idx, &phoneme)| (phoneme, make_features(idx)))
            .collect();

        nc.train(&samples);

        // Classify with exact centroid
        let features = make_features(0);
        let result = nc.classify(&features).expect("should classify");
        assert_eq!(result.phoneme, phonemes[0]);
        assert!(
            result.confidence > 0.8,
            "exact centroid match confidence {:.3} should be > 0.8",
            result.confidence
        );
    }

    #[test]
    fn equidistant_low_confidence() {
        let mut nc = NearestCentroid::new();
        let phonemes = all_phonemes();

        let samples: Vec<(Phoneme, Vec<f32>)> = phonemes
            .iter()
            .enumerate()
            .map(|(idx, &phoneme)| (phoneme, make_features(idx)))
            .collect();

        nc.train(&samples);

        // Zero vector is equidistant from all centroids (cosine similarity = 0.0)
        let zero_features = vec![0.0; FEATURE_DIM];
        let result = nc.classify(&zero_features);
        // Zero vector yields cosine similarity = 0 with everything, so confidence should be low
        if let Some(classification) = result {
            assert!(
                classification.confidence < 0.5,
                "equidistant (zero-vector) confidence {:.3} should be < 0.5",
                classification.confidence
            );
        }
        // None is also acceptable since all similarities are 0
    }

    #[test]
    fn wrong_feature_dim_returns_none() {
        let mut nc = NearestCentroid::new();
        let phonemes = all_phonemes();

        let samples: Vec<(Phoneme, Vec<f32>)> = phonemes
            .iter()
            .enumerate()
            .map(|(idx, &phoneme)| (phoneme, make_features(idx)))
            .collect();

        nc.train(&samples);

        // Wrong length: too short
        let short = vec![1.0; FEATURE_DIM - 1];
        assert!(nc.classify(&short).is_none());

        // Wrong length: too long
        let long = vec![1.0; FEATURE_DIM + 1];
        assert!(nc.classify(&long).is_none());

        // Empty
        let empty: Vec<f32> = Vec::new();
        assert!(nc.classify(&empty).is_none());
    }

    #[test]
    fn single_centroid_full_confidence() {
        let mut nc = NearestCentroid::new();
        let phoneme = Phoneme::Consonant(Consonant::K);
        let features = make_features(2);

        nc.train(&[(phoneme, features.clone())]);
        assert!(nc.is_trained());

        let result = nc.classify(&features).expect("should classify");
        assert_eq!(result.phoneme, phoneme);
        assert!(
            (result.confidence - 1.0).abs() < f32::EPSILON,
            "single centroid confidence should be exactly 1.0, got {:.6}",
            result.confidence
        );
    }
}
