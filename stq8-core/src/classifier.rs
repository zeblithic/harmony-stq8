//! Syllable classifier with nearest-centroid implementation.
//!
//! Takes a 52-dimensional MFCC feature vector and returns the most likely
//! [`Syllable`] with a confidence score. The [`Classifier`] trait abstracts
//! the inference backend so we can swap in Coral/GPU backends later.

use crate::mfcc::FEATURE_DIM;
use crate::q8::Syllable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Softmax temperature for confidence computation.
///
/// Tuned for cosine similarity values in the 0.5–1.0 range with 16 classes.
/// Lower temperature → sharper discrimination. Higher → more uniform.
const SOFTMAX_TEMPERATURE: f32 = 0.2;

/// Result of classifying a single syllable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub syllable: Syllable,
    pub confidence: f32,
}

/// Classifier trait — abstracts inference backend.
pub trait Classifier {
    fn classify(&self, features: &[f32]) -> Option<Classification>;
    fn train(&mut self, samples: &[(Syllable, Vec<f32>)]);
    fn is_trained(&self) -> bool;
}

/// Cosine-similarity nearest-centroid classifier.
///
/// Stores one centroid (mean feature vector) per syllable. Classification
/// finds the centroid with highest cosine similarity to the input.
/// Confidence is softmax-normalized: how much probability mass is on the
/// best class given the similarity distribution.
pub struct NearestCentroid {
    centroids: Vec<(Syllable, Vec<f32>)>,
}

impl NearestCentroid {
    /// Create a new untrained classifier.
    pub fn new() -> Self {
        Self {
            centroids: Vec::new(),
        }
    }
}

impl NearestCentroid {
    /// Access the trained centroids (syllable, mean feature vector) pairs.
    pub fn centroids(&self) -> &[(Syllable, Vec<f32>)] {
        &self.centroids
    }

    /// Load pre-computed centroids directly, bypassing the `train` averaging step.
    ///
    /// Use this when importing a serialized profile where centroids are already
    /// final mean vectors. Unlike `train`, this preserves the input exactly —
    /// duplicate syllable entries are not merged.
    pub fn load_centroids(&mut self, centroids: Vec<(Syllable, Vec<f32>)>) {
        self.centroids = centroids;
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

        let mut similarities: Vec<(Syllable, f32)> = self
            .centroids
            .iter()
            .filter(|(_, centroid)| centroid.len() == FEATURE_DIM)
            .map(|(syllable, centroid)| (*syllable, cosine_similarity(features, centroid)))
            .collect();

        if similarities.is_empty() {
            return None;
        }

        // Sort descending by similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = similarities[0];
        let confidence = if similarities.len() == 1 {
            1.0
        } else {
            // Softmax confidence: probability mass on the best class.
            // Subtract max for numerical stability before exp.
            let max_sim = best.1;
            let sum_exp: f32 = similarities
                .iter()
                .map(|(_, s)| ((s - max_sim) / SOFTMAX_TEMPERATURE).exp())
                .sum();
            (1.0 / sum_exp).clamp(0.0, 1.0)
        };

        Some(Classification {
            syllable: best.0,
            confidence,
        })
    }

    fn train(&mut self, samples: &[(Syllable, Vec<f32>)]) {
        // Group samples by syllable
        let mut groups: HashMap<Syllable, Vec<&Vec<f32>>> = HashMap::new();
        for (syllable, features) in samples {
            groups.entry(*syllable).or_default().push(features);
        }

        // Compute mean feature vector per syllable
        self.centroids.clear();
        for (syllable, vectors) in groups {
            let dim = vectors[0].len();
            // Skip samples with mismatched dimensions
            let valid: Vec<&&Vec<f32>> = vectors.iter().filter(|v| v.len() == dim).collect();
            let n = valid.len() as f32;
            let mut mean = vec![0.0_f32; dim];
            for v in &valid {
                for (i, &val) in v.iter().enumerate() {
                    mean[i] += val;
                }
            }
            for v in &mut mean {
                *v /= n;
            }
            self.centroids.push((syllable, mean));
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

    fn make_features(syllable_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; FEATURE_DIM];
        // Each syllable gets a unique region of the 52-dim space.
        // With 16 syllables × 3 dims each = 48 dims used, fits in 52.
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
    fn untrained_returns_none() {
        let nc = NearestCentroid::new();
        assert!(!nc.is_trained());
        let features = vec![0.0; FEATURE_DIM];
        assert!(nc.classify(&features).is_none());
    }

    #[test]
    fn trained_classifies_correctly() {
        let mut nc = NearestCentroid::new();
        let syllables = all_syllables();

        // 16 syllables × 5 samples each
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

        nc.train(&samples);
        assert!(nc.is_trained());

        for (idx, &syllable) in syllables.iter().enumerate() {
            let features = make_features(idx);
            let result = nc.classify(&features).expect("should classify");
            assert_eq!(
                result.syllable, syllable,
                "syllable index {idx} ({syllable}) should classify correctly"
            );
            assert!(
                result.confidence > 0.5,
                "syllable index {idx} confidence {:.3} should be > 0.5",
                result.confidence
            );
        }
    }

    #[test]
    fn exact_centroid_high_confidence() {
        let mut nc = NearestCentroid::new();
        let syllables = all_syllables();

        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();

        nc.train(&samples);

        let features = make_features(0);
        let result = nc.classify(&features).expect("should classify");
        assert_eq!(result.syllable, syllables[0]);
        assert!(
            result.confidence > 0.8,
            "exact centroid match confidence {:.3} should be > 0.8",
            result.confidence
        );
    }

    #[test]
    fn equidistant_low_confidence() {
        let mut nc = NearestCentroid::new();
        let syllables = all_syllables();

        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();

        nc.train(&samples);

        // Zero vector is equidistant from all centroids (cosine similarity = 0.0)
        let zero_features = vec![0.0; FEATURE_DIM];
        let result = nc.classify(&zero_features);
        if let Some(classification) = result {
            assert!(
                classification.confidence < 0.5,
                "equidistant (zero-vector) confidence {:.3} should be < 0.5",
                classification.confidence
            );
        }
    }

    #[test]
    fn wrong_feature_dim_returns_none() {
        let mut nc = NearestCentroid::new();
        let syllables = all_syllables();

        let samples: Vec<(Syllable, Vec<f32>)> = syllables
            .iter()
            .enumerate()
            .map(|(idx, &syllable)| (syllable, make_features(idx)))
            .collect();

        nc.train(&samples);

        assert!(nc.classify(&vec![1.0; FEATURE_DIM - 1]).is_none());
        assert!(nc.classify(&vec![1.0; FEATURE_DIM + 1]).is_none());
        assert!(nc.classify(&Vec::<f32>::new()).is_none());
    }

    #[test]
    fn single_centroid_full_confidence() {
        let mut nc = NearestCentroid::new();
        let syllable = Syllable::new(Consonant::K, Vowel::O);
        let features = make_features(8); // KO = nibble 8

        nc.train(&[(syllable, features.clone())]);
        assert!(nc.is_trained());

        let result = nc.classify(&features).expect("should classify");
        assert_eq!(result.syllable, syllable);
        assert!(
            (result.confidence - 1.0).abs() < f32::EPSILON,
            "single centroid confidence should be exactly 1.0, got {:.6}",
            result.confidence
        );
    }

    #[test]
    fn softmax_confidence_has_good_dynamic_range() {
        // With the old formula (1 - second/best), close similarities gave near-zero
        // confidence. Softmax should give usable values even when similarities cluster.
        let mut nc = NearestCentroid::new();

        // Train with only 2 syllables that are somewhat similar
        let s1 = Syllable::from_nibble(0);
        let s2 = Syllable::from_nibble(1);

        let mut f1 = vec![0.0f32; FEATURE_DIM];
        f1[0] = 1.0;
        f1[1] = 0.5;
        let mut f2 = vec![0.0f32; FEATURE_DIM];
        f2[0] = 0.8; // somewhat similar to f1
        f2[2] = 0.6;

        nc.train(&[(s1, f1.clone()), (s2, f2)]);

        let result = nc.classify(&f1).expect("should classify");
        assert_eq!(result.syllable, s1);
        // Softmax should give meaningful confidence even with somewhat similar centroids
        assert!(
            result.confidence > 0.3,
            "softmax confidence {:.3} should have usable dynamic range",
            result.confidence
        );
    }

    #[test]
    fn wrong_centroid_dim_skipped_in_classify() {
        let mut nc = NearestCentroid::new();
        let s1 = Syllable::from_nibble(0);
        let s2 = Syllable::from_nibble(1);

        // Manually build centroids with mismatched dimensions
        let good = vec![1.0f32; FEATURE_DIM];
        let bad = vec![1.0f32; 30]; // wrong dimension
        nc.centroids = vec![(s1, good.clone()), (s2, bad)];

        // Should still classify using only the good centroid
        let result = nc.classify(&good).expect("should classify with valid centroid");
        assert_eq!(result.syllable, s1);

        // If ALL centroids are wrong dimension, returns None
        nc.centroids = vec![(s1, vec![1.0f32; 30])];
        assert!(nc.classify(&good).is_none());
    }

    #[test]
    fn load_centroids_preserves_duplicates() {
        let mut nc = NearestCentroid::new();
        let s1 = Syllable::from_nibble(0);

        // Two entries for the same syllable — load_centroids must not merge them
        let c1 = vec![1.0f32; FEATURE_DIM];
        let c2 = vec![0.0f32; FEATURE_DIM];
        nc.load_centroids(vec![(s1, c1.clone()), (s1, c2.clone())]);

        assert_eq!(nc.centroids().len(), 2, "load_centroids should preserve duplicate entries");
        assert_eq!(nc.centroids()[0].1, c1);
        assert_eq!(nc.centroids()[1].1, c2);
    }

    #[test]
    fn load_centroids_vs_train_with_duplicates() {
        let s1 = Syllable::from_nibble(0);
        let c1 = vec![1.0f32; FEATURE_DIM];
        let c2 = vec![0.0f32; FEATURE_DIM];
        let entries = vec![(s1, c1), (s1, c2)];

        // train() merges duplicates into one averaged centroid
        let mut trained = NearestCentroid::new();
        trained.train(&entries);
        assert_eq!(trained.centroids().len(), 1, "train should merge duplicates");

        // load_centroids() preserves them as-is
        let mut loaded = NearestCentroid::new();
        loaded.load_centroids(entries);
        assert_eq!(loaded.centroids().len(), 2, "load_centroids should not merge");
    }
}
