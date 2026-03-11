//! Transversal classifier using orthogonal Latin square phrase intersection.
//!
//! Two phrases of 4 syllables each (8 total) form orthogonal transversals of the
//! 4x4 consonant-vowel grid. Any of the 16 possible Q8 nibbles can be reconstructed
//! by intersecting the Top-2 consonant and vowel sets from each phrase.
//!
//! This reduces the training burden from 16 syllables to 8, while maintaining
//! the ability to classify all 16 via the intersection algorithm.

use crate::classifier::cosine_similarity;
use crate::mfcc::FEATURE_DIM;
use crate::q8::{Consonant, Syllable, Vowel};
use serde::{Deserialize, Serialize};

/// Phrase 1 (diagonal): 'O(0), JU(5), KE(10), VI(15)
pub const PHRASE_1: [Syllable; 4] = [
    Syllable {
        consonant: Consonant::GlottalStop,
        vowel: Vowel::O,
    },
    Syllable {
        consonant: Consonant::J,
        vowel: Vowel::U,
    },
    Syllable {
        consonant: Consonant::K,
        vowel: Vowel::E,
    },
    Syllable {
        consonant: Consonant::V,
        vowel: Vowel::I,
    },
];

/// Phrase 2 (anti-diagonal): 'I(3), JE(6), KO(8), VU(13)
pub const PHRASE_2: [Syllable; 4] = [
    Syllable {
        consonant: Consonant::GlottalStop,
        vowel: Vowel::I,
    },
    Syllable {
        consonant: Consonant::J,
        vowel: Vowel::E,
    },
    Syllable {
        consonant: Consonant::K,
        vowel: Vowel::O,
    },
    Syllable {
        consonant: Consonant::V,
        vowel: Vowel::U,
    },
];

/// Result of transversal classification, including per-phrase Top-2 details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransversalClassification {
    pub syllable: Syllable,
    pub confidence: f32,
    pub phrase1_top2: [(Syllable, f32); 2],
    pub phrase2_top2: [(Syllable, f32); 2],
}

/// Errors that can occur during transversal training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    /// phrase_index was not 0 or 1.
    InvalidPhraseIndex(u8),
    /// The syllable does not belong to the specified phrase.
    WrongSyllableForPhrase {
        syllable: Syllable,
        phrase_index: u8,
    },
    /// Not all 4 syllables in a phrase had training samples.
    IncompletePhraseTraining {
        phrase_index: u8,
        missing_count: usize,
    },
}

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::InvalidPhraseIndex(idx) => {
                write!(f, "invalid phrase index: {idx} (must be 0 or 1)")
            }
            TrainError::WrongSyllableForPhrase {
                syllable,
                phrase_index,
            } => {
                write!(
                    f,
                    "syllable {syllable} does not belong to phrase {phrase_index}"
                )
            }
            TrainError::IncompletePhraseTraining {
                phrase_index,
                missing_count,
            } => {
                write!(
                    f,
                    "phrase {phrase_index} is missing {missing_count} syllable(s)"
                )
            }
        }
    }
}

impl std::error::Error for TrainError {}

/// Classifier that uses two orthogonal Latin square transversal phrases
/// to reconstruct any of the 16 Q8 nibbles via Top-2-per-phrase intersection.
pub struct TransversalClassifier {
    phrase_centroids: [Vec<(Syllable, Vec<f32>)>; 2],
    trained: bool,
}

impl TransversalClassifier {
    /// Create a new untrained transversal classifier.
    pub fn new() -> Self {
        Self {
            phrase_centroids: [Vec::new(), Vec::new()],
            trained: false,
        }
    }

    /// Returns whether the classifier has been trained (both phrases have 4 centroids).
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Access centroids for a given phrase (0 or 1).
    ///
    /// Returns `None` if `phrase` is not 0 or 1.
    pub fn phrase_centroids(&self, phrase: u8) -> Option<&[(Syllable, Vec<f32>)]> {
        match phrase {
            0 => Some(&self.phrase_centroids[0]),
            1 => Some(&self.phrase_centroids[1]),
            _ => None,
        }
    }

    /// Train from tagged samples.
    ///
    /// Each sample is `(phrase_index, syllable, features)`. Validates:
    /// - `phrase_index` is 0 or 1
    /// - `syllable` belongs to the specified phrase
    /// - All 4 syllables per phrase have at least one sample
    ///
    /// Computes mean centroid per syllable within each phrase.
    pub fn train(&mut self, samples: &[(u8, Syllable, Vec<f32>)]) -> Result<(), TrainError> {
        // Validate all samples first
        for (phrase_index, syllable, _) in samples {
            if *phrase_index > 1 {
                return Err(TrainError::InvalidPhraseIndex(*phrase_index));
            }
            let phrase = if *phrase_index == 0 {
                &PHRASE_1
            } else {
                &PHRASE_2
            };
            if !phrase.contains(syllable) {
                return Err(TrainError::WrongSyllableForPhrase {
                    syllable: *syllable,
                    phrase_index: *phrase_index,
                });
            }
        }

        // Group samples by (phrase_index, syllable) and compute mean centroids
        let mut phrase_groups: [std::collections::HashMap<Syllable, Vec<&Vec<f32>>>; 2] = [
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        ];

        for (phrase_index, syllable, features) in samples {
            phrase_groups[*phrase_index as usize]
                .entry(*syllable)
                .or_default()
                .push(features);
        }

        // Validate completeness: each phrase must have all 4 syllables
        for (pi, phrase) in [&PHRASE_1, &PHRASE_2].iter().enumerate() {
            let missing: usize = phrase
                .iter()
                .filter(|s| !phrase_groups[pi].contains_key(s))
                .count();
            if missing > 0 {
                return Err(TrainError::IncompletePhraseTraining {
                    phrase_index: pi as u8,
                    missing_count: missing,
                });
            }
        }

        // Compute mean centroids
        for (pi, group) in phrase_groups.iter().enumerate() {
            self.phrase_centroids[pi].clear();
            for (syllable, vectors) in group {
                let dim = vectors[0].len();
                let n = vectors.len() as f32;
                let mut mean = vec![0.0_f32; dim];
                for v in vectors {
                    for (i, &val) in v.iter().enumerate() {
                        mean[i] += val;
                    }
                }
                for v in &mut mean {
                    *v /= n;
                }
                self.phrase_centroids[pi].push((*syllable, mean));
            }
        }

        self.trained = self.phrase_centroids[0].len() == 4 && self.phrase_centroids[1].len() == 4;
        Ok(())
    }

    /// Load pre-computed centroids directly for profile import.
    ///
    /// Sets `trained = true` if both phrases have exactly 4 centroids.
    pub fn load_centroids(
        &mut self,
        phrase0: Vec<(Syllable, Vec<f32>)>,
        phrase1: Vec<(Syllable, Vec<f32>)>,
    ) {
        self.phrase_centroids[0] = phrase0;
        self.phrase_centroids[1] = phrase1;
        self.trained = self.phrase_centroids[0].len() == 4 && self.phrase_centroids[1].len() == 4;
    }

    /// Classify a feature vector using Top-2-per-phrase intersection.
    ///
    /// Algorithm:
    /// 1. Compute cosine similarity to all 8 centroids (4 per phrase)
    /// 2. For each phrase, rank by similarity, take Top-2
    /// 3. Intersect consonant sets across phrases -> single consonant
    /// 4. Intersect vowel sets across phrases -> single vowel
    /// 5. Combine -> predicted nibble
    /// 6. Confidence = mean of per-phrase margins (top2_mean - bottom2_mean), clamped to [0, 1]
    /// 7. Return None if intersection fails or classifier is untrained
    pub fn classify(&self, features: &[f32]) -> Option<TransversalClassification> {
        if !self.trained || features.len() != FEATURE_DIM {
            return None;
        }

        // Compute similarities for each phrase
        let mut phrase_sims: [Vec<(Syllable, f32)>; 2] = [Vec::new(), Vec::new()];
        for (sims, centroids) in phrase_sims.iter_mut().zip(self.phrase_centroids.iter()) {
            for (syllable, centroid) in centroids {
                if centroid.len() != FEATURE_DIM {
                    return None;
                }
                let sim = cosine_similarity(features, centroid);
                sims.push((*syllable, sim));
            }
            // Sort descending by similarity
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Ensure we have at least 2 entries per phrase
        if phrase_sims[0].len() < 2 || phrase_sims[1].len() < 2 {
            return None;
        }

        // Top-2 per phrase
        let p1_top2: [(Syllable, f32); 2] = [phrase_sims[0][0], phrase_sims[0][1]];
        let p2_top2: [(Syllable, f32); 2] = [phrase_sims[1][0], phrase_sims[1][1]];

        // For each phrase, collect the effective consonant and vowel candidates.
        // When the best match dominates (>> second best), the second entry is noise
        // and should not contribute to the intersection. In that case, use only the
        // best match's consonant and vowel from that phrase.
        const DOMINANCE_RATIO: f32 = 5.0;

        let effective_candidates = |top2: &[(Syllable, f32); 2]| -> (Vec<Consonant>, Vec<Vowel>) {
            let best_sim = top2[0].1.max(0.0);
            let second_sim = top2[1].1.max(0.0);
            let dominated = best_sim > second_sim * DOMINANCE_RATIO && second_sim < 0.1;
            if dominated {
                // Best match dominates — use only its consonant and vowel
                (vec![top2[0].0.consonant], vec![top2[0].0.vowel])
            } else {
                (
                    top2.iter().map(|(s, _)| s.consonant).collect(),
                    top2.iter().map(|(s, _)| s.vowel).collect(),
                )
            }
        };

        let (p1_consonants, p1_vowels) = effective_candidates(&p1_top2);
        let (p2_consonants, p2_vowels) = effective_candidates(&p2_top2);

        // Intersect consonant sets across phrases
        let consonant_intersection: Vec<Consonant> = p1_consonants
            .iter()
            .filter(|c| p2_consonants.contains(c))
            .copied()
            .collect();

        if consonant_intersection.len() != 1 {
            return None;
        }
        let consonant = consonant_intersection[0];

        // Intersect vowel sets across phrases
        let vowel_intersection: Vec<Vowel> = p1_vowels
            .iter()
            .filter(|v| p2_vowels.contains(v))
            .copied()
            .collect();

        if vowel_intersection.len() != 1 {
            return None;
        }
        let vowel = vowel_intersection[0];

        let syllable = Syllable::new(consonant, vowel);

        // Confidence: mean of per-phrase margins (top2_mean - bottom2_mean), clamped to [0, 1]
        let mut total_margin = 0.0_f32;
        for sims in &phrase_sims {
            let top2_mean = (sims[0].1 + sims[1].1) / 2.0;
            let bottom2_mean = (sims[2].1 + sims[3].1) / 2.0;
            total_margin += top2_mean - bottom2_mean;
        }
        let confidence = (total_margin / 2.0).clamp(0.0, 1.0);

        Some(TransversalClassification {
            syllable,
            confidence,
            phrase1_top2: p1_top2,
            phrase2_top2: p2_top2,
        })
    }
}

impl Default for TransversalClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Build features using a consonant+vowel decomposition.
    ///
    /// Dimensions 0..25 encode the consonant (strong activation for matching consonant,
    /// small noise for others to avoid zero-similarity ties).
    /// Dimensions 26..51 encode the vowel (same scheme).
    ///
    /// This ensures cosine similarity reflects consonant AND vowel overlap,
    /// and that all centroids have nonzero similarity (no ties at 0), making
    /// the Top-2-per-phrase intersection algorithm work correctly.
    fn make_cv_features(consonant_bits: u8, vowel_bits: u8) -> Vec<f32> {
        let mut v = vec![0.0f32; FEATURE_DIM];
        // Consonant region: dims 0..23 (4 consonants × 6 dims)
        // Strong activation for matching consonant, weak noise for others
        for c in 0..4u8 {
            let c_base = (c as usize) * 6;
            if c == consonant_bits {
                v[c_base] = 1.0;
                v[c_base + 1] = 0.7;
                v[c_base + 2] = 0.5;
            } else {
                // Small but distinct noise so no two non-matching consonants tie
                let noise = 0.01 * (c as f32 + 1.0);
                v[c_base] = noise;
            }
        }
        // Vowel region: dims 26..49 (4 vowels × 6 dims)
        for vw in 0..4u8 {
            let v_base = 26 + (vw as usize) * 6;
            if vw == vowel_bits {
                v[v_base] = 1.0;
                v[v_base + 1] = 0.7;
                v[v_base + 2] = 0.5;
            } else {
                let noise = 0.01 * (vw as f32 + 1.0);
                v[v_base] = noise;
            }
        }
        v
    }

    fn make_syllable_features(s: &Syllable) -> Vec<f32> {
        make_cv_features(s.consonant.bits(), s.vowel.bits())
    }

    fn train_with_synthetic(tc: &mut TransversalClassifier) {
        let mut samples = Vec::new();
        for s in &PHRASE_1 {
            samples.push((0u8, *s, make_syllable_features(s)));
        }
        for s in &PHRASE_2 {
            samples.push((1u8, *s, make_syllable_features(s)));
        }
        tc.train(&samples).expect("training should succeed");
    }

    #[test]
    fn phrases_cover_all_consonants_and_vowels() {
        for phrase in [&PHRASE_1, &PHRASE_2] {
            let consonants: HashSet<Consonant> = phrase.iter().map(|s| s.consonant).collect();
            let vowels: HashSet<Vowel> = phrase.iter().map(|s| s.vowel).collect();
            assert_eq!(consonants.len(), 4, "phrase should cover all 4 consonants");
            assert_eq!(vowels.len(), 4, "phrase should cover all 4 vowels");
        }
    }

    #[test]
    fn phrases_are_orthogonal() {
        let p1_set: HashSet<Syllable> = PHRASE_1.iter().copied().collect();
        let p2_set: HashSet<Syllable> = PHRASE_2.iter().copied().collect();
        let intersection: HashSet<&Syllable> = p1_set.intersection(&p2_set).collect();
        assert!(
            intersection.is_empty(),
            "phrases should share no syllables, but share: {:?}",
            intersection
        );
    }

    #[test]
    fn untrained_returns_none() {
        let tc = TransversalClassifier::new();
        assert!(!tc.is_trained());
        let features = vec![0.0; FEATURE_DIM];
        assert!(tc.classify(&features).is_none());
    }

    #[test]
    fn reconstructs_all_16_nibbles() {
        let mut tc = TransversalClassifier::new();
        train_with_synthetic(&mut tc);

        // For each of the 16 nibbles, construct a feature vector from its
        // consonant+vowel decomposition. Since centroids use the same scheme,
        // cosine similarity will be high for centroids sharing the same consonant
        // or vowel, enabling the Top-2 intersection to reconstruct any nibble.
        for nibble in 0..16u8 {
            let target = Syllable::from_nibble(nibble);
            let features = make_syllable_features(&target);

            let result = tc
                .classify(&features)
                .unwrap_or_else(|| panic!("nibble {nibble} ({target}) should classify"));
            assert_eq!(
                result.syllable, target,
                "nibble {nibble}: expected {target}, got {}",
                result.syllable
            );
        }
    }

    #[test]
    fn calibrated_syllables_have_high_confidence() {
        let mut tc = TransversalClassifier::new();
        train_with_synthetic(&mut tc);

        // Test each calibrated syllable (exact centroid from phrase 1)
        for s in &PHRASE_1 {
            let features = make_syllable_features(s);
            let result = tc.classify(&features);
            // Calibrated syllable is in one phrase — classify may or may not succeed
            // depending on intersection. If it does, confidence should be reasonable.
            if let Some(classification) = result {
                assert!(
                    classification.confidence > 0.3,
                    "phrase1 ({s}) confidence {:.3} should be > 0.3",
                    classification.confidence
                );
            }
        }

        // Test each calibrated syllable (exact centroid from phrase 2)
        for s in &PHRASE_2 {
            let features = make_syllable_features(s);
            let result = tc.classify(&features);
            if let Some(classification) = result {
                assert!(
                    classification.confidence > 0.3,
                    "phrase2 ({s}) confidence {:.3} should be > 0.3",
                    classification.confidence
                );
            }
        }
    }

    #[test]
    fn wrong_feature_dim_returns_none() {
        let mut tc = TransversalClassifier::new();
        train_with_synthetic(&mut tc);

        assert!(tc.classify(&vec![1.0; FEATURE_DIM - 1]).is_none());
        assert!(tc.classify(&vec![1.0; FEATURE_DIM + 1]).is_none());
        assert!(tc.classify(&Vec::<f32>::new()).is_none());
    }

    #[test]
    fn train_rejects_invalid_phrase_index() {
        let mut tc = TransversalClassifier::new();
        let result = tc.train(&[(2, PHRASE_1[0], vec![0.0; FEATURE_DIM])]);
        assert!(result.is_err());
        match result.unwrap_err() {
            TrainError::InvalidPhraseIndex(idx) => assert_eq!(idx, 2),
            other => panic!("expected InvalidPhraseIndex, got {other:?}"),
        }
    }

    #[test]
    fn train_rejects_wrong_syllable_for_phrase() {
        let mut tc = TransversalClassifier::new();
        // PHRASE_2[0] = 'I, which is not in PHRASE_1
        let wrong_syllable = PHRASE_2[0];
        let result = tc.train(&[(0, wrong_syllable, vec![0.0; FEATURE_DIM])]);
        assert!(result.is_err());
        match result.unwrap_err() {
            TrainError::WrongSyllableForPhrase {
                syllable,
                phrase_index,
            } => {
                assert_eq!(syllable, wrong_syllable);
                assert_eq!(phrase_index, 0);
            }
            other => panic!("expected WrongSyllableForPhrase, got {other:?}"),
        }
    }
}
