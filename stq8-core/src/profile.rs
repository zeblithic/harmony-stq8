//! User profile: trained classifier state, confidence thresholds, phoneme remapping.
//!
//! A [`UserProfile`] stores the centroids from a trained [`NearestCentroid`] classifier
//! along with [`Thresholds`] for the three-tier confidence model (accept/suggest/reject)
//! and an optional custom phoneme remapping for accessibility.
//!
//! Profiles serialize to JSON (~2KB) for portability across devices.

use crate::q8::Phoneme;
use serde::{Deserialize, Serialize};

/// Confidence thresholds for the three-tier model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    /// Above this: auto-accept (default 0.85)
    pub auto_accept: f32,
    /// Above this but below auto_accept: show suggestion (default 0.60)
    pub suggest: f32,
    /// Below this: reject entirely (default 0.40)
    pub reject: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            auto_accept: 0.85,
            suggest: 0.60,
            reject: 0.40,
        }
    }
}

/// What to do with a classification result.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Accept(Phoneme, f32),
    Suggest(Phoneme, f32),
    Reject(f32),
}

impl Thresholds {
    /// Decide what to do with a classification result based on confidence.
    ///
    /// - confidence >= auto_accept -> Accept
    /// - confidence >= suggest -> Suggest
    /// - otherwise -> Reject
    pub fn decide(&self, phoneme: Phoneme, confidence: f32) -> Decision {
        if confidence >= self.auto_accept {
            Decision::Accept(phoneme, confidence)
        } else if confidence >= self.suggest {
            Decision::Suggest(phoneme, confidence)
        } else {
            Decision::Reject(confidence)
        }
    }
}

/// A user's STQ8 profile: trained centroids + thresholds + remapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub version: u8,
    pub centroids: Vec<(Phoneme, Vec<f32>)>,
    pub thresholds: Thresholds,
    pub custom_map: Vec<(Phoneme, Phoneme)>,
    pub created_epoch_secs: u64,
}

impl UserProfile {
    /// Apply the custom phoneme remapping.
    ///
    /// Looks up `phoneme` in `custom_map`; returns the mapped value if found,
    /// otherwise returns the input unchanged.
    pub fn apply_remap(&self, phoneme: Phoneme) -> Phoneme {
        for (from, to) in &self.custom_map {
            if *from == phoneme {
                return *to;
            }
        }
        phoneme
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mfcc::FEATURE_DIM;
    use crate::q8::{Consonant, Vowel};

    /// Helper: build a minimal UserProfile for testing.
    fn sample_profile() -> UserProfile {
        let glottal = Phoneme::Consonant(Consonant::GlottalStop);
        let j = Phoneme::Consonant(Consonant::J);
        let k = Phoneme::Consonant(Consonant::K);

        UserProfile {
            version: 1,
            centroids: vec![
                (glottal, vec![0.0; FEATURE_DIM]),
                (j, vec![1.0; FEATURE_DIM]),
                (k, vec![0.5; FEATURE_DIM]),
            ],
            thresholds: Thresholds::default(),
            custom_map: vec![(glottal, j)],
            created_epoch_secs: 1_710_000_000,
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let profile = sample_profile();
        let json = serde_json::to_string(&profile).expect("serialize");
        let restored: UserProfile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.version, profile.version);
        assert_eq!(restored.centroids.len(), profile.centroids.len());
        for (orig, rest) in profile.centroids.iter().zip(restored.centroids.iter()) {
            assert_eq!(orig.0, rest.0);
            assert_eq!(orig.1, rest.1);
        }
        assert!(
            (restored.thresholds.auto_accept - profile.thresholds.auto_accept).abs() < f32::EPSILON
        );
        assert!((restored.thresholds.suggest - profile.thresholds.suggest).abs() < f32::EPSILON);
        assert!((restored.thresholds.reject - profile.thresholds.reject).abs() < f32::EPSILON);
        assert_eq!(restored.custom_map, profile.custom_map);
        assert_eq!(restored.created_epoch_secs, profile.created_epoch_secs);
    }

    #[test]
    fn default_thresholds() {
        let t = Thresholds::default();
        assert!((t.auto_accept - 0.85).abs() < f32::EPSILON);
        assert!((t.suggest - 0.60).abs() < f32::EPSILON);
        assert!((t.reject - 0.40).abs() < f32::EPSILON);
    }

    #[test]
    fn decide_auto_accept() {
        let t = Thresholds::default();
        let phoneme = Phoneme::Vowel(Vowel::O);
        let decision = t.decide(phoneme, 0.92);
        assert_eq!(decision, Decision::Accept(phoneme, 0.92));
    }

    #[test]
    fn decide_suggest() {
        let t = Thresholds::default();
        let phoneme = Phoneme::Consonant(Consonant::K);
        let decision = t.decide(phoneme, 0.72);
        assert_eq!(decision, Decision::Suggest(phoneme, 0.72));
    }

    #[test]
    fn decide_reject() {
        let t = Thresholds::default();
        let phoneme = Phoneme::Consonant(Consonant::V);
        let decision = t.decide(phoneme, 0.30);
        assert_eq!(decision, Decision::Reject(0.30));
    }

    #[test]
    fn custom_map_applies() {
        let profile = sample_profile();
        let glottal = Phoneme::Consonant(Consonant::GlottalStop);
        let j = Phoneme::Consonant(Consonant::J);

        let result = profile.apply_remap(glottal);
        assert_eq!(result, j, "GlottalStop should remap to J");
    }

    #[test]
    fn custom_map_passthrough() {
        let profile = sample_profile();
        let vowel_e = Phoneme::Vowel(Vowel::E);

        let result = profile.apply_remap(vowel_e);
        assert_eq!(
            result, vowel_e,
            "unmapped phoneme should pass through unchanged"
        );
    }
}
