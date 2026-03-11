//! User profile: trained classifier state, confidence thresholds, phoneme remapping.
//!
//! A [`UserProfile`] stores the centroids from a trained [`NearestCentroid`] classifier
//! along with [`Thresholds`] for the three-tier confidence model (accept/suggest/reject)
//! and an optional custom phoneme remapping for accessibility.
//!
//! Profiles serialize to JSON (~2KB) for portability across devices.

use crate::q8::{Phoneme, Syllable};
use serde::{Deserialize, Serialize};

/// Which calibration strategy was used to train this profile.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMode {
    /// 16-sound: all CV pairs calibrated directly
    #[default]
    Full,
    /// 8-sound: two orthogonal transversal phrases, reconstruct via intersection
    Transversal,
}

/// Confidence thresholds for the three-tier model.
///
/// The suggest zone is implicitly defined as [reject, auto_accept).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    /// Above this: auto-accept (default 0.85)
    pub auto_accept: f32,
    /// Below this: reject entirely (default 0.40)
    pub reject: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            auto_accept: 0.85,
            reject: 0.40,
        }
    }
}

/// What to do with a classification result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Decision {
    Accept(Syllable, f32),
    Suggest(Syllable, f32),
    Reject(f32),
}

impl Thresholds {
    /// Decide what to do with a classification result based on confidence.
    ///
    /// - confidence >= auto_accept -> Accept
    /// - confidence >= reject -> Suggest (show "did you mean?")
    /// - confidence < reject -> Reject (too ambiguous, no suggestion)
    pub fn decide(&self, syllable: Syllable, confidence: f32) -> Decision {
        if confidence >= self.auto_accept {
            Decision::Accept(syllable, confidence)
        } else if confidence >= self.reject {
            Decision::Suggest(syllable, confidence)
        } else {
            Decision::Reject(confidence)
        }
    }
}

/// A user's STQ8 profile: trained centroids + thresholds + remapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub version: u8,
    pub centroids: Vec<(Syllable, Vec<f32>)>,
    pub thresholds: Thresholds,
    /// Phoneme-level remapping for accessibility.
    ///
    /// Applied to syllable components: if a consonant or vowel matches
    /// the `from` phoneme, it's replaced with the `to` phoneme.
    pub custom_map: Vec<(Phoneme, Phoneme)>,
    #[serde(default)]
    pub calibration_mode: CalibrationMode,
    pub created_epoch_secs: u64,
}

impl UserProfile {
    /// Apply the custom phoneme remapping to a syllable's components.
    ///
    /// Looks up the syllable's consonant and vowel in `custom_map`;
    /// replaces matching components. Non-matching components pass through.
    pub fn apply_remap(&self, syllable: Syllable) -> Syllable {
        let orig_consonant = syllable.consonant;
        let orig_vowel = syllable.vowel;
        let mut consonant = syllable.consonant;
        let mut vowel = syllable.vowel;
        let mut consonant_remapped = false;
        let mut vowel_remapped = false;
        for (from, to) in &self.custom_map {
            match (from, to) {
                (Phoneme::Consonant(f), Phoneme::Consonant(t))
                    if *f == orig_consonant && !consonant_remapped =>
                {
                    consonant = *t;
                    consonant_remapped = true;
                }
                (Phoneme::Vowel(f), Phoneme::Vowel(t)) if *f == orig_vowel && !vowel_remapped => {
                    vowel = *t;
                    vowel_remapped = true;
                }
                // Cross-type rules (consonant→vowel, vowel→consonant) are skipped.
                _ => {}
            }
        }
        Syllable::new(consonant, vowel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mfcc::FEATURE_DIM;
    use crate::q8::{Consonant, Vowel};

    /// Helper: build a minimal UserProfile for testing.
    fn sample_profile() -> UserProfile {
        let glottal_o = Syllable::new(Consonant::GlottalStop, Vowel::O);
        let jo = Syllable::new(Consonant::J, Vowel::O);
        let ku = Syllable::new(Consonant::K, Vowel::U);

        UserProfile {
            version: 1,
            centroids: vec![
                (glottal_o, vec![0.0; FEATURE_DIM]),
                (jo, vec![1.0; FEATURE_DIM]),
                (ku, vec![0.5; FEATURE_DIM]),
            ],
            thresholds: Thresholds::default(),
            // Remap glottal stop consonant → J consonant
            custom_map: vec![(
                Phoneme::Consonant(Consonant::GlottalStop),
                Phoneme::Consonant(Consonant::J),
            )],
            calibration_mode: CalibrationMode::default(),
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
        assert!((restored.thresholds.reject - profile.thresholds.reject).abs() < f32::EPSILON);
        assert_eq!(restored.custom_map, profile.custom_map);
        assert_eq!(restored.created_epoch_secs, profile.created_epoch_secs);
    }

    #[test]
    fn default_thresholds() {
        let t = Thresholds::default();
        assert!((t.auto_accept - 0.85).abs() < f32::EPSILON);
        assert!((t.reject - 0.40).abs() < f32::EPSILON);
    }

    #[test]
    fn decide_auto_accept() {
        let t = Thresholds::default();
        let syllable = Syllable::new(Consonant::K, Vowel::O);
        let decision = t.decide(syllable, 0.92);
        assert_eq!(decision, Decision::Accept(syllable, 0.92));
    }

    #[test]
    fn decide_suggest() {
        let t = Thresholds::default();
        let syllable = Syllable::new(Consonant::K, Vowel::U);
        let decision = t.decide(syllable, 0.72);
        assert_eq!(decision, Decision::Suggest(syllable, 0.72));
    }

    #[test]
    fn decide_reject() {
        let t = Thresholds::default();
        let syllable = Syllable::new(Consonant::V, Vowel::E);
        let decision = t.decide(syllable, 0.30);
        assert_eq!(decision, Decision::Reject(0.30));
    }

    #[test]
    fn custom_map_remaps_consonant() {
        let profile = sample_profile();
        // GlottalStop consonant should be remapped to J
        let glottal_o = Syllable::new(Consonant::GlottalStop, Vowel::O);
        let j_o = Syllable::new(Consonant::J, Vowel::O);

        let result = profile.apply_remap(glottal_o);
        assert_eq!(result, j_o, "GlottalStop consonant should remap to J");
    }

    #[test]
    fn custom_map_preserves_vowel() {
        let profile = sample_profile();
        // The vowel should pass through since only the consonant is remapped
        let glottal_i = Syllable::new(Consonant::GlottalStop, Vowel::I);
        let j_i = Syllable::new(Consonant::J, Vowel::I);

        let result = profile.apply_remap(glottal_i);
        assert_eq!(
            result, j_i,
            "consonant remap should apply regardless of vowel"
        );
    }

    #[test]
    fn custom_map_passthrough() {
        let profile = sample_profile();
        // KU has no remap rule — should pass through unchanged
        let ku = Syllable::new(Consonant::K, Vowel::U);

        let result = profile.apply_remap(ku);
        assert_eq!(
            result, ku,
            "unmapped syllable should pass through unchanged"
        );
    }

    #[test]
    fn custom_map_no_chaining() {
        // Rules [(GlottalStop→J), (J→K)] should NOT chain:
        // GlottalStop should map to J, not K.
        let profile = UserProfile {
            version: 1,
            centroids: Vec::new(),
            thresholds: Thresholds::default(),
            custom_map: vec![
                (
                    Phoneme::Consonant(Consonant::GlottalStop),
                    Phoneme::Consonant(Consonant::J),
                ),
                (
                    Phoneme::Consonant(Consonant::J),
                    Phoneme::Consonant(Consonant::K),
                ),
            ],
            calibration_mode: CalibrationMode::default(),
            created_epoch_secs: 0,
        };

        let input = Syllable::new(Consonant::GlottalStop, Vowel::O);
        let result = profile.apply_remap(input);
        assert_eq!(
            result,
            Syllable::new(Consonant::J, Vowel::O),
            "remap should apply once, not chain through consecutive rules"
        );
    }

    #[test]
    fn custom_map_duplicate_from_uses_first() {
        // Two rules for the same source: first match wins.
        let profile = UserProfile {
            version: 1,
            centroids: Vec::new(),
            thresholds: Thresholds::default(),
            custom_map: vec![
                (
                    Phoneme::Consonant(Consonant::GlottalStop),
                    Phoneme::Consonant(Consonant::J),
                ),
                (
                    Phoneme::Consonant(Consonant::GlottalStop),
                    Phoneme::Consonant(Consonant::K),
                ),
            ],
            calibration_mode: CalibrationMode::default(),
            created_epoch_secs: 0,
        };

        let input = Syllable::new(Consonant::GlottalStop, Vowel::O);
        let result = profile.apply_remap(input);
        assert_eq!(
            result,
            Syllable::new(Consonant::J, Vowel::O),
            "duplicate from rules should use the first match, not last-write-wins"
        );
    }

    #[test]
    fn custom_map_cross_type_ignored() {
        // A consonant→vowel rule should have no effect.
        let profile = UserProfile {
            version: 1,
            centroids: Vec::new(),
            thresholds: Thresholds::default(),
            custom_map: vec![(
                Phoneme::Consonant(Consonant::GlottalStop),
                Phoneme::Vowel(Vowel::I),
            )],
            calibration_mode: CalibrationMode::default(),
            created_epoch_secs: 0,
        };

        let input = Syllable::new(Consonant::GlottalStop, Vowel::O);
        let result = profile.apply_remap(input);
        assert_eq!(
            result, input,
            "cross-type remap rules should be silently skipped"
        );
    }

    #[test]
    fn calibration_mode_default_is_full() {
        assert_eq!(CalibrationMode::default(), CalibrationMode::Full);
    }

    #[test]
    fn calibration_mode_serialization_roundtrip() {
        let mut profile = sample_profile();
        profile.calibration_mode = CalibrationMode::Transversal;

        let json = serde_json::to_string(&profile).expect("serialize");
        let restored: UserProfile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(
            restored.calibration_mode,
            CalibrationMode::Transversal,
            "calibration_mode should survive serialization roundtrip"
        );
    }

    #[test]
    fn calibration_mode_defaults_to_full_when_missing_in_json() {
        // JSON without the calibration_mode field — should deserialize as Full
        let json = r#"{
            "version": 1,
            "centroids": [],
            "thresholds": {"auto_accept": 0.85, "reject": 0.40},
            "custom_map": [],
            "created_epoch_secs": 0
        }"#;

        let profile: UserProfile = serde_json::from_str(json).expect("deserialize");
        assert_eq!(
            profile.calibration_mode,
            CalibrationMode::Full,
            "missing calibration_mode should default to Full for backward compat"
        );
    }
}
