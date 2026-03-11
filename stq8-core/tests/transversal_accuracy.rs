//! Accuracy comparison benchmarks: NearestCentroid (16-sound) vs TransversalClassifier (8-sound).
//!
//! Trains both classifiers on the same synthetic feature data and compares
//! accuracy, confidence, and noise robustness.

use stq8_core::classifier::{Classifier, NearestCentroid};
use stq8_core::mfcc::FEATURE_DIM;
use stq8_core::q8::Syllable;
use stq8_core::transversal::TransversalClassifier;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build features using a consonant+vowel decomposition.
///
/// Dimensions 0..23 encode the consonant (strong activation for matching consonant,
/// small noise for others). Dimensions 26..49 encode the vowel (same scheme).
///
/// This ensures cosine similarity reflects consonant AND vowel overlap, making
/// both the NearestCentroid and the TransversalClassifier's Top-2 intersection
/// algorithm work correctly on all 16 nibbles.
fn make_features(syllable_idx: usize) -> Vec<f32> {
    let nibble = syllable_idx as u8;
    let consonant_bits = (nibble >> 2) & 0x03;
    let vowel_bits = nibble & 0x03;

    let mut v = vec![0.0f32; FEATURE_DIM];
    // Consonant region: dims 0..23 (4 consonants x 6 dims)
    for c in 0..4u8 {
        let c_base = (c as usize) * 6;
        if c == consonant_bits {
            v[c_base] = 1.0;
            v[c_base + 1] = 0.7;
            v[c_base + 2] = 0.5;
        } else {
            let noise = 0.01 * (c as f32 + 1.0);
            v[c_base] = noise;
        }
    }
    // Vowel region: dims 26..49 (4 vowels x 6 dims)
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

/// Add deterministic pseudo-random noise to a feature vector.
fn add_noise(features: &[f32], noise_level: f32, seed: u32) -> Vec<f32> {
    features
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            // LCG-style hash, truncated to u32 for bounded output
            let hash = ((seed as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64)) as u32;
            let noise = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;
            v + noise * noise_level
        })
        .collect()
}

/// Train a full 16-centroid NearestCentroid classifier.
fn train_full() -> NearestCentroid {
    let mut nc = NearestCentroid::new();
    let samples: Vec<(Syllable, Vec<f32>)> = (0..16)
        .map(|i| (Syllable::from_nibble(i), make_features(i as usize)))
        .collect();
    nc.train(&samples);
    nc
}

/// Train a transversal classifier using the two orthogonal phrases (8 sounds).
fn train_transversal() -> TransversalClassifier {
    let mut tc = TransversalClassifier::new();
    // Phrase 1 (diagonal):   'O(0), JU(5), KE(10), VI(15)
    let phrase1_indices: [u8; 4] = [0, 5, 10, 15];
    // Phrase 2 (anti-diag):  'I(3), JE(6), KO(8), VU(13)
    let phrase2_indices: [u8; 4] = [3, 6, 8, 13];

    let mut samples = Vec::new();
    for &idx in &phrase1_indices {
        samples.push((0u8, Syllable::from_nibble(idx), make_features(idx as usize)));
    }
    for &idx in &phrase2_indices {
        samples.push((1u8, Syllable::from_nibble(idx), make_features(idx as usize)));
    }
    tc.train(&samples)
        .expect("transversal training should succeed");
    tc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn clean_accuracy_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    let mut nc_correct = 0u32;
    let mut tc_correct = 0u32;

    println!("\n=== Clean Accuracy Comparison ===");
    println!("{:<8} {:<12} {:<12}", "Nibble", "Full(16)", "Transversal(8)");
    println!("{:-<34}", "");

    for nibble in 0..16u8 {
        let target = Syllable::from_nibble(nibble);
        let features = make_features(nibble as usize);

        let nc_result = nc.classify(&features);
        let tc_result = tc.classify(&features);

        let nc_ok = nc_result
            .as_ref()
            .map_or(false, |r| r.syllable == target);
        let tc_ok = tc_result
            .as_ref()
            .map_or(false, |r| r.syllable == target);

        if nc_ok {
            nc_correct += 1;
        }
        if tc_ok {
            tc_correct += 1;
        }

        println!(
            "{:<8} {:<12} {:<12}",
            format!("{nibble:2} ({target})"),
            if nc_ok { "OK" } else { "FAIL" },
            if tc_ok { "OK" } else { "FAIL" },
        );
    }

    let nc_acc = nc_correct as f32 / 16.0 * 100.0;
    let tc_acc = tc_correct as f32 / 16.0 * 100.0;
    println!("{:-<34}", "");
    println!("Full accuracy:        {nc_acc:.1}% ({nc_correct}/16)");
    println!("Transversal accuracy: {tc_acc:.1}% ({tc_correct}/16)");

    assert_eq!(nc_correct, 16, "NearestCentroid should be 100% on clean data");
    assert_eq!(tc_correct, 16, "TransversalClassifier should be 100% on clean data");
}

#[test]
fn noisy_accuracy_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    let noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30];
    let trials_per_nibble = 50u32;

    println!("\n=== Noisy Accuracy Comparison ===");
    println!("  ({trials_per_nibble} trials per nibble per noise level)");
    println!("{:<10} {:<15} {:<15}", "Noise", "Full(16)", "Transversal(8)");
    println!("{:-<42}", "");

    for &noise_level in &noise_levels {
        let mut nc_correct = 0u32;
        let mut tc_correct = 0u32;
        let total = 16 * trials_per_nibble;

        for nibble in 0..16u8 {
            let target = Syllable::from_nibble(nibble);
            let clean = make_features(nibble as usize);

            for trial in 0..trials_per_nibble {
                let seed = (nibble as u32) * 1000 + trial;
                let noisy = add_noise(&clean, noise_level, seed);

                if let Some(r) = nc.classify(&noisy) {
                    if r.syllable == target {
                        nc_correct += 1;
                    }
                }
                if let Some(r) = tc.classify(&noisy) {
                    if r.syllable == target {
                        tc_correct += 1;
                    }
                }
            }
        }

        let nc_acc = nc_correct as f32 / total as f32 * 100.0;
        let tc_acc = tc_correct as f32 / total as f32 * 100.0;
        println!(
            "{:<10} {:<15} {:<15}",
            format!("{noise_level:.2}"),
            format!("{nc_acc:.1}% ({nc_correct}/{total})"),
            format!("{tc_acc:.1}% ({tc_correct}/{total})"),
        );
    }

    // No hard assertions -- noise results vary by feature geometry.
    println!("\n  (no assertions -- noise robustness is informational)");
}

#[test]
fn confidence_distribution_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    println!("\n=== Confidence Distribution Comparison ===");
    println!(
        "{:<8} {:<18} {:<18}",
        "Nibble", "Full conf", "Transversal conf"
    );
    println!("{:-<46}", "");

    let mut nc_sum = 0.0f32;
    let mut tc_sum = 0.0f32;
    let mut nc_count = 0u32;
    let mut tc_count = 0u32;

    for nibble in 0..16u8 {
        let target = Syllable::from_nibble(nibble);
        let features = make_features(nibble as usize);

        let nc_conf = nc
            .classify(&features)
            .map(|r| {
                nc_sum += r.confidence;
                nc_count += 1;
                format!("{:.4}", r.confidence)
            })
            .unwrap_or_else(|| "N/A".to_string());

        let tc_conf = tc
            .classify(&features)
            .map(|r| {
                tc_sum += r.confidence;
                tc_count += 1;
                format!("{:.4}", r.confidence)
            })
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<8} {:<18} {:<18}",
            format!("{nibble:2} ({target})"),
            nc_conf,
            tc_conf,
        );
    }

    println!("{:-<46}", "");
    if nc_count > 0 {
        println!("Full mean confidence:        {:.4}", nc_sum / nc_count as f32);
    }
    if tc_count > 0 {
        println!("Transversal mean confidence: {:.4}", tc_sum / tc_count as f32);
    }

    // No hard assertions -- confidence distribution is diagnostic.
    println!("\n  (no assertions -- confidence distribution is informational)");
}

#[test]
fn inference_cost_comparison() {
    let nc = train_full();
    let tc = train_transversal();

    // Pick an arbitrary nibble (KU = 9) to verify correctness
    let nibble = 9u8;
    let target = Syllable::from_nibble(nibble);
    let features = make_features(nibble as usize);

    let nc_result = nc.classify(&features).expect("NearestCentroid should classify");
    let tc_result = tc.classify(&features).expect("TransversalClassifier should classify");

    assert_eq!(nc_result.syllable, target, "NearestCentroid should get KU correct");
    assert_eq!(tc_result.syllable, target, "TransversalClassifier should get KU correct");

    println!("\n=== Inference Cost Comparison ===");
    println!("Target: nibble {nibble} ({target})");
    println!();
    println!("NearestCentroid (Full 16-sound):");
    println!("  - Cosine similarity ops: 16");
    println!("  - Result: {} (confidence {:.4})", nc_result.syllable, nc_result.confidence);
    println!();
    println!("TransversalClassifier (8-sound):");
    println!("  - Cosine similarity ops: 8  (4 per phrase x 2 phrases)");
    println!("  - Top-2 intersection:    2  (consonant + vowel set ops)");
    println!("  - Result: {} (confidence {:.4})", tc_result.syllable, tc_result.confidence);
    println!();
    println!("Cost reduction: 50% fewer cosine similarity operations");
    println!("Training reduction: 50% fewer syllables to record (8 vs 16)");
}
