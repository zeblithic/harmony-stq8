//! Flashcard engine for Q8 pronunciation training.
//!
//! Provides difficulty levels, challenge generation, and row-by-row validation
//! of spoken syllables against expected byte data.

use serde::{Deserialize, Serialize};

use crate::q8::Syllable;

/// Difficulty levels for flashcard challenges.
///
/// Each level defines a grid of bytes with specific dimensions.
/// The grid is arranged as `num_rows` rows of `bytes_per_row` bytes each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Level {
    /// 1x1 grid: 1 byte, 1 row of 1
    Novice,
    /// 1x2 grid: 2 bytes, 1 row of 2
    Apprentice,
    /// 2x2 grid: 4 bytes, 2 rows of 2
    Journeyman,
    /// 3x3 grid: 9 bytes, 3 rows of 3
    Expert,
    /// 8x4 grid: 32 bytes, 8 rows of 4
    Master,
}

impl Level {
    /// All levels in order of increasing difficulty.
    pub const ALL: [Level; 5] = [
        Level::Novice,
        Level::Apprentice,
        Level::Journeyman,
        Level::Expert,
        Level::Master,
    ];

    /// Total number of bytes in the challenge grid.
    pub fn total_bytes(self) -> usize {
        match self {
            Level::Novice => 1,
            Level::Apprentice => 2,
            Level::Journeyman => 4,
            Level::Expert => 9,
            Level::Master => 32,
        }
    }

    /// Number of bytes per row in the grid.
    pub fn bytes_per_row(self) -> usize {
        match self {
            Level::Novice => 1,
            Level::Apprentice => 2,
            Level::Journeyman => 2,
            Level::Expert => 3,
            Level::Master => 4,
        }
    }

    /// Number of rows in the grid.
    pub fn num_rows(self) -> usize {
        match self {
            Level::Novice => 1,
            Level::Apprentice => 1,
            Level::Journeyman => 2,
            Level::Expert => 3,
            Level::Master => 8,
        }
    }

    /// Total number of bits in the challenge (total_bytes * 8).
    pub fn total_bits(self) -> usize {
        self.total_bytes() * 8
    }
}

/// A flashcard challenge: a grid of random bytes split into rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Challenge {
    /// The difficulty level of this challenge.
    pub level: Level,
    /// The full random byte data (length = level.total_bytes()).
    pub data: Vec<u8>,
    /// The data split into rows according to the level's grid dimensions.
    pub rows: Vec<Vec<u8>>,
}

/// Generate a challenge from caller-provided random bytes (sans-I/O).
///
/// If `rng_bytes` is shorter than needed, the data is padded with zeros.
/// If longer, it is truncated to the required length.
pub fn generate(level: Level, rng_bytes: &[u8]) -> Challenge {
    let total = level.total_bytes();
    let bpr = level.bytes_per_row();

    let mut data = vec![0u8; total];
    let copy_len = rng_bytes.len().min(total);
    data[..copy_len].copy_from_slice(&rng_bytes[..copy_len]);

    let rows = data.chunks(bpr).map(|chunk| chunk.to_vec()).collect();

    Challenge { level, data, rows }
}

/// Result of validating a single row of heard syllables against expected bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowResult {
    /// Whether the heard syllables exactly match the expected syllables.
    pub matched: bool,
    /// The expected syllables derived from the expected bytes (2 per byte: high nibble, low nibble).
    pub expected: Vec<Syllable>,
    /// The syllables that were heard (from classification).
    pub heard: Vec<Syllable>,
}

/// Validate a row of heard syllables against expected bytes.
///
/// Each byte produces 2 syllables (high nibble first, low nibble second).
/// A match requires the same count and each position to match exactly.
pub fn validate_row(expected_bytes: &[u8], heard: &[Syllable]) -> RowResult {
    let expected: Vec<Syllable> = expected_bytes
        .iter()
        .flat_map(|&b| {
            let high = Syllable::from_nibble((b >> 4) & 0x0F);
            let low = Syllable::from_nibble(b & 0x0F);
            [high, low]
        })
        .collect();

    let matched =
        expected.len() == heard.len() && expected.iter().zip(heard.iter()).all(|(e, h)| e == h);

    RowResult {
        matched,
        expected,
        heard: heard.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Task 4: Level dimensions ---

    #[test]
    fn level_novice_dimensions() {
        assert_eq!(Level::Novice.total_bytes(), 1);
        assert_eq!(Level::Novice.bytes_per_row(), 1);
        assert_eq!(Level::Novice.num_rows(), 1);
        assert_eq!(Level::Novice.total_bits(), 8);
    }

    #[test]
    fn level_apprentice_dimensions() {
        assert_eq!(Level::Apprentice.total_bytes(), 2);
        assert_eq!(Level::Apprentice.bytes_per_row(), 2);
        assert_eq!(Level::Apprentice.num_rows(), 1);
        assert_eq!(Level::Apprentice.total_bits(), 16);
    }

    #[test]
    fn level_journeyman_dimensions() {
        assert_eq!(Level::Journeyman.total_bytes(), 4);
        assert_eq!(Level::Journeyman.bytes_per_row(), 2);
        assert_eq!(Level::Journeyman.num_rows(), 2);
        assert_eq!(Level::Journeyman.total_bits(), 32);
    }

    #[test]
    fn level_expert_dimensions() {
        assert_eq!(Level::Expert.total_bytes(), 9);
        assert_eq!(Level::Expert.bytes_per_row(), 3);
        assert_eq!(Level::Expert.num_rows(), 3);
        assert_eq!(Level::Expert.total_bits(), 72);
    }

    #[test]
    fn level_master_dimensions() {
        assert_eq!(Level::Master.total_bytes(), 32);
        assert_eq!(Level::Master.bytes_per_row(), 4);
        assert_eq!(Level::Master.num_rows(), 8);
        assert_eq!(Level::Master.total_bits(), 256);
    }

    #[test]
    fn all_levels_rows_times_bytes_per_row_equals_total() {
        for level in Level::ALL {
            assert_eq!(
                level.num_rows() * level.bytes_per_row(),
                level.total_bytes(),
                "{level:?}: num_rows * bytes_per_row should equal total_bytes"
            );
        }
    }

    // --- Task 5: Challenge generation ---

    #[test]
    fn generate_novice_one_byte() {
        let challenge = generate(Level::Novice, &[0x42]);
        assert_eq!(challenge.level, Level::Novice);
        assert_eq!(challenge.data, vec![0x42]);
        assert_eq!(challenge.rows, vec![vec![0x42]]);
    }

    #[test]
    fn generate_master_32_bytes() {
        let rng: Vec<u8> = (0..32).collect();
        let challenge = generate(Level::Master, &rng);
        assert_eq!(challenge.data.len(), 32);
        assert_eq!(challenge.rows.len(), 8);
        for row in &challenge.rows {
            assert_eq!(row.len(), 4);
        }
        let flat: Vec<u8> = challenge.rows.iter().flatten().copied().collect();
        assert_eq!(flat, challenge.data);
    }

    #[test]
    fn generate_truncates_excess_rng() {
        let challenge = generate(Level::Novice, &vec![0xAB; 100]);
        assert_eq!(challenge.data, vec![0xAB]);
    }

    #[test]
    fn generate_pads_insufficient_rng() {
        let challenge = generate(Level::Apprentice, &[]);
        assert_eq!(challenge.data.len(), 2);
    }

    #[test]
    fn generate_expert_nine_bytes_three_rows() {
        let rng: Vec<u8> = (10..19).collect();
        let challenge = generate(Level::Expert, &rng);
        assert_eq!(challenge.rows[0], vec![10, 11, 12]);
        assert_eq!(challenge.rows[1], vec![13, 14, 15]);
        assert_eq!(challenge.rows[2], vec![16, 17, 18]);
    }

    // --- Task 6: Row validation ---

    #[test]
    fn validate_row_exact_match() {
        let s0 = Syllable::from_nibble(0);
        let result = validate_row(&[0x00], &[s0, s0]);
        assert!(result.matched);
        assert_eq!(result.expected, vec![s0, s0]);
        assert_eq!(result.heard, vec![s0, s0]);
    }

    #[test]
    fn validate_row_mismatch() {
        let s0 = Syllable::from_nibble(0);
        let s1 = Syllable::from_nibble(1);
        let result = validate_row(&[0x00], &[s0, s1]);
        assert!(!result.matched);
        assert_eq!(result.expected, vec![s0, s0]);
        assert_eq!(result.heard, vec![s0, s1]);
    }

    #[test]
    fn validate_row_multi_byte() {
        let syllables: Vec<Syllable> = [5, 9, 10, 3]
            .iter()
            .map(|&n| Syllable::from_nibble(n))
            .collect();
        let result = validate_row(&[0x59, 0xA3], &syllables);
        assert!(result.matched);
        assert_eq!(result.heard.len(), 4);
    }

    #[test]
    fn validate_row_wrong_length_too_few() {
        let result = validate_row(&[0x00], &[Syllable::from_nibble(0)]);
        assert!(!result.matched);
    }

    #[test]
    fn validate_row_wrong_length_too_many() {
        let s0 = Syllable::from_nibble(0);
        let result = validate_row(&[0x00], &[s0, s0, s0]);
        assert!(!result.matched);
    }

    #[test]
    fn validate_row_empty_expected() {
        let result = validate_row(&[], &[]);
        assert!(result.matched);
        assert!(result.expected.is_empty());
    }
}
