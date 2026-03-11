# Q8 Flashcard System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add challenge generation, row validation, and Q8-BOX/FLAT formatting to stq8-core for the flashcard practice system.

**Architecture:** Challenge engine and formatting in Rust (stq8-core), exposed via WASM bindings (stq8-web). Session state and UI will live in harmony-client (separate repo, separate plan).

**Tech Stack:** Rust, wasm-bindgen, serde, serde_json

**Design doc:** `docs/plans/2026-03-11-flashcard-design.md`

---

### Task 1: Q8-BOX Formatting Constants

Add BOX-format character arrays alongside the existing FLAT-format arrays in `q8.rs`.

**Files:**
- Modify: `stq8-core/src/q8.rs:153-156` (near existing CONSONANT_CHARS/VOWEL_CHARS)
- Test: `stq8-core/src/q8.rs` (inline tests module)

**Context:** The existing `q8.rs` has two const arrays at line 153-156:
```rust
const CONSONANT_CHARS: [char; 4] = ['\'', 'J', 'K', 'V'];
const VOWEL_CHARS: [char; 4] = ['O', 'U', 'E', 'I'];
```
These are the Q8-FLAT (phonetic) characters. We need a parallel set for Q8-BOX (visual grid) display:
- BOX consonants: `A` (glottal), `>` (J), `<` (K), `V` (V)
- BOX vowels: `O` (O), `=` (U), `X` (E), `I` (I)

**Step 1: Write the failing tests**

Add these tests to the `#[cfg(test)] mod tests` block at the bottom of `q8.rs`:

```rust
// --- Q8-BOX format ---

#[test]
fn box_consonant_chars_match_bit_order() {
    // 00=A(glottal), 01=>(J), 10=<(K), 11=V
    assert_eq!(BOX_CONSONANT_CHARS[0], 'A');
    assert_eq!(BOX_CONSONANT_CHARS[1], '>');
    assert_eq!(BOX_CONSONANT_CHARS[2], '<');
    assert_eq!(BOX_CONSONANT_CHARS[3], 'V');
}

#[test]
fn box_vowel_chars_match_bit_order() {
    // 00=O, 01==(U), 10=X(E), 11=I
    assert_eq!(BOX_VOWEL_CHARS[0], 'O');
    assert_eq!(BOX_VOWEL_CHARS[1], '=');
    assert_eq!(BOX_VOWEL_CHARS[2], 'X');
    assert_eq!(BOX_VOWEL_CHARS[3], 'I');
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core box_consonant_chars -- --nocapture`
Expected: FAIL — `BOX_CONSONANT_CHARS` not found

**Step 3: Add the constants**

Add directly after the existing `VOWEL_CHARS` constant (around line 156):

```rust
/// BOX-format consonant characters: visual symbols for grid display.
/// Bit order: 00=A (glottal stop), 01=> (J), 10=< (K), 11=V.
const BOX_CONSONANT_CHARS: [char; 4] = ['A', '>', '<', 'V'];
/// BOX-format vowel characters: visual symbols for grid display.
/// Bit order: 00=O, 01== (U), 10=X (E), 11=I.
const BOX_VOWEL_CHARS: [char; 4] = ['O', '=', 'X', 'I'];
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core box_ -- --nocapture`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add stq8-core/src/q8.rs
git commit -m "feat(q8): add BOX-format character constants"
```

---

### Task 2: `format_box()` Function

Renders bytes as a Q8-BOX split grid: consonant row on top, vowel row on bottom.

**Files:**
- Modify: `stq8-core/src/q8.rs`
- Test: `stq8-core/src/q8.rs` (inline tests)

**Context:** For each row of bytes, output two lines:
- Line 1 (consonants): For each byte, the high nibble's BOX consonant + low nibble's BOX consonant. Bytes separated by spaces.
- Line 2 (vowels): Same layout with BOX vowels.

Example: bytes `[0x92, 0x03]` = nibbles `[9,2, 0,3]`:
- Nibble 9 = consonant bits `10`=`<`, vowel bits `01`=`=`
- Nibble 2 = consonant bits `00`=`A`, vowel bits `10`=`X`
- Nibble 0 = consonant bits `00`=`A`, vowel bits `00`=`O`
- Nibble 3 = consonant bits `00`=`A`, vowel bits `11`=`I`
- Consonant line: `<A AA` (byte1 high+low consonants, space, byte2 high+low consonants)
- Vowel line: `=X OI`

Row pairs are separated by a blank line.

**Step 1: Write the failing tests**

```rust
#[test]
fn format_box_single_byte() {
    // Byte 0x00 = nibbles 0,0 = ('O,'O) in flat
    // BOX: consonants A+A, vowels O+O
    let result = format_box(&[0x00], 1);
    assert_eq!(result, "AA\nOO");
}

#[test]
fn format_box_two_bytes_one_row() {
    // Bytes [0x92, 0x03]
    // 0x92 = nibbles 9,2: consonants <,A  vowels =,X
    // 0x03 = nibbles 0,3: consonants A,A  vowels O,I
    let result = format_box(&[0x92, 0x03], 2);
    assert_eq!(result, "<A AA\n=X OI");
}

#[test]
fn format_box_four_bytes_two_rows() {
    // 4 bytes, 2 per row → 2 row-pairs separated by blank line
    let result = format_box(&[0x00, 0xFF, 0x92, 0x03], 2);
    let expected = "AA VV\nOO II\n\n<A AA\n=X OI";
    assert_eq!(result, expected);
}

#[test]
fn format_box_empty() {
    assert_eq!(format_box(&[], 4), "");
}

#[test]
fn format_box_all_nibbles() {
    // Byte 0xFF = nibbles 15,15 = V,V consonants and I,I vowels
    let result = format_box(&[0xFF], 1);
    assert_eq!(result, "VV\nII");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core format_box -- --nocapture`
Expected: FAIL — `format_box` not found

**Step 3: Write the implementation**

Add this public function in `q8.rs` (after the existing `encode` function, before the tests module):

```rust
/// Render bytes in Q8-BOX format: consonant row on top, vowel row on bottom.
///
/// Each byte produces two characters per line (one per nibble). Bytes within
/// a row are space-separated. Row pairs are separated by blank lines.
///
/// Uses the BOX character set: consonants `A > < V`, vowels `O = X I`.
pub fn format_box(data: &[u8], bytes_per_row: usize) -> String {
    if data.is_empty() || bytes_per_row == 0 {
        return String::new();
    }

    let mut rows = Vec::new();
    for chunk in data.chunks(bytes_per_row) {
        let mut cons = Vec::new();
        let mut vows = Vec::new();
        for &b in chunk {
            let high = (b >> 4) & 0x0F;
            let low = b & 0x0F;
            cons.push(format!(
                "{}{}",
                BOX_CONSONANT_CHARS[((high >> 2) & 0x3) as usize],
                BOX_CONSONANT_CHARS[((low >> 2) & 0x3) as usize],
            ));
            vows.push(format!(
                "{}{}",
                BOX_VOWEL_CHARS[(high & 0x3) as usize],
                BOX_VOWEL_CHARS[(low & 0x3) as usize],
            ));
        }
        rows.push(format!("{}\n{}", cons.join(" "), vows.join(" ")));
    }
    rows.join("\n\n")
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core format_box -- --nocapture`
Expected: 5 tests PASS

**Step 5: Run full test suite**

Run: `cargo test -p stq8-core`
Expected: All existing + new tests pass

**Step 6: Commit**

```bash
git add stq8-core/src/q8.rs
git commit -m "feat(q8): add format_box() for Q8-BOX grid display"
```

---

### Task 3: `format_flat()` Function

Explicit Q8-FLAT formatting function. The existing `encode()` already uses the flat character set, but wraps at 8 words per line. `format_flat()` wraps at a caller-specified bytes-per-row to match the challenge grid layout.

**Files:**
- Modify: `stq8-core/src/q8.rs`
- Test: `stq8-core/src/q8.rs` (inline tests)

**Step 1: Write the failing tests**

```rust
#[test]
fn format_flat_single_byte() {
    // Byte 0x92 = KU'E in flat
    assert_eq!(format_flat(&[0x92], 1), "KU'E");
}

#[test]
fn format_flat_two_bytes() {
    assert_eq!(format_flat(&[0x92, 0x03], 2), "KU'E 'O'I");
}

#[test]
fn format_flat_wraps_at_bytes_per_row() {
    // 4 bytes, 2 per row
    let result = format_flat(&[0x00, 0xFF, 0x92, 0x03], 2);
    assert_eq!(result, "'O'O VIVI\nKU'E 'O'I");
}

#[test]
fn format_flat_empty() {
    assert_eq!(format_flat(&[], 4), "");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core format_flat -- --nocapture`
Expected: FAIL — `format_flat` not found

**Step 3: Write the implementation**

```rust
/// Render bytes in Q8-FLAT format: phonetic 4-char words, wrapped at `bytes_per_row`.
///
/// Uses the FLAT character set: consonants `' J K V`, vowels `O U E I`.
/// Each byte becomes one 4-character word. Words are space-separated.
/// Lines break at `bytes_per_row` words.
pub fn format_flat(data: &[u8], bytes_per_row: usize) -> String {
    if data.is_empty() || bytes_per_row == 0 {
        return String::new();
    }

    let words: Vec<String> = data.iter().map(|&b| byte_to_word(b)).collect();
    let lines: Vec<String> = words
        .chunks(bytes_per_row)
        .map(|chunk| chunk.join(" "))
        .collect();
    lines.join("\n")
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core format_flat -- --nocapture`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add stq8-core/src/q8.rs
git commit -m "feat(q8): add format_flat() with configurable row width"
```

---

### Task 4: Flashcard `Level` Enum

Create the `flashcard` module with the `Level` enum and its dimension methods.

**Files:**
- Create: `stq8-core/src/flashcard.rs`
- Modify: `stq8-core/src/lib.rs` (add `pub mod flashcard;`)

**Step 1: Write the failing tests**

Create `stq8-core/src/flashcard.rs` with just the test module first:

```rust
//! Flashcard challenge generation and validation.
//!
//! Generates random Q8 challenges at five difficulty levels and validates
//! spoken syllables against expected rows.

#[cfg(test)]
mod tests {
    use super::*;

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
}
```

**Step 2: Run tests to verify they fail**

First, add `pub mod flashcard;` to `stq8-core/src/lib.rs`.

Run: `cargo test -p stq8-core flashcard -- --nocapture`
Expected: FAIL — `Level` not found

**Step 3: Write the implementation**

Add above the `#[cfg(test)]` block in `flashcard.rs`:

```rust
use serde::{Deserialize, Serialize};

/// Challenge difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Level {
    /// 1×1 grid: 1 byte, 1 row of 1.
    Novice,
    /// 1×2 grid: 2 bytes, 1 row of 2.
    Apprentice,
    /// 2×2 grid: 4 bytes, 2 rows of 2.
    Journeyman,
    /// 3×3 grid: 9 bytes, 3 rows of 3.
    Expert,
    /// 8×4 grid: 32 bytes, 8 rows of 4.
    Master,
}

impl Level {
    /// All levels in order.
    pub const ALL: [Level; 5] = [
        Level::Novice,
        Level::Apprentice,
        Level::Journeyman,
        Level::Expert,
        Level::Master,
    ];

    /// Total bytes in the challenge.
    pub fn total_bytes(self) -> usize {
        match self {
            Level::Novice => 1,
            Level::Apprentice => 2,
            Level::Journeyman => 4,
            Level::Expert => 9,
            Level::Master => 32,
        }
    }

    /// Bytes per row (max 4).
    pub fn bytes_per_row(self) -> usize {
        match self {
            Level::Novice => 1,
            Level::Apprentice => 2,
            Level::Journeyman => 2,
            Level::Expert => 3,
            Level::Master => 4,
        }
    }

    /// Number of rows to speak.
    pub fn num_rows(self) -> usize {
        match self {
            Level::Novice => 1,
            Level::Apprentice => 1,
            Level::Journeyman => 2,
            Level::Expert => 3,
            Level::Master => 8,
        }
    }

    /// Total bits in the challenge.
    pub fn total_bits(self) -> usize {
        self.total_bytes() * 8
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core flashcard -- --nocapture`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add stq8-core/src/flashcard.rs stq8-core/src/lib.rs
git commit -m "feat(flashcard): add Level enum with grid dimensions"
```

---

### Task 5: Challenge Generation

`generate()` takes a level and caller-provided random bytes, returns a `Challenge` with data split into rows.

**Files:**
- Modify: `stq8-core/src/flashcard.rs`

**Step 1: Write the failing tests**

```rust
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
    assert_eq!(challenge.level, Level::Master);
    assert_eq!(challenge.data.len(), 32);
    assert_eq!(challenge.rows.len(), 8);
    for row in &challenge.rows {
        assert_eq!(row.len(), 4);
    }
    // Rows concatenated should equal data
    let flat: Vec<u8> = challenge.rows.iter().flatten().copied().collect();
    assert_eq!(flat, challenge.data);
}

#[test]
fn generate_truncates_excess_rng() {
    // Provide 100 bytes but Novice only needs 1
    let rng = vec![0xAB; 100];
    let challenge = generate(Level::Novice, &rng);
    assert_eq!(challenge.data, vec![0xAB]);
}

#[test]
fn generate_pads_insufficient_rng() {
    // Provide 0 bytes — should pad with zeros
    let challenge = generate(Level::Apprentice, &[]);
    assert_eq!(challenge.data.len(), 2);
    assert_eq!(challenge.rows.len(), 1);
}

#[test]
fn generate_expert_nine_bytes_three_rows() {
    let rng: Vec<u8> = (10..19).collect();
    let challenge = generate(Level::Expert, &rng);
    assert_eq!(challenge.data.len(), 9);
    assert_eq!(challenge.rows.len(), 3);
    assert_eq!(challenge.rows[0], vec![10, 11, 12]);
    assert_eq!(challenge.rows[1], vec![13, 14, 15]);
    assert_eq!(challenge.rows[2], vec![16, 17, 18]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core generate_ -- --nocapture`
Expected: FAIL — `generate` not found

**Step 3: Write the implementation**

```rust
/// A flashcard challenge: random data split into rows for validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Challenge {
    pub level: Level,
    /// The full byte sequence for this challenge.
    pub data: Vec<u8>,
    /// Data split into rows (each row has `level.bytes_per_row()` bytes).
    pub rows: Vec<Vec<u8>>,
}

/// Generate a challenge from caller-provided random bytes.
///
/// Sans-I/O: the caller supplies randomness (e.g. `crypto.getRandomValues()`
/// in JS, `getrandom` in native Rust). If `rng_bytes` is shorter than needed,
/// the remaining bytes are zero-filled. If longer, excess is truncated.
pub fn generate(level: Level, rng_bytes: &[u8]) -> Challenge {
    let total = level.total_bytes();
    let bpr = level.bytes_per_row();

    let mut data = vec![0u8; total];
    let copy_len = rng_bytes.len().min(total);
    data[..copy_len].copy_from_slice(&rng_bytes[..copy_len]);

    let rows: Vec<Vec<u8>> = data.chunks(bpr).map(|c| c.to_vec()).collect();

    Challenge { level, data, rows }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core generate_ -- --nocapture`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add stq8-core/src/flashcard.rs
git commit -m "feat(flashcard): add Challenge struct and generate()"
```

---

### Task 6: Row Validation

`validate_row()` compares classified syllables against expected bytes for one row.

**Files:**
- Modify: `stq8-core/src/flashcard.rs`

**Context:** Each byte in the expected row produces two syllables (high nibble, low nibble). The `heard` slice comes from the pipeline's `UtteranceResult.syllables` — we extract the `Syllable` from each accepted/suggested classification. Matching is exact: each position must be the same `Syllable`.

**Step 1: Write the failing tests**

```rust
use crate::q8::{Consonant, Vowel, Syllable};

#[test]
fn validate_row_exact_match() {
    // Byte 0x00 = syllables 'O, 'O (nibbles 0, 0)
    let expected = &[0x00u8];
    let s0 = Syllable::from_nibble(0); // 'O
    let result = validate_row(expected, &[s0, s0]);
    assert!(result.matched);
    assert_eq!(result.expected, vec![s0, s0]);
    assert_eq!(result.heard, vec![s0, s0]);
}

#[test]
fn validate_row_mismatch() {
    let expected = &[0x00u8]; // 'O 'O
    let s0 = Syllable::from_nibble(0); // 'O
    let s1 = Syllable::from_nibble(1); // 'U
    let result = validate_row(expected, &[s0, s1]);
    assert!(!result.matched);
    assert_eq!(result.expected, vec![s0, s0]);
    assert_eq!(result.heard, vec![s0, s1]);
}

#[test]
fn validate_row_multi_byte() {
    // Two bytes: 0x59 = JU KU, 0xA3 = KE 'I (nibbles 5,9 and 10,3)
    let expected = &[0x59u8, 0xA3];
    let syllables: Vec<Syllable> = [5, 9, 10, 3]
        .iter()
        .map(|&n| Syllable::from_nibble(n))
        .collect();
    let result = validate_row(expected, &syllables);
    assert!(result.matched);
    assert_eq!(result.heard.len(), 4);
}

#[test]
fn validate_row_wrong_length_too_few() {
    let expected = &[0x00u8]; // needs 2 syllables
    let s0 = Syllable::from_nibble(0);
    let result = validate_row(expected, &[s0]); // only 1
    assert!(!result.matched);
}

#[test]
fn validate_row_wrong_length_too_many() {
    let expected = &[0x00u8]; // needs 2 syllables
    let s0 = Syllable::from_nibble(0);
    let result = validate_row(expected, &[s0, s0, s0]); // 3 given
    assert!(!result.matched);
}

#[test]
fn validate_row_empty_expected() {
    let result = validate_row(&[], &[]);
    assert!(result.matched);
    assert!(result.expected.is_empty());
    assert!(result.heard.is_empty());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p stq8-core validate_row -- --nocapture`
Expected: FAIL — `validate_row` not found

**Step 3: Write the implementation**

```rust
use crate::q8::Syllable;

/// Result of validating spoken syllables against one expected row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowResult {
    /// Whether all syllables matched exactly.
    pub matched: bool,
    /// The expected syllables (derived from the row's bytes).
    pub expected: Vec<Syllable>,
    /// The syllables that were heard (from classification).
    pub heard: Vec<Syllable>,
}

/// Validate heard syllables against one row of expected bytes.
///
/// Each byte in `expected_bytes` produces two syllables (high nibble, low nibble).
/// The `heard` slice must have exactly `expected_bytes.len() * 2` syllables and
/// each must match positionally for `matched` to be true.
pub fn validate_row(expected_bytes: &[u8], heard: &[Syllable]) -> RowResult {
    let expected: Vec<Syllable> = expected_bytes
        .iter()
        .flat_map(|&b| {
            let high = (b >> 4) & 0x0F;
            let low = b & 0x0F;
            [Syllable::from_nibble(high), Syllable::from_nibble(low)]
        })
        .collect();

    let matched = expected.len() == heard.len()
        && expected.iter().zip(heard.iter()).all(|(e, h)| e == h);

    RowResult {
        matched,
        expected,
        heard: heard.to_vec(),
    }
}
```

**Note:** The `use crate::q8::Syllable;` should be at the top of `flashcard.rs` alongside the serde import. The test module will also need `use crate::q8::{Consonant, Vowel, Syllable};` — though `Consonant` and `Vowel` aren't strictly needed since tests use `Syllable::from_nibble()`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p stq8-core validate_row -- --nocapture`
Expected: 6 tests PASS

**Step 5: Run full suite**

Run: `cargo test -p stq8-core`
Expected: All tests pass

**Step 6: Commit**

```bash
git add stq8-core/src/flashcard.rs
git commit -m "feat(flashcard): add validate_row() for syllable checking"
```

---

### Task 7: WASM Bindings

Expose the new flashcard and formatting APIs through `stq8-web` for JavaScript consumption.

**Files:**
- Modify: `stq8-web/src/lib.rs`

**Context:** The existing `WasmPipeline` struct has static methods like `encode_q8()`. We'll add:
- Static methods for formatting: `format_box_q8()`, `format_flat_q8()`
- A standalone `#[wasm_bindgen]` function or methods for flashcard operations. Since `generate` and `validate_row` don't need a Pipeline instance, they can be free functions or static methods.
- `Level` needs to cross the WASM boundary. wasm-bindgen doesn't support Rust enums with methods well, so we'll use u8 (0-4) on the JS side and convert.

**Step 1: Write the implementation**

Add these to `stq8-web/src/lib.rs`:

```rust
use stq8_core::flashcard;
use stq8_core::q8;
```

Add these static methods to the existing `#[wasm_bindgen] impl WasmPipeline` block:

```rust
    /// Format bytes as Q8-BOX grid (consonant row / vowel row).
    pub fn format_box_q8(data: &[u8], bytes_per_row: usize) -> String {
        q8::format_box(data, bytes_per_row)
    }

    /// Format bytes as Q8-FLAT phonetic text, wrapped at bytes_per_row.
    pub fn format_flat_q8(data: &[u8], bytes_per_row: usize) -> String {
        q8::format_flat(data, bytes_per_row)
    }

    /// Generate a flashcard challenge.
    /// level: 0=Novice, 1=Apprentice, 2=Journeyman, 3=Expert, 4=Master.
    /// rng_bytes: caller-provided random bytes.
    /// Returns JSON: { level, data: number[], rows: number[][] }
    pub fn generate_challenge(level: u8, rng_bytes: &[u8]) -> Result<String, JsError> {
        let level = match level {
            0 => flashcard::Level::Novice,
            1 => flashcard::Level::Apprentice,
            2 => flashcard::Level::Journeyman,
            3 => flashcard::Level::Expert,
            4 => flashcard::Level::Master,
            _ => return Err(JsError::new(&format!("invalid level {level} (0–4)"))),
        };
        let challenge = flashcard::generate(level, rng_bytes);
        serde_json::to_string(&challenge).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Validate heard syllables against an expected row.
    /// expected_bytes: the row's bytes.
    /// heard_nibbles: array of nibble values (0-15) from classification.
    /// Returns JSON: { matched: bool, expected: [...], heard: [...] }
    pub fn validate_row(expected_bytes: &[u8], heard_nibbles: &[u8]) -> Result<String, JsError> {
        let heard: Vec<stq8_core::q8::Syllable> = heard_nibbles
            .iter()
            .map(|&n| stq8_core::q8::Syllable::from_nibble(n))
            .collect();
        let result = flashcard::validate_row(expected_bytes, &heard);
        serde_json::to_string(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get level dimensions as JSON.
    /// level: 0=Novice, 1=Apprentice, 2=Journeyman, 3=Expert, 4=Master.
    /// Returns JSON: { total_bytes, bytes_per_row, num_rows, total_bits }
    pub fn level_info(level: u8) -> Result<String, JsError> {
        let level = match level {
            0 => flashcard::Level::Novice,
            1 => flashcard::Level::Apprentice,
            2 => flashcard::Level::Journeyman,
            3 => flashcard::Level::Expert,
            4 => flashcard::Level::Master,
            _ => return Err(JsError::new(&format!("invalid level {level} (0–4)"))),
        };
        let info = serde_json::json!({
            "total_bytes": level.total_bytes(),
            "bytes_per_row": level.bytes_per_row(),
            "num_rows": level.num_rows(),
            "total_bits": level.total_bits(),
        });
        Ok(info.to_string())
    }
```

**Step 2: Add `serde_json` to stq8-web dependencies if not present**

Check `stq8-web/Cargo.toml` — `serde_json` is already listed. Good.

**Step 3: Verify it compiles**

Run: `cargo check -p stq8-web`
Expected: No errors. (We can't run wasm-bindgen tests without a wasm target, but `cargo check` verifies the code compiles.)

**Step 4: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass (stq8-web has no test targets, stq8-core runs all tests)

**Step 5: Run clippy**

Run: `cargo clippy --workspace`
Expected: Zero warnings

**Step 6: Commit**

```bash
git add stq8-web/src/lib.rs
git commit -m "feat(stq8-web): add WASM bindings for flashcard and Q8 formatting"
```

---

## Summary

| Task | What | Tests added |
|------|------|-------------|
| 1 | BOX character constants | 2 |
| 2 | `format_box()` | 5 |
| 3 | `format_flat()` | 4 |
| 4 | `Level` enum | 6 |
| 5 | `generate()` | 5 |
| 6 | `validate_row()` | 6 |
| 7 | WASM bindings | (compile check) |
| **Total** | | **28 new tests** |

After this plan, stq8-core will have the complete flashcard engine. The harmony-client Svelte UI (FlashcardSession, FlashcardGrid, FlashcardStats components) is a separate plan for the harmony-client repo.
