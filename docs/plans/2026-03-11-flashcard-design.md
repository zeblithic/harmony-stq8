# Q8 Flashcard System Design

## Goal

A voice-driven flashcard app for practicing Q8 symbol recognition and recall.
After calibrating their voice profile, users exercise their ability to read
Q8-BOX grids and speak the corresponding syllables with speed and accuracy.

## Q8 Display Formats

Two representations of the same data, optimized for different tasks:

| Bits | Consonant (Box / Flat) | Vowel (Box / Flat) |
|------|------------------------|-------------------|
| `00` | `A` / `'`              | `O` / `O`         |
| `01` | `>` / `J`              | `=` / `U`         |
| `10` | `<` / `K`              | `X` / `E`         |
| `11` | `V` / `V`              | `I` / `I`         |

- **Q8-BOX** — 2D grid with consonant row on top, vowel row on bottom.
  Uses visual symbols (`A > < V` / `O = X I`). Optimized for grid scanning.
- **Q8-FLAT** — inline phonetic, packed as 4-char words (e.g. `KU'E` for
  byte `0x92`). Uses spoken-word symbols (`' J K V` / `O U E I`).
  Optimized for reading aloud.

## Challenge Levels

Five levels with fixed grid shapes. Every row is ≤ 4 bytes.

| Level       | Grid | Bytes | Bits | Bytes/row | Rows |
|-------------|------|-------|------|-----------|------|
| Novice      | 1×1  | 1     | 8    | 1         | 1    |
| Apprentice  | 1×2  | 2     | 16   | 2         | 1    |
| Journeyman  | 2×2  | 4     | 32   | 2         | 2    |
| Expert      | 3×3  | 9     | 72   | 3         | 3    |
| Master      | 8×4  | 32    | 256  | 4         | 8    |

Users pick any level freely — no gating or unlock progression.

## Interaction Model

### PTT Rules

- **Hold PTT = active.** Syllables are classified and validated against
  the current row.
- **Release PTT = cancel current row.** Progress on the current row resets.
  Completed rows on the same card are banked (checkpoint at row boundaries).
- **2-second momentum timeout.** If 2 seconds pass without a successfully
  classified syllable advancing position in the current row, the row resets.
  PTT stays held — retry immediately.

### Flow

1. User picks a level.
2. Random challenge generated (random bytes → Q8-BOX grid).
3. First row highlighted as active.
4. User holds PTT, speaks the row's syllables left-to-right.
5. **Row matches:** Row turns green, advance to next row. If last row →
   card complete, next card auto-generated. Keep going while PTT held.
6. **Row mismatches:** Show what was heard vs expected (in Q8-FLAT phonetic
   form). Row stays active — retry.
7. **Momentum timeout or PTT release:** Current row resets. Banked rows kept.

### Combo Chaining

At all levels, holding PTT and passing cards back-to-back builds a combo.
Novice and Apprentice are natural combo levels (single-row cards chain
rapidly). The combo counter and effective bitrate reward sustained accuracy.

### Mismatch Feedback

No penalty on wrong answers. Show what was heard vs expected:

```
Expected: KEVO 'O'I JU'E
Heard:    KEVO 'O'I JU'O
                       ^^
```

If nothing could be classified: "Couldn't hear that clearly, try again."

## Stats

Tracked per level, per session. No persistence in v1.

- **Best time** — fastest card completion (first syllable to last row validated)
- **Average time** — rolling mean of all completed cards
- **Previous time** — most recent card (for immediate comparison)
- **Effective bitrate** — `bits_completed / elapsed_seconds`. Only passed
  data counts as bits; retries consume time but don't add bits. For combo
  chains: total bits of all chained cards / total seconds of the chain.
- **Combo counter** — consecutive cards passed without PTT release or timeout

## Display

Primary view: Q8-BOX grid with row highlighting (completed / active / upcoming).

**Hint toggle:** When enabled, shows Q8-FLAT phonetic text for the active
row only.

**Stats display:** Current elapsed timer, combo counter during play.
Best/avg/prev times and session bitrate visible in sidebar/footer.

## Architecture

**Approach:** Challenge engine in Rust (stq8-core), session state in
Svelte (harmony-client).

### stq8-core Additions

**`q8` module — formatting functions:**
- `format_box(data: &[u8], bytes_per_row: usize) -> String` — Q8-BOX
- `format_flat(data: &[u8]) -> String` — Q8-FLAT phonetic

**`flashcard` module:**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Novice,      // 1×1, 1 byte,  1 row
    Apprentice,  // 1×2, 2 bytes, 1 row
    Journeyman,  // 2×2, 4 bytes, 2 rows
    Expert,      // 3×3, 9 bytes, 3 rows
    Master,      // 8×4, 32 bytes, 8 rows
}

impl Level {
    pub fn total_bytes(&self) -> usize;
    pub fn bytes_per_row(&self) -> usize;
    pub fn num_rows(&self) -> usize;
    pub fn total_bits(&self) -> usize;
}

pub struct Challenge {
    pub level: Level,
    pub data: Vec<u8>,
    pub rows: Vec<Vec<u8>>,
}

/// Caller provides random bytes (sans-I/O: no rand crate in core).
pub fn generate(level: Level, rng_bytes: &[u8]) -> Challenge;

pub struct RowResult {
    pub matched: bool,
    pub expected: Vec<Syllable>,
    pub heard: Vec<Syllable>,
}

pub fn validate_row(expected_bytes: &[u8], heard: &[Syllable]) -> RowResult;
```

**WASM bindings** in `stq8-web` for all new functions.

### harmony-client Additions (Svelte 5)

- **`FlashcardSession`** — session state: current card, current row,
  combo counter, timers, stats
- **`FlashcardGrid`** — renders Q8-BOX grid with row highlighting
- **`FlashcardStats`** — best/avg/prev times, effective bitrate, combo
- **PTT integration** — Web Audio capture, PTT release → row reset,
  2-second momentum timer
- **Level selector** — all 5 levels, freely accessible
- **Hint toggle** — Q8-FLAT for active row

### Data Flow

```
PTT hold → Web Audio capture → pipeline.process(pcm) → UtteranceResult
  → validate_row(expected, heard_syllables) → RowResult
  → update UI (advance row / show mismatch / reset on timeout)
```
