# Transversal Classifier Design

## Goal

Add an 8-sound calibration mode to the STQ8 pipeline that reconstructs all 16 nibbles from just two 4-syllable phrases, using orthogonal Latin square transversals. Compare accuracy, confidence, and performance against the existing 16-sound calibration.

## Architecture

The Q8 grid is a 4x4 matrix (4 consonants x 4 vowels = 16 syllables). Two carefully chosen phrases — each covering all 4 consonants and all 4 vowels exactly once — form orthogonal transversals of this grid. By calibrating on only these 8 syllables and using Top-2-per-phrase intersection at inference time, we can reconstruct any of the 16 possible nibbles.

## Transversal Phrases

```
Phrase 1 (diagonal):      'O(00) JU(05) KE(10) VI(15)
Phrase 2 (anti-diagonal): 'I(03) JE(06) KO(08) VU(13)
```

Grid coverage (P1 = Phrase 1, P2 = Phrase 2):

|       | V=O  | V=U  | V=E  | V=I  |
|-------|------|------|------|------|
| C='   | P1   |      |      | P2   |
| C=J   |      | P1   | P2   |      |
| C=K   | P2   |      | P1   |      |
| C=V   |      | P2   |      | P1   |

Every uncalibrated cell shares exactly one consonant with a P1 sample and one with a P2 sample, and one vowel with a P1 sample and one with a P2 sample. This guarantees unique reconstruction via set intersection.

## Inference: Top-2-Per-Phrase Intersection

1. Compute cosine similarity of incoming MFCC features to all 8 centroids
2. For each phrase, rank by similarity, take Top-2
3. Extract the consonant indices and vowel indices from each phrase's Top-2
4. Intersect consonant sets across phrases -> single consonant index
5. Intersect vowel sets across phrases -> single vowel index
6. Combine: `(consonant << 2) | vowel` -> predicted nibble

### Worked Example: User says "VE" (nibble 14)

- P1 Top-2: KE (shares E) and VI (shares V) -> consonants {K,V}, vowels {E,I}
- P2 Top-2: JE (shares E) and VU (shares V) -> consonants {J,V}, vowels {E,U}
- Intersect consonants: {K,V} ∩ {J,V} = {V}
- Intersect vowels: {E,I} ∩ {E,U} = {E}
- Result: V + E = VE (nibble 14)

## Confidence: Margin-Based

For each phrase, the Top-2 are the "matching pair" (share one axis with the target). The bottom 2 share neither axis.

```
per_phrase_margin = mean(top2_similarities) - mean(bottom2_similarities)
overall_confidence = mean(phrase1_margin, phrase2_margin)
```

Normalized to 0.0-1.0. High margin = clean intersection = reliable prediction.

## Data Model

### CalibrationMode

```rust
pub enum CalibrationMode {
    Full,        // 16-sound: all CV pairs calibrated directly
    Transversal, // 8-sound: two orthogonal phrases, reconstruct via intersection
}
```

### TransversalClassifier

```rust
pub struct TransversalClassifier {
    phrase_centroids: [Vec<(Syllable, Vec<f32>)>; 2],
    trained: bool,
}

pub struct TransversalClassification {
    pub syllable: Syllable,
    pub confidence: f32,
    pub phrase1_top2: [(Syllable, f32); 2],
    pub phrase2_top2: [(Syllable, f32); 2],
}
```

### UserProfile Changes

`UserProfile` gains a `mode: CalibrationMode` field (defaults to `Full`). When `Transversal`, stores 8 centroids tagged with phrase index. Thresholds are stored per-mode since margin-based confidence has a different scale than softmax.

## Pipeline Integration

### Calibration

- `Pipeline::add_transversal_sample(phrase_index: u8, syllable, pcm)` — accumulates tagged samples
- `Pipeline::finalize_transversal_calibration()` — trains TransversalClassifier, stores in profile

Existing 16-sound calibration methods remain unchanged.

### Inference

`Pipeline::process()` checks the profile's `CalibrationMode`:
- `Full` -> `Classifier::classify()` -> direct nibble + softmax confidence
- `Transversal` -> `TransversalClassifier::classify()` -> reconstructed nibble + margin confidence

Everything downstream (threshold decision, profile remap, carry state, Q8 encoding) is identical — both paths produce `(Syllable, confidence)`.

## Testing & Benchmarking

### Unit Tests
- Reconstruction correctness for all 16 nibbles from 8 centroids
- Confidence high when margins clean, low when ambiguous
- Defensive handling if intersection fails to produce unique answer
- Calibration rejects syllables not in phrase set

### Accuracy Comparison
- Generate 16 ground-truth synthetic centroids
- Train both classifiers from same data (16 vs 8 samples)
- Compare accuracy, confidence distributions, noise robustness

### Performance
- Inference latency (8 vs 16 cosine similarity computations)
- Calibration time (8 vs 16 samples)
- Memory footprint

## Scope

### In Scope
- `TransversalClassifier` struct in stq8-core
- `CalibrationMode` enum in profile
- Pipeline calibration + inference integration
- Transversal phrase constants
- Unit tests + accuracy comparison benchmarks
- WASM bindings for new calibration methods

### Out of Scope
- Express lane for 8-sound mode (acceptable trade-off)
- Client UI for selecting calibration mode (separate bead)
- Custom phrase selection (hardcoded is fine)
- Profile migration (new field defaults to Full)

## Risk

MFCC similarity may not decompose cleanly along consonant/vowel axes for all speakers. If a non-matching centroid is closer than a matching one, Top-2 selection breaks. Accuracy benchmarks will surface this. Real audio testing is a follow-up concern.
