use stq8_core::flashcard;
use stq8_core::mfcc;
use stq8_core::pipeline::Pipeline;
use stq8_core::q8::{self, Syllable};
use wasm_bindgen::prelude::*;

/// Parse a u8 level index (0–4) into a flashcard Level.
fn parse_level(level: u8) -> Result<flashcard::Level, JsError> {
    match level {
        0 => Ok(flashcard::Level::Novice),
        1 => Ok(flashcard::Level::Apprentice),
        2 => Ok(flashcard::Level::Journeyman),
        3 => Ok(flashcard::Level::Expert),
        4 => Ok(flashcard::Level::Master),
        _ => Err(JsError::new(&format!(
            "invalid level {level} (expected 0–4)"
        ))),
    }
}

#[wasm_bindgen]
pub struct WasmPipeline {
    inner: Pipeline,
}

impl Default for WasmPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Pipeline::new(),
        }
    }

    /// Feed a calibration sample. syllable_index: 0-15
    /// (maps to Q8 syllables: 0='O, 1='U, ..., 15=VI).
    /// pcm: raw f32 samples at 16kHz.
    pub fn add_calibration_sample(
        &mut self,
        syllable_index: u8,
        pcm: &[f32],
    ) -> Result<(), JsError> {
        if syllable_index > 15 {
            return Err(JsError::new(&format!(
                "syllable_index {syllable_index} out of range (0–15)"
            )));
        }
        let syllable = Syllable::from_nibble(syllable_index);
        let features = mfcc::extract_features(pcm);
        self.inner.add_calibration_sample(syllable, features);
        Ok(())
    }

    pub fn finalize_calibration(&mut self) {
        self.inner.finalize_calibration();
    }

    /// Feed a transversal calibration sample.
    /// phrase_index: 0 or 1 (phrase 1 = diagonal, phrase 2 = anti-diagonal).
    /// syllable_index: 0-15 (must be a valid syllable for the given phrase).
    /// pcm: raw f32 samples at 16kHz.
    pub fn add_transversal_sample(
        &mut self,
        phrase_index: u8,
        syllable_index: u8,
        pcm: &[f32],
    ) -> Result<(), JsError> {
        if phrase_index > 1 {
            return Err(JsError::new(&format!(
                "phrase_index {phrase_index} out of range (0–1)"
            )));
        }
        if syllable_index > 15 {
            return Err(JsError::new(&format!(
                "syllable_index {syllable_index} out of range (0–15)"
            )));
        }
        let syllable = Syllable::from_nibble(syllable_index);
        let features = mfcc::extract_features(pcm);
        self.inner
            .add_transversal_sample(phrase_index, syllable, features);
        Ok(())
    }

    pub fn finalize_transversal_calibration(&mut self) {
        self.inner.finalize_transversal_calibration();
    }

    pub fn is_calibrated(&self) -> bool {
        self.inner.is_calibrated()
    }

    /// Process a PTT utterance. Returns JSON-encoded UtteranceResult,
    /// or a JSON error object if serialization fails.
    pub fn process(&mut self, pcm: &[f32]) -> String {
        let result = self.inner.process(pcm);
        serde_json::to_string(&result).unwrap_or_else(|e| {
            let escaped = e.to_string().replace('\\', "\\\\").replace('"', "\\\"");
            format!("{{\"error\":\"{escaped}\"}}")
        })
    }

    /// Encode bytes to Q8 text.
    pub fn encode_q8(data: &[u8]) -> String {
        q8::encode(data)
    }

    /// Decode Q8 text to bytes. Returns an error if the text contains
    /// invalid Q8 words.
    pub fn decode_q8(text: &str) -> Result<Vec<u8>, JsError> {
        q8::decode(text).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Set the profile creation timestamp (seconds since Unix epoch).
    /// Call this before `export_profile` so exported profiles carry a
    /// meaningful timestamp.
    pub fn set_created_epoch_secs(&mut self, secs: u64) {
        self.inner.set_created_epoch_secs(secs);
    }

    pub fn export_profile(&self) -> Result<String, JsError> {
        self.inner
            .export_profile_json()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn import_profile(&mut self, json: &str) -> Result<(), JsError> {
        self.inner
            .import_profile_json(json)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Format bytes as Q8-BOX grid (consonant row / vowel row).
    pub fn format_box_q8(data: &[u8], bytes_per_row: usize) -> Result<String, JsError> {
        if bytes_per_row == 0 {
            return Err(JsError::new("bytes_per_row must be > 0"));
        }
        Ok(q8::format_box(data, bytes_per_row))
    }

    /// Format bytes as Q8-FLAT phonetic text, wrapped at bytes_per_row.
    pub fn format_flat_q8(data: &[u8], bytes_per_row: usize) -> Result<String, JsError> {
        if bytes_per_row == 0 {
            return Err(JsError::new("bytes_per_row must be > 0"));
        }
        Ok(q8::format_flat(data, bytes_per_row))
    }

    /// Generate a flashcard challenge. Returns JSON-serialized Challenge.
    /// Level: 0=Novice, 1=Apprentice, 2=Journeyman, 3=Expert, 4=Master.
    pub fn generate_challenge(level: u8, rng_bytes: &[u8]) -> Result<String, JsError> {
        let level = parse_level(level)?;
        let required = level.total_bytes();
        if rng_bytes.len() < required {
            return Err(JsError::new(&format!(
                "rng_bytes too short: need {} bytes for {:?}, got {}",
                required, level, rng_bytes.len()
            )));
        }
        let challenge = flashcard::generate(level, rng_bytes);
        serde_json::to_string(&challenge).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Validate a row of heard nibbles against expected bytes.
    /// Returns JSON-serialized RowResult.
    /// Each value in `heard_nibbles` must be 0–15.
    pub fn validate_row(expected_bytes: &[u8], heard_nibbles: &[u8]) -> Result<String, JsError> {
        for &n in heard_nibbles {
            if n > 15 {
                return Err(JsError::new(&format!(
                    "heard nibble {n} out of range (0–15)"
                )));
            }
        }
        let heard: Vec<Syllable> = heard_nibbles
            .iter()
            .map(|&n| Syllable::from_nibble(n))
            .collect();
        let result = flashcard::validate_row(expected_bytes, &heard);
        serde_json::to_string(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Return JSON with level metadata: total_bytes, bytes_per_row, num_rows, total_bits.
    /// Level: 0=Novice, 1=Apprentice, 2=Journeyman, 3=Expert, 4=Master.
    pub fn level_info(level: u8) -> Result<String, JsError> {
        let level = parse_level(level)?;
        let info = serde_json::json!({
            "total_bytes": level.total_bytes(),
            "bytes_per_row": level.bytes_per_row(),
            "num_rows": level.num_rows(),
            "total_bits": level.total_bits(),
        });
        Ok(info.to_string())
    }
}
