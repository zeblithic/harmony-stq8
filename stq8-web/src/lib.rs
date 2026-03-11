use stq8_core::mfcc;
use stq8_core::pipeline::Pipeline;
use stq8_core::q8::{self, Syllable};
use wasm_bindgen::prelude::*;

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

    pub fn is_calibrated(&self) -> bool {
        self.inner.is_calibrated()
    }

    /// Process a PTT utterance. Returns JSON-encoded UtteranceResult,
    /// or a JSON error object if serialization fails.
    pub fn process(&mut self, pcm: &[f32]) -> String {
        let result = self.inner.process(pcm);
        serde_json::to_string(&result)
            .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
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
}
