use wasm_bindgen::prelude::*;
use stq8_core::pipeline::Pipeline;
use stq8_core::q8::{self, Phoneme, Consonant, Vowel};
use stq8_core::mfcc;

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
        Self { inner: Pipeline::new() }
    }

    /// Feed a calibration sample. phoneme_index: 0-7
    /// (0=GlottalStop, 1=J, 2=K, 3=V, 4=O, 5=U, 6=E, 7=I).
    /// pcm: raw f32 samples at 16kHz.
    pub fn add_calibration_sample(&mut self, phoneme_index: u8, pcm: &[f32]) {
        let phoneme = index_to_phoneme(phoneme_index);
        let features = mfcc::extract_features(pcm);
        self.inner.add_calibration_sample(phoneme, features);
    }

    pub fn finalize_calibration(&mut self) {
        self.inner.finalize_calibration();
    }

    pub fn is_calibrated(&self) -> bool {
        self.inner.is_calibrated()
    }

    /// Process a PTT utterance. Returns JSON-encoded UtteranceResult.
    pub fn process(&self, pcm: &[f32]) -> String {
        let result = self.inner.process(pcm);
        serde_json::to_string(&result).unwrap_or_default()
    }

    /// Encode bytes to Q8 text.
    pub fn encode_q8(data: &[u8]) -> String {
        q8::encode(data)
    }

    /// Decode Q8 text to bytes. Returns empty vec on error.
    pub fn decode_q8(text: &str) -> Vec<u8> {
        q8::decode(text).unwrap_or_default()
    }

    pub fn export_profile(&self) -> String {
        self.inner.export_profile_json().unwrap_or_default()
    }

    pub fn import_profile(&mut self, json: &str) -> bool {
        self.inner.import_profile_json(json).is_ok()
    }
}

fn index_to_phoneme(idx: u8) -> Phoneme {
    match idx {
        0 => Phoneme::Consonant(Consonant::GlottalStop),
        1 => Phoneme::Consonant(Consonant::J),
        2 => Phoneme::Consonant(Consonant::K),
        3 => Phoneme::Consonant(Consonant::V),
        4 => Phoneme::Vowel(Vowel::O),
        5 => Phoneme::Vowel(Vowel::U),
        6 => Phoneme::Vowel(Vowel::E),
        7 => Phoneme::Vowel(Vowel::I),
        _ => Phoneme::Vowel(Vowel::O), // fallback
    }
}
