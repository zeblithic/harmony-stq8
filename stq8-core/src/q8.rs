//! Q8 encoding: pronounceable binary representation.
//!
//! Each nibble (4 bits) maps to a consonant-vowel syllable:
//! - Consonant (high 2 bits): `'`=00, `J`=01, `K`=10, `V`=11
//! - Vowel (low 2 bits): `O`=00, `U`=01, `E`=10, `I`=11
//!
//! Two syllables form a word (one byte). Eight words per line.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The four consonant phonemes, indexed by their 2-bit value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Consonant {
    /// `'` — glottal stop (bits 00)
    GlottalStop,
    /// `J` (bits 01)
    J,
    /// `K` (bits 10)
    K,
    /// `V` (bits 11)
    V,
}

impl Consonant {
    /// Returns the 2-bit value for this consonant.
    pub fn bits(self) -> u8 {
        match self {
            Consonant::GlottalStop => 0b00,
            Consonant::J => 0b01,
            Consonant::K => 0b10,
            Consonant::V => 0b11,
        }
    }

    /// Construct from a 2-bit value (only low 2 bits are used).
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Consonant::GlottalStop,
            0b01 => Consonant::J,
            0b10 => Consonant::K,
            0b11 => Consonant::V,
            _ => unreachable!(),
        }
    }

    /// The character representation.
    pub fn char(self) -> char {
        match self {
            Consonant::GlottalStop => '\'',
            Consonant::J => 'J',
            Consonant::K => 'K',
            Consonant::V => 'V',
        }
    }
}

/// The four vowel phonemes, indexed by their 2-bit value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vowel {
    /// `O` (bits 00)
    O,
    /// `U` (bits 01)
    U,
    /// `E` (bits 10)
    E,
    /// `I` (bits 11)
    I,
}

impl Vowel {
    /// Returns the 2-bit value for this vowel.
    pub fn bits(self) -> u8 {
        match self {
            Vowel::O => 0b00,
            Vowel::U => 0b01,
            Vowel::E => 0b10,
            Vowel::I => 0b11,
        }
    }

    /// Construct from a 2-bit value (only low 2 bits are used).
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Vowel::O,
            0b01 => Vowel::U,
            0b10 => Vowel::E,
            0b11 => Vowel::I,
            _ => unreachable!(),
        }
    }

    /// The character representation.
    pub fn char(self) -> char {
        match self {
            Vowel::O => 'O',
            Vowel::U => 'U',
            Vowel::E => 'E',
            Vowel::I => 'I',
        }
    }
}

/// A single phoneme: either a consonant or a vowel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phoneme {
    Consonant(Consonant),
    Vowel(Vowel),
}

/// Errors that can occur during Q8 decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// A word in the input is not a valid Q8 word (two valid syllables).
    InvalidWord(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::InvalidWord(w) => write!(f, "invalid Q8 word: {w:?}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Consonant characters in bit-order.
const CONSONANT_CHARS: [char; 4] = ['\'', 'J', 'K', 'V'];
/// Vowel characters in bit-order.
const VOWEL_CHARS: [char; 4] = ['O', 'U', 'E', 'I'];

/// Convert a nibble (low 4 bits of `nibble`) to a two-character syllable string.
pub fn nibble_to_syllable(nibble: u8) -> String {
    let consonant = CONSONANT_CHARS[((nibble >> 2) & 0x3) as usize];
    let vowel = VOWEL_CHARS[(nibble & 0x3) as usize];
    format!("{consonant}{vowel}")
}

/// Parse a two-character syllable string back to a nibble value (0..15).
/// Returns `None` if the string is not a valid syllable.
pub fn syllable_to_nibble(s: &str) -> Option<u8> {
    let mut chars = s.chars();
    let c = chars.next()?;
    let v = chars.next()?;
    if chars.next().is_some() {
        return None; // too many characters
    }

    let c_upper = c.to_ascii_uppercase();
    let v_upper = v.to_ascii_uppercase();

    let c_bits = CONSONANT_CHARS.iter().position(|&ch| ch == c_upper)?;
    let v_bits = VOWEL_CHARS.iter().position(|&ch| ch == v_upper)?;

    Some(((c_bits as u8) << 2) | (v_bits as u8))
}

/// Convert a byte to a four-character Q8 word (two syllables).
pub fn byte_to_word(b: u8) -> String {
    let high = (b >> 4) & 0x0F;
    let low = b & 0x0F;
    format!("{}{}", nibble_to_syllable(high), nibble_to_syllable(low))
}

/// Parse a four-character Q8 word back to a byte.
/// Returns `None` if the word is not a valid Q8 word.
pub fn word_to_byte(word: &str) -> Option<u8> {
    // A valid word is exactly 4 characters: two syllables of 2 characters each.
    let chars: Vec<char> = word.chars().collect();
    if chars.len() != 4 {
        return None;
    }
    let high_syllable: String = chars[0..2].iter().collect();
    let low_syllable: String = chars[2..4].iter().collect();
    let high = syllable_to_nibble(&high_syllable)?;
    let low = syllable_to_nibble(&low_syllable)?;
    Some((high << 4) | low)
}

/// Encode a byte slice into Q8 text.
///
/// Words are space-separated, with a newline every 8 words.
pub fn encode(data: &[u8]) -> String {
    let words: Vec<String> = data.iter().map(|&b| byte_to_word(b)).collect();
    let lines: Vec<String> = words.chunks(8).map(|chunk| chunk.join(" ")).collect();
    lines.join("\n")
}

/// Decode Q8 text back into bytes.
///
/// Whitespace (spaces and newlines) is used to split words.
/// Each word must be exactly 4 characters (two valid syllables).
pub fn decode(text: &str) -> Result<Vec<u8>, DecodeError> {
    let mut bytes = Vec::new();
    for word in text.split_whitespace() {
        match word_to_byte(word) {
            Some(b) => bytes.push(b),
            None => return Err(DecodeError::InvalidWord(word.to_string())),
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Consonant enum ---

    #[test]
    fn consonant_bits_roundtrip() {
        for bits in 0..4u8 {
            let c = Consonant::from_bits(bits);
            assert_eq!(c.bits(), bits);
        }
    }

    #[test]
    fn consonant_chars() {
        assert_eq!(Consonant::GlottalStop.char(), '\'');
        assert_eq!(Consonant::J.char(), 'J');
        assert_eq!(Consonant::K.char(), 'K');
        assert_eq!(Consonant::V.char(), 'V');
    }

    #[test]
    fn consonant_bits_values() {
        assert_eq!(Consonant::GlottalStop.bits(), 0b00);
        assert_eq!(Consonant::J.bits(), 0b01);
        assert_eq!(Consonant::K.bits(), 0b10);
        assert_eq!(Consonant::V.bits(), 0b11);
    }

    // --- Vowel enum ---

    #[test]
    fn vowel_bits_roundtrip() {
        for bits in 0..4u8 {
            let v = Vowel::from_bits(bits);
            assert_eq!(v.bits(), bits);
        }
    }

    #[test]
    fn vowel_chars() {
        assert_eq!(Vowel::O.char(), 'O');
        assert_eq!(Vowel::U.char(), 'U');
        assert_eq!(Vowel::E.char(), 'E');
        assert_eq!(Vowel::I.char(), 'I');
    }

    #[test]
    fn vowel_bits_values() {
        assert_eq!(Vowel::O.bits(), 0b00);
        assert_eq!(Vowel::U.bits(), 0b01);
        assert_eq!(Vowel::E.bits(), 0b10);
        assert_eq!(Vowel::I.bits(), 0b11);
    }

    // --- Phoneme enum ---

    #[test]
    fn phoneme_variants() {
        let pc = Phoneme::Consonant(Consonant::K);
        let pv = Phoneme::Vowel(Vowel::E);
        // Just verify they are distinct and Debug works
        assert_ne!(pc, pv);
        assert_eq!(pc, Phoneme::Consonant(Consonant::K));
        assert_eq!(pv, Phoneme::Vowel(Vowel::E));
    }

    #[test]
    fn phoneme_is_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Phoneme::Consonant(Consonant::J));
        set.insert(Phoneme::Vowel(Vowel::O));
        assert_eq!(set.len(), 2);
    }

    // --- nibble_to_syllable ---

    #[test]
    fn nibble_to_syllable_all_16() {
        let expected = [
            "'O", "'U", "'E", "'I", "JO", "JU", "JE", "JI", "KO", "KU", "KE", "KI", "VO", "VU",
            "VE", "VI",
        ];
        for (nibble, &exp) in expected.iter().enumerate() {
            assert_eq!(
                nibble_to_syllable(nibble as u8),
                exp,
                "nibble {nibble} should encode to {exp:?}"
            );
        }
    }

    // --- syllable_to_nibble ---

    #[test]
    fn syllable_to_nibble_all_16() {
        let syllables = [
            "'O", "'U", "'E", "'I", "JO", "JU", "JE", "JI", "KO", "KU", "KE", "KI", "VO", "VU",
            "VE", "VI",
        ];
        for (expected, &syl) in syllables.iter().enumerate() {
            assert_eq!(
                syllable_to_nibble(syl),
                Some(expected as u8),
                "{syl:?} should decode to {expected}"
            );
        }
    }

    #[test]
    fn syllable_to_nibble_invalid() {
        assert_eq!(syllable_to_nibble("XX"), None);
        assert_eq!(syllable_to_nibble("A"), None);
        assert_eq!(syllable_to_nibble("JOX"), None); // too long
        assert_eq!(syllable_to_nibble(""), None);
        assert_eq!(syllable_to_nibble("ZO"), None); // Z is not a valid consonant
        assert_eq!(syllable_to_nibble("JA"), None); // A is not a valid vowel
    }

    // --- byte_to_word / word_to_byte ---

    #[test]
    fn byte_to_word_known_values() {
        assert_eq!(byte_to_word(0x00), "'O'O");
        assert_eq!(byte_to_word(0x92), "KU'E");
        assert_eq!(byte_to_word(0xFF), "VIVI");
    }

    #[test]
    fn byte_word_roundtrip_all_256() {
        for b in 0..=255u8 {
            let word = byte_to_word(b);
            assert_eq!(
                word_to_byte(&word),
                Some(b),
                "roundtrip failed for byte 0x{b:02X} (word {word:?})"
            );
        }
    }

    #[test]
    fn word_to_byte_invalid() {
        assert_eq!(word_to_byte("ABCD"), None);
        assert_eq!(word_to_byte("JO"), None); // too short
        assert_eq!(word_to_byte("JOJOJO"), None); // too long
        assert_eq!(word_to_byte(""), None);
    }

    // --- encode ---

    #[test]
    fn encode_hello() {
        let data = b"hello";
        assert_eq!(encode(data), "JEKO JEJU JEVO JEVO JEVI");
    }

    #[test]
    fn encode_empty() {
        assert_eq!(encode(b""), "");
    }

    #[test]
    fn encode_wraps_at_8_words() {
        // 9 bytes -> 9 words -> 8 on first line, 1 on second
        let data: Vec<u8> = (0..9).collect();
        let encoded = encode(&data);
        let lines: Vec<&str> = encoded.lines().collect();
        assert_eq!(lines.len(), 2, "9 words should produce 2 lines");

        // First line has 8 words
        let first_words: Vec<&str> = lines[0].split(' ').collect();
        assert_eq!(first_words.len(), 8);

        // Second line has 1 word
        let second_words: Vec<&str> = lines[1].split(' ').collect();
        assert_eq!(second_words.len(), 1);
    }

    #[test]
    fn encode_exactly_8_words_no_trailing_newline() {
        let data: Vec<u8> = (0..8).collect();
        let encoded = encode(&data);
        assert!(
            !encoded.contains('\n'),
            "exactly 8 words should be one line"
        );
    }

    #[test]
    fn encode_16_words_two_lines() {
        let data: Vec<u8> = (0..16).collect();
        let encoded = encode(&data);
        let lines: Vec<&str> = encoded.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let words: Vec<&str> = line.split(' ').collect();
            assert_eq!(words.len(), 8);
        }
    }

    // --- decode ---

    #[test]
    fn decode_roundtrip() {
        let data = b"hello, world!";
        let encoded = encode(data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn decode_roundtrip_all_256() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = encode(&data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
        assert_eq!(decode("   ").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_invalid_word() {
        let err = decode("JOJO XXXX").unwrap_err();
        assert_eq!(err, DecodeError::InvalidWord("XXXX".to_string()));
    }

    #[test]
    fn decode_error_display() {
        let err = DecodeError::InvalidWord("BLAH".to_string());
        let msg = format!("{err}");
        assert!(
            msg.contains("BLAH"),
            "display should include the invalid word"
        );
    }

    #[test]
    fn decode_handles_multiline() {
        // Build a multiline-encoded string and verify decode handles newlines
        let data: Vec<u8> = (0..16).collect();
        let encoded = encode(&data);
        assert!(encoded.contains('\n'));
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
