use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;

pub const MAX_SCROLL_SIZE: usize = 4096;

mod hex_btree {
    use super::*;

    pub fn serialize<S>(map: &BTreeMap<Vec<u8>, Scroll>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut ser_map = serializer.serialize_map(Some(map.len()))?;
        for (key, value) in map {
            let hex_key: String = key.iter().map(|b| format!("{:02x}", b)).collect();
            ser_map.serialize_entry(&hex_key, value)?;
        }
        ser_map.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BTreeMap<Vec<u8>, Scroll>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let string_map: BTreeMap<String, Scroll> = BTreeMap::deserialize(deserializer)?;
        let mut result = BTreeMap::new();
        for (hex_key, value) in string_map {
            if !hex_key.is_ascii() {
                return Err(D::Error::custom("non-ASCII hex key"));
            }
            if hex_key.len() % 2 != 0 {
                return Err(D::Error::custom("odd-length hex key"));
            }
            let bytes: Result<Vec<u8>, _> = (0..hex_key.len())
                .step_by(2)
                .map(|i| {
                    u8::from_str_radix(&hex_key[i..i + 2], 16)
                        .map_err(|_| D::Error::custom("invalid hex in key"))
                })
                .collect();
            result.insert(bytes?, value);
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScrollKind {
    Text(String),
    Reference(String),
    Alias(Vec<u8>),
    Action { program: String, args: Vec<u8> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Scroll {
    pub version: u8,
    pub kind: ScrollKind,
    pub title: String,
    pub created_epoch_secs: u64,
    pub updated_epoch_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LookupResult {
    Found(Vec<u8>, Scroll),
    Ambiguous(Vec<(Vec<u8>, String)>),
    NotFound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TomeError {
    ScrollTooLarge(usize),
    SerializationFailed,
}

impl fmt::Display for TomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TomeError::ScrollTooLarge(size) => {
                write!(
                    f,
                    "scroll too large: {} bytes (max {})",
                    size, MAX_SCROLL_SIZE
                )
            }
            TomeError::SerializationFailed => {
                write!(f, "scroll serialization failed")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tome {
    #[serde(with = "hex_btree")]
    entries: BTreeMap<Vec<u8>, Scroll>,
}

impl Tome {
    pub fn new() -> Self {
        Tome {
            entries: BTreeMap::new(),
        }
    }

    /// Insert without size check. Only for test use.
    #[cfg(test)]
    fn insert(&mut self, address: Vec<u8>, scroll: Scroll) {
        self.entries.insert(address, scroll);
    }

    pub fn try_insert(&mut self, address: Vec<u8>, scroll: Scroll) -> Result<(), TomeError> {
        let serialized = serde_json::to_vec(&scroll).map_err(|_| TomeError::SerializationFailed)?;
        let size = serialized.len();
        if size > MAX_SCROLL_SIZE {
            return Err(TomeError::ScrollTooLarge(size));
        }
        self.entries.insert(address, scroll);
        Ok(())
    }

    pub fn remove(&mut self, address: &[u8]) -> bool {
        self.entries.remove(address).is_some()
    }

    pub fn lookup(&self, prefix: &[u8]) -> LookupResult {
        let mut matches: Vec<(Vec<u8>, Scroll)> = Vec::new();

        for (key, scroll) in self.entries.range(prefix.to_vec()..) {
            if key.starts_with(prefix) {
                matches.push((key.clone(), scroll.clone()));
            } else {
                break;
            }
        }

        match matches.len() {
            0 => LookupResult::NotFound,
            1 => {
                let (addr, scroll) = matches.into_iter().next().unwrap();
                LookupResult::Found(addr, scroll)
            }
            _ => LookupResult::Ambiguous(
                matches
                    .into_iter()
                    .map(|(addr, scroll)| (addr, scroll.title))
                    .collect(),
            ),
        }
    }

    pub fn list(&self) -> Vec<(&Vec<u8>, &Scroll)> {
        self.entries.iter().collect()
    }
}

impl Default for Tome {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_scroll(title: &str, text: &str) -> Scroll {
        Scroll {
            version: 1,
            kind: ScrollKind::Text(text.to_string()),
            title: title.to_string(),
            created_epoch_secs: 1000,
            updated_epoch_secs: 1000,
        }
    }

    #[test]
    fn empty_tome_lookup() {
        let tome = Tome::new();
        assert_eq!(tome.lookup(&[0xAA, 0xBB]), LookupResult::NotFound);
    }

    #[test]
    fn insert_and_lookup_exact() {
        let mut tome = Tome::new();
        let addr = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let scroll = text_scroll("greeting", "hello world");
        tome.insert(addr.clone(), scroll.clone());

        match tome.lookup(&addr) {
            LookupResult::Found(found_addr, found_scroll) => {
                assert_eq!(found_addr, addr);
                assert_eq!(found_scroll, scroll);
            }
            other => panic!("expected Found, got {:?}", other),
        }
    }

    #[test]
    fn shortest_prefix_unique() {
        let mut tome = Tome::new();
        let addr = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let scroll = text_scroll("unique", "only entry");
        tome.insert(addr.clone(), scroll.clone());

        // A single-byte prefix should resolve when there's only one entry starting with 0xAA
        match tome.lookup(&[0xAA]) {
            LookupResult::Found(found_addr, found_scroll) => {
                assert_eq!(found_addr, addr);
                assert_eq!(found_scroll, scroll);
            }
            other => panic!("expected Found, got {:?}", other),
        }
    }

    #[test]
    fn ambiguous_prefix() {
        let mut tome = Tome::new();
        let addr1 = vec![0xAA, 0xBB, 0x01];
        let addr2 = vec![0xAA, 0xBB, 0x02];
        tome.insert(addr1.clone(), text_scroll("first", "entry one"));
        tome.insert(addr2.clone(), text_scroll("second", "entry two"));

        match tome.lookup(&[0xAA, 0xBB]) {
            LookupResult::Ambiguous(entries) => {
                assert_eq!(entries.len(), 2);
                assert!(entries.contains(&(addr1, "first".to_string())));
                assert!(entries.contains(&(addr2, "second".to_string())));
            }
            other => panic!("expected Ambiguous, got {:?}", other),
        }
    }

    #[test]
    fn remove_entry() {
        let mut tome = Tome::new();
        let addr = vec![0x01, 0x02];
        tome.insert(addr.clone(), text_scroll("ephemeral", "gone soon"));

        assert!(tome.remove(&addr));
        assert_eq!(tome.lookup(&addr), LookupResult::NotFound);
    }

    #[test]
    fn remove_nonexistent() {
        let mut tome = Tome::new();
        assert!(!tome.remove(&[0xFF, 0xFF]));
    }

    #[test]
    fn list_all() {
        let mut tome = Tome::new();
        tome.insert(vec![0x01], text_scroll("a", "alpha"));
        tome.insert(vec![0x02], text_scroll("b", "beta"));
        tome.insert(vec![0x03], text_scroll("c", "gamma"));

        let all = tome.list();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn serialize_roundtrip() {
        let mut tome = Tome::new();
        let addr = vec![0xCA, 0xFE];
        let scroll = text_scroll("roundtrip", "survives serialization");
        tome.insert(addr.clone(), scroll.clone());

        let json = serde_json::to_string(&tome).expect("serialize");
        let restored: Tome = serde_json::from_str(&json).expect("deserialize");

        match restored.lookup(&addr) {
            LookupResult::Found(found_addr, found_scroll) => {
                assert_eq!(found_addr, addr);
                assert_eq!(found_scroll, scroll);
            }
            other => panic!("expected Found, got {:?}", other),
        }
    }

    #[test]
    fn scroll_size_limit() {
        let mut tome = Tome::new();
        // Create a scroll with text large enough to exceed MAX_SCROLL_SIZE when serialized
        let huge_text = "x".repeat(MAX_SCROLL_SIZE + 1);
        let scroll = text_scroll("huge", &huge_text);

        let result = tome.try_insert(vec![0x01], scroll);
        match result {
            Err(TomeError::ScrollTooLarge(size)) => {
                assert!(size > MAX_SCROLL_SIZE);
            }
            other => panic!("expected ScrollTooLarge error, got {:?}", other),
        }

        // Verify nothing was inserted
        assert_eq!(tome.lookup(&[0x01]), LookupResult::NotFound);
    }

    #[test]
    fn tome_error_display() {
        let err = TomeError::ScrollTooLarge(5000);
        let msg = format!("{}", err);
        assert!(msg.contains("5000"));
        assert!(msg.contains("4096"));
    }

    #[test]
    fn deserialize_rejects_non_ascii_hex_key() {
        // Non-ASCII UTF-8 in a key would panic on byte-level slicing without the ASCII check
        let json = r#"{"entries":{"café":{"version":1,"kind":{"Text":"t"},"title":"t","created_epoch_secs":0,"updated_epoch_secs":0}}}"#;
        let result: Result<Tome, _> = serde_json::from_str(json);
        assert!(result.is_err(), "non-ASCII hex key should be rejected");
    }
}
