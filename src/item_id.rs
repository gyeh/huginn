use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Represents an identifier for a tagged item. If using for a hash map the
/// bytes used for the id should come from a decent hash function as 4 bytes
/// from the middle are used for the hash code of the id object.
///
/// The data bytes are usually the results of computing a SHA1 hash
/// over a normalized representation of the tags.
#[derive(Clone, Debug)]
pub struct ItemId {
    data: Vec<u8>,
}

impl ItemId {
    /// Create a new id from a vector of bytes.
    pub fn new(data: Vec<u8>) -> Result<Self, String> {
        // Typically it should be 20 bytes for SHA1. Require at least 16 to avoid
        // checks for other operations.
        if data.len() < 16 {
            return Err(format!("ItemId data must be at least 16 bytes, got {}", data.len()));
        }
        Ok(ItemId { data })
    }

    /// Create a new id from a slice of bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        Self::new(data.to_vec())
    }

    /// Create a new id from a hex string. The string should match the output of
    /// the `Display` implementation of an `ItemId`.
    pub fn from_hex_string(hex_str: &str) -> Result<Self, String> {
        // Pad to min size for id. Allows it to work easily with hex strings from number types
        let padded = Self::zero_pad(hex_str, 32);

        if padded.len() % 2 != 0 {
            return Err(format!("Invalid item id string: {}", padded));
        }

        let mut bytes = Vec::with_capacity(padded.len() / 2);

        for chunk in padded.as_bytes().chunks(2) {
            let c1 = Self::hex_to_int(chunk[0] as char)?;
            let c2 = Self::hex_to_int(chunk[1] as char)?;
            let value = (c1 << 4) | c2;
            bytes.push(value as u8);
        }

        Self::new(bytes)
    }

    /// Create a new id from a big integer represented as a hex string.
    pub fn from_big_int_hex(hex_str: &str) -> Result<Self, String> {
        Self::from_hex_string(hex_str)
    }

    /// Convert to a big integer representation (as a hex string).
    pub fn to_big_int_hex(&self) -> String {
        self.to_string()
    }

    /// Get the integer value from the last 4 bytes of the data.
    pub fn int_value(&self) -> u32 {
        let start = self.data.len() - 4;
        u32::from_be_bytes([
            self.data[start],
            self.data[start + 1],
            self.data[start + 2],
            self.data[start + 3],
        ])
    }

    /// Get the raw byte data.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    fn zero_pad(s: &str, min_length: usize) -> String {
        if s.len() >= min_length {
            s.to_string()
        } else {
            format!("{:0>width$}", s, width = min_length)
        }
    }

    fn hex_to_int(c: char) -> Result<u8, String> {
        match c {
            '0'..='9' => Ok((c as u8) - b'0'),
            'a'..='f' => Ok((c as u8) - b'a' + 10),
            'A'..='F' => Ok((c as u8) - b'A' + 10),
            _ => Err(format!("Invalid hex digit: {}", c)),
        }
    }
}

impl Hash for ItemId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Choose middle bytes. The id should be generated using decent hash
        // function so in theory any subset will do. In some cases data is
        // routed based on the prefix or a modulo of the int_value. Choosing
        // bytes toward the middle helps to mitigate that.
        let hash_value = u32::from_be_bytes([
            self.data[12],
            self.data[13],
            self.data[14],
            self.data[15],
        ]);
        hash_value.hash(state);
    }
}

impl PartialEq for ItemId {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for ItemId {}

impl PartialOrd for ItemId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ItemId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}

impl fmt::Display for ItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.data {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_length() {
        let data = vec![0u8; 20]; // SHA1 length
        let item_id = ItemId::new(data).unwrap();
        assert_eq!(item_id.as_bytes().len(), 20);
    }

    #[test]
    fn test_new_invalid_length() {
        let data = vec![0u8; 10]; // Too short
        assert!(ItemId::new(data).is_err());
    }

    #[test]
    fn test_from_hex_string() {
        let hex = "0123456789abcdef0123456789abcdef01234567";
        let item_id = ItemId::from_hex_string(hex).unwrap();
        assert_eq!(item_id.to_string(), hex);
    }

    #[test]
    fn test_hex_padding() {
        let hex = "123";
        let item_id = ItemId::from_hex_string(hex).unwrap();
        // Should be padded to 32 characters
        assert_eq!(item_id.to_string().len(), 32);
    }

    #[test]
    fn test_int_value() {
        let data = vec![0u8; 16];
        let mut item_id_data = data;
        // Set last 4 bytes to represent the number 0x12345678
        item_id_data[12] = 0x12;
        item_id_data[13] = 0x34;
        item_id_data[14] = 0x56;
        item_id_data[15] = 0x78;

        let item_id = ItemId::new(item_id_data).unwrap();
        assert_eq!(item_id.int_value(), 0x12345678);
    }

    #[test]
    fn test_equality() {
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let data2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let data3 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17];

        let item_id1 = ItemId::new(data1).unwrap();
        let item_id2 = ItemId::new(data2).unwrap();
        let item_id3 = ItemId::new(data3).unwrap();

        assert_eq!(item_id1, item_id2);
        assert_ne!(item_id1, item_id3);
    }

    #[test]
    fn test_ordering() {
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        let item_id1 = ItemId::new(data1).unwrap();
        let item_id2 = ItemId::new(data2).unwrap();

        assert!(item_id1 < item_id2);
    }
}