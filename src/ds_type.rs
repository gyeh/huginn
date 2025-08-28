use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsType {
    Rate,
    Gauge,
}

impl DsType {
    pub fn from_key(key: &str) -> Self {
        match key {
            "gauge" => DsType::Gauge,
            _ => DsType::Rate, // counter, rate, sum
        }
    }

    pub fn from_tags(tags: &HashMap<String, String>) -> Self {
        // Assuming TagKey::DS_TYPE is defined elsewhere as a constant
        // If not, you can define it as:
        // const DS_TYPE_KEY: &str = "ds_type";

        let key = tags.get(TagKey::DS_TYPE).map(|s| s.as_str()).unwrap_or("rate");
        Self::from_key(key)
    }
}

// If TagKey is a struct with constants, define it like this:
pub struct TagKey;

impl TagKey {
    pub const DS_TYPE: &'static str = "ds_type";
}