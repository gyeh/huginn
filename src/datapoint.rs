use std::collections::HashMap;
use crate::ds_type::DsType;
use crate::item_id::ItemId;
use crate::time_seq::{TimeSeq};
use crate::time_series::TimeSeries;

/// Time series with a single value.
///
/// # Fields
/// * `tags` - Metadata for identifying the datapoint.
/// * `timestamp` - Timestamp for the data point. The time is the end of an interval that
///                 starts at `timestamp - step`.
/// * `value` - Value for the interval.
/// * `step` - Step size for the datapoint. Defaults to the configured step size for the service.
#[derive(Debug, Clone, PartialEq)]
pub struct Datapoint {
    pub tags: HashMap<String, String>,
    pub timestamp: i64,
    pub value: f64,
    pub step: i64,
}

impl Datapoint {
    /// Default step size (would come from configuration)
    const DEFAULT_STEP: i64 = 60000; // Example: 60 seconds in milliseconds

    /// Create a new Datapoint with default step size
    pub fn new(tags: HashMap<String, String>, timestamp: i64, value: f64) -> Result<Self, String> {
        Self::with_step(tags, timestamp, value, Self::DEFAULT_STEP)
    }

    /// Create a new Datapoint with custom step size
    pub fn with_step(
        tags: HashMap<String, String>,
        timestamp: i64,
        value: f64,
        step: i64
    ) -> Result<Self, String> {
        // Validation (equivalent to Scala's require)
        if timestamp < 0 {
            return Err(format!("invalid timestamp: {}", timestamp));
        }

        Ok(Datapoint {
            tags,
            timestamp,
            value,
            step,
        })
    }

    /// Convert to tuple representation
    pub fn to_tuple(&self) -> DatapointTuple {
        DatapointTuple {
            id: self.id(),
            tags: self.tags.clone(),
            timestamp: self.timestamp,
            value: self.value,
        }
    }
}

impl TimeSeries for Datapoint {
    fn label(&self) -> &str {
        &self.tags.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn data(&self) -> &dyn TimeSeq {
        self
    }

    fn clone_box(&self) -> Box<dyn TimeSeries> {
        Box::new(self.clone())
    }
}

impl TimeSeq for Datapoint {
    fn apply(&self, t: i64) -> f64 {
        if t == self.timestamp {
            self.value
        } else {
            f64::NAN
        }
    }

    fn ds_type(&self) -> DsType {
        DsType::from_tags(&self.tags)
    }

    fn step(&self) -> i64 {
        self.step
    }

    fn clone_box(&self) -> Box<dyn TimeSeq> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatapointTuple {
    pub id: ItemId,
    pub tags: HashMap<String, String>,
    pub timestamp: i64,
    pub value: f64,
}

impl DatapointTuple {
    pub fn new(id: ItemId, tags: HashMap<String, String>, timestamp: i64, value: f64) -> Self {
        Self {
            id,
            tags,
            timestamp,
            value,
        }
    }

    /// Converts this DatapointTuple to a Datapoint
    pub fn to_datapoint(&self) -> Result<Datapoint, String> {
        Datapoint::new(self.tags.clone(), self.timestamp, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datapoint_creation() {
        let mut tags = HashMap::new();
        tags.insert("name".to_string(), "cpu.usage".to_string());
        tags.insert("host".to_string(), "server1".to_string());

        let datapoint = Datapoint::new(tags.clone(), 1234567890, 85.5).unwrap();

        assert_eq!(datapoint.tags, tags);
        assert_eq!(datapoint.timestamp, 1234567890);
        assert_eq!(datapoint.value, 85.5);
        assert_eq!(datapoint.step, Datapoint::DEFAULT_STEP);
    }

    #[test]
    fn test_invalid_timestamp() {
        let tags = HashMap::new();
        let result = Datapoint::new(tags, -1, 10.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid timestamp"));
    }

    #[test]
    fn test_apply_function() {
        let tags = HashMap::new();
        let datapoint = Datapoint::new(tags, 1000, 42.0).unwrap();

        assert_eq!(datapoint.apply(1000), 42.0);
        assert!(datapoint.apply(999).is_nan());
        assert!(datapoint.apply(1001).is_nan());
    }
}