use std::collections::HashMap;
use crate::item_id::ItemId;
use crate::time_seq::{BinaryOpTimeSeq, OffsetTimeSeq, TimeSeq, UnaryOpTimeSeq};

// TimeSeries module
pub mod time_series {
    use crate::query::Query;
    use crate::time_seq::{DsType, FunctionTimeSeq};
    use super::*;

    lazy_static::lazy_static! {
        static ref NO_DATA_TAGS: HashMap<String, String> = {
            let mut tags = HashMap::new();
            tags.insert("name".to_string(), "NO_DATA".to_string());
            tags
        };
        static ref NO_DATA_ID: ItemId = ItemId::from_bytes("NO_DATA".as_ref()).unwrap();
    }

    /// Create a time series with all NaN values representing no data
    pub fn no_data(step: i64) -> Box<dyn TimeSeries> {
        let data = FunctionTimeSeq::new(DsType::Gauge, step, |_| f64::NAN);
        Box::new(BasicTimeSeries::new(
            NO_DATA_ID.clone(),
            NO_DATA_TAGS.clone(),
            "NO DATA".to_string(),
            Box::new(data),
        ))
    }

    /// Create a time series with all NaN values for a specific query
    pub fn no_data_for_query(query: &Query, step: i64) -> Box<dyn TimeSeries> {
        let tags = if query.tags().is_empty() {
            NO_DATA_TAGS.clone()
        } else {
            query.tags().clone()
        };
        let data = FunctionTimeSeq::new(DsType::Gauge, step, |_| f64::NAN);
        Box::new(LazyTimeSeries::new(
            tags,
            "NO DATA".to_string(),
            Box::new(data),
        ))
    }

    /// Create a time series with tags and data
    pub fn create(tags: HashMap<String, String>, data: Box<dyn TimeSeq>) -> Box<dyn TimeSeries> {
        Box::new(LazyTimeSeries::new(tags.clone(), to_label(&tags), data))
    }

    /// Create a time series with tags, label, and data
    pub fn create_with_label(
        tags: HashMap<String, String>,
        label: String,
        data: Box<dyn TimeSeq>,
    ) -> Box<dyn TimeSeries> {
        Box::new(LazyTimeSeries::new(tags, label, data))
    }

    /// Convert tags to a label string
    pub fn to_label(tags: &HashMap<String, String>) -> String {
        if tags.is_empty() {
            "NO TAGS".to_string()
        } else {
            let mut keys: Vec<_> = tags.keys().cloned().collect();
            keys.sort();
            to_label_with_keys(&keys, tags)
        }
    }

    /// Convert tags to a label string with specific key ordering
    pub fn to_label_with_keys(keys: &[String], tags: &HashMap<String, String>) -> String {
        let labels: Vec<String> = keys
            .iter()
            .map(|k| format!("{}={}", k, tags.get(k).unwrap_or(&"NULL".to_string())))
            .collect();
        labels.join(", ")
    }
}

// Main TimeSeries trait
pub trait TimeSeries {
    fn label(&self) -> &str;
    fn data(&self) -> &dyn TimeSeq;

    fn unary_op(&self, label_fmt: &str, f: Box<dyn Fn(f64) -> f64>) -> Box<dyn TimeSeries> {
        let new_label = label_fmt.replace("{}", self.label());
        let new_data = Box::new(UnaryOpTimeSeq::new(self.data().clone_box(), f));
        Box::new(LazyTimeSeries::new(
            self.tags().clone(),
            new_label,
            new_data,
        ))
    }

    fn binary_op(
        &self,
        ts: &dyn TimeSeries,
        label_fmt: &str,
        f: Box<dyn Fn(f64, f64) -> f64>,
    ) -> Box<dyn TimeSeries> {
        let new_label = label_fmt
            .replacen("{}", self.label(), 1)
            .replacen("{}", ts.label(), 1);
        let new_data = Box::new(BinaryOpTimeSeq::new(
            self.data().clone_box(),
            ts.data().clone_box(),
            f,
        ));
        Box::new(LazyTimeSeries::new(
            self.tags().clone(),
            new_label,
            new_data,
        ))
    }

    fn with_tags(&self, tags: HashMap<String, String>) -> Box<dyn TimeSeries> {
        Box::new(LazyTimeSeries::new(
            tags,
            self.label().to_string(),
            self.data().clone_box(),
        ))
    }

    fn with_label(&self, s: &str) -> Box<dyn TimeSeries> {
        if s.is_empty() {
            self.clone_box()
        } else {
            Box::new(LazyTimeSeries::new(
                self.tags().clone(),
                s.to_string(),
                self.data().clone_box(),
            ))
        }
    }

    fn map_time_seq<F>(&self, f: F) -> Box<dyn TimeSeries>
    where
        F: FnOnce(&dyn TimeSeq) -> Box<dyn TimeSeq>,
    {
        Box::new(LazyTimeSeries::new(
            self.tags().clone(),
            self.label().to_string(),
            f(self.data()),
        ))
    }

    fn offset(&self, dur: i64) -> Box<dyn TimeSeries> {
        Box::new(LazyTimeSeries::new(
            self.tags().clone(),
            self.label().to_string(),
            Box::new(OffsetTimeSeq::new(self.data().clone_box(), dur)),
        ))
    }

    fn clone_box(&self) -> Box<dyn TimeSeries>;
}

// BasicTimeSeries implementation
pub struct BasicTimeSeries {
    id: ItemId,
    tags: HashMap<String, String>,
    label: String,
    data: Box<dyn TimeSeq>,
}

impl BasicTimeSeries {
    pub fn new(id: ItemId, tags: HashMap<String, String>, label: String, data: Box<dyn TimeSeq>) -> Self {
        BasicTimeSeries { id, tags, label, data }
    }
}

impl TimeSeries for BasicTimeSeries {
    fn label(&self) -> &str {
        &self.label
    }

    fn data(&self) -> &dyn TimeSeq {
        &*self.data
    }

    fn clone_box(&self) -> Box<dyn TimeSeries> {
        Box::new(BasicTimeSeries::new(
            self.id.clone(),
            self.tags.clone(),
            self.label.clone(),
            self.data.clone_box(),
        ))
    }
}

// LazyTimeSeries implementation
pub struct LazyTimeSeries {
    tags: HashMap<String, String>,
    label: String,
    data: Box<dyn TimeSeq>,
    id: std::sync::OnceLock<ItemId>,
}

impl LazyTimeSeries {
    pub fn new(tags: HashMap<String, String>, label: String, data: Box<dyn TimeSeq>) -> Self {
        LazyTimeSeries {
            tags,
            label,
            data,
            id: std::sync::OnceLock::new(),
        }
    }
}

impl TimeSeries for LazyTimeSeries {
    fn label(&self) -> &str {
        &self.label
    }

    fn data(&self) -> &dyn TimeSeq {
        &*self.data
    }

    fn clone_box(&self) -> Box<dyn TimeSeries> {
        Box::new(LazyTimeSeries::new(
            self.tags.clone(),
            self.label.clone(),
            self.data.clone_box(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::time_seq::{ArrayTimeSeq, DsType};
    use super::*;

    #[test]
    fn test_no_data_series() {
        let ts = time_series::no_data(60);
        assert_eq!(ts.label(), "NO DATA");
        assert!(ts.data().apply(0).is_nan());
        assert!(ts.data().apply(60).is_nan());
    }

    #[test]
    fn test_create_time_series() {
        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "server1".to_string());
        tags.insert("metric".to_string(), "cpu".to_string());

        let data = vec![1.0, 2.0, 3.0];
        let seq = Box::new(ArrayTimeSeq::new(DsType::Gauge, 0, 60, data));
        let ts = time_series::create(tags.clone(), seq);

        assert!(ts.label().contains("host=server1"));
        assert!(ts.label().contains("metric=cpu"));
        assert_eq!(ts.data().apply(0), 1.0);
    }
}