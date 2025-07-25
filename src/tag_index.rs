use std::collections::{HashMap, HashSet};
use crate::block_store::BlockStoreItem;
use crate::item_id::ItemId;
use crate::query::Query;
use crate::tag::Tag;

#[derive(Debug, Clone)]
pub struct TagQuery {
    pub query: Option<Query>,
    pub key: Option<String>,
    pub offset: String,
    pub limit: usize,
}

impl TagQuery {
    /// Parse the offset string to a tag object
    pub fn offset_tag(&self) -> Tag {
        if let Some(comma_pos) = self.offset.find(',') {
            let (key, value) = self.offset.split_at(comma_pos);
            Tag::with_count(
                key.to_string(),
                value[1..].to_string(), // Skip the comma
                i32::MAX,
            )
        } else {
            Tag::with_count(self.offset.clone(), String::new(), i32::MAX)
        }
    }
}

pub struct SimpleTagIndex {
    items: Vec<BlockStoreItem>,
    /// All item indices
    all: HashSet<usize>,
    /// Index structure: key -> value -> set of item indices
    index: HashMap<String, HashMap<String, HashSet<usize>>>,
}

impl SimpleTagIndex {
    /// Create a new SimpleTagIndex from a slice of BlockStoreItems
    pub fn new(items: Vec<BlockStoreItem>) -> Self {
        let all: HashSet<usize> = (0..items.len()).collect();
        let index = Self::build_index(&items);

        Self { items, all, index }
    }

    /// Build the index from items
    fn build_index(items: &[BlockStoreItem]) -> HashMap<String, HashMap<String, HashSet<usize>>> {
        let mut index = HashMap::new();

        for (i, item) in items.iter().enumerate() {
            for (key, value) in &item.tags {
                let value_map = index.entry(key.clone()).or_insert_with(HashMap::new);
                let item_set = value_map.entry(value.clone()).or_insert_with(HashSet::new);
                item_set.insert(i);
            }
        }

        index
    }

    /// Find item indices matching the query
    fn find_impl(&self, query: &Query) -> HashSet<usize> {
        match query {
            Query::True => self.all.clone(),
            Query::False => HashSet::new(),

            Query::And(q1, q2) => {
                let set1 = self.find_impl(q1);
                let set2 = self.find_impl(q2);
                set1.intersection(&set2).cloned().collect()
            }

            Query::Or(q1, q2) => {
                let set1 = self.find_impl(q1);
                let set2 = self.find_impl(q2);
                set1.union(&set2).cloned().collect()
            }

            Query::Not(q) => {
                let matches = self.find_impl(q);
                self.all.difference(&matches).cloned().collect()
            }

            Query::Equal(k, v) => {
                self.index
                    .get(k)
                    .and_then(|value_map| value_map.get(v))
                    .cloned()
                    .unwrap_or_else(HashSet::new)
            }

            Query::HasKey(k) => {
                self.index
                    .get(k)
                    .map(|value_map| {
                        value_map
                            .values()
                            .fold(HashSet::new(), |mut acc, indices| {
                                acc.extend(indices);
                                acc
                            })
                    })
                    .unwrap_or_else(HashSet::new)
            }

            // For key-value queries (LessThan, GreaterThan, Regex, In, etc.)
            query => {
                if let Some(key) = Self::extract_key(query) {
                    self.index
                        .get(key)
                        .map(|value_map| {
                            value_map
                                .iter()
                                .filter_map(|(value, indices)| {
                                    if Self::check_value(query, key, value) {
                                        Some(indices.clone())
                                    } else {
                                        None
                                    }
                                })
                                .fold(HashSet::new(), |mut acc, indices| {
                                    acc.extend(indices);
                                    acc
                                })
                        })
                        .unwrap_or_else(HashSet::new)
                } else {
                    HashSet::new()
                }
            }
        }
    }

    /// Extract the key from a query if it's a key-value query
    fn extract_key(query: &Query) -> Option<&str> {
        match query {
            Query::Equal(k, _)
            | Query::LessThan(k, _)
            | Query::LessThanEqual(k, _)
            | Query::GreaterThan(k, _)
            | Query::GreaterThanEqual(k, _)
            | Query::Regex(k, _, _)
            | Query::RegexIgnoreCase(k, _, _)
            | Query::In(k, _) => Some(k),
            _ => None,
        }
    }

    /// Check if a value matches the query condition
    fn check_value(query: &Query, key: &str, value: &str) -> bool {
        let tags = HashMap::from([(key.to_string(), value.to_string())]);
        query.matches(&tags)
    }

    /// Find items matching the optional query
    fn find_items_impl(&self, query: Option<&Query>) -> Vec<&BlockStoreItem> {
        match query {
            None => self.items.iter().collect(),
            Some(q) => {
                let indices = self.find_impl(q);
                indices.into_iter().map(|i| &self.items[i]).collect()
            }
        }
    }

    /// Find tags matching the query
    pub fn find_tags(&self, query: &TagQuery) -> Vec<Tag> {
        if query.key.is_none() {
            return Vec::new();
        }

        let matches = self.find_items_impl(query.query.as_ref());

        // Collect unique tags
        let mut unique_tags = HashSet::new();
        for item in matches {
            for (k, v) in &item.tags {
                unique_tags.insert(Tag::new(k.clone(), v.clone()));
            }
        }

        // Filter by key if specified
        let mut filtered: Vec<Tag> = if let Some(key) = &query.key {
            unique_tags.into_iter().filter(|tag| tag.key == *key).collect()
        } else {
            unique_tags.into_iter().collect()
        };

        // Filter by offset
        let offset_tag = query.offset_tag();
        filtered.retain(|tag| tag > &offset_tag);

        // Sort and limit
        filtered.sort();
        filtered.truncate(query.limit);

        filtered
    }

    /// Find keys matching the query
    pub fn find_keys(&self, query: &TagQuery) -> Vec<String> {
        let matches = self.find_items_impl(query.query.as_ref());

        // Collect unique keys
        let mut unique_keys = HashSet::new();
        for item in matches {
            for (k, _) in &item.tags {
                unique_keys.insert(k.clone());
            }
        }

        // Filter, sort, and limit
        let mut keys: Vec<String> = unique_keys
            .into_iter()
            .filter(|k| k > &query.offset)
            .collect();

        keys.sort();
        keys.truncate(query.limit);

        keys
    }

    /// Find values for a specific key matching the query
    pub fn find_values(&self, query: &TagQuery) -> Vec<String> {
        let key = query.key.as_ref().expect("Key must be defined for find_values");

        let matches = self.find_items_impl(query.query.as_ref());

        // Collect unique values for the key
        let mut unique_values = HashSet::new();
        for item in matches {
            if let Some(value) = item.tags.get(key) {
                unique_values.insert(value.clone());
            }
        }

        // Filter, sort, and limit
        let mut values: Vec<String> = unique_values
            .into_iter()
            .filter(|v| v > &query.offset)
            .collect();

        values.sort();
        values.truncate(query.limit);

        values
    }

    /// Find items matching the query
    pub fn find_items(&self, query: &TagQuery) -> Vec<&BlockStoreItem> {
        let mut matches = self.find_items_impl(query.query.as_ref());

        // Filter by offset (using item ID string)
        matches.retain(|item| item.id.to_string() > query.offset);

        // Sort by item ID
        matches.sort_by(|a, b| a.id.cmp(&b.id));

        // Limit results
        matches.truncate(query.limit);

        matches
    }

    /// Get the number of items in the index
    pub fn size(&self) -> usize {
        self.items.len()
    }

    /// Get an iterator over all items
    pub fn iter(&self) -> impl Iterator<Item = &BlockStoreItem> {
        self.items.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_store::MemoryBlockStore;

    fn create_test_item(id: &str, tags: Vec<(&str, &str)>) -> BlockStoreItem {
        let item_id = ItemId::from_hex_string(id).unwrap();
        let tag_map: HashMap<String, String> = tags
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        BlockStoreItem {
            id: item_id,
            tags: tag_map,
            blocks: MemoryBlockStore::new(1000, 60, 5),
        }
    }

    #[test]
    fn test_basic_index_creation() {
        let items = vec![
            create_test_item("0001", vec![("env", "prod"), ("app", "api")]),
            create_test_item("0002", vec![("env", "dev"), ("app", "api")]),
            create_test_item("0003", vec![("env", "prod"), ("app", "web")]),
        ];

        let index = SimpleTagIndex::new(items);
        assert_eq!(index.size(), 3);
    }

    #[test]
    fn test_find_keys() {
        let items = vec![
            create_test_item("0001", vec![("env", "prod"), ("app", "api")]),
            create_test_item("0002", vec![("region", "us-east"), ("app", "api")]),
        ];

        let index = SimpleTagIndex::new(items);
        let query = TagQuery {
            query: None,
            key: None,
            offset: String::new(),
            limit: 10,
        };

        let keys = index.find_keys(&query);
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"app".to_string()));
        assert!(keys.contains(&"env".to_string()));
        assert!(keys.contains(&"region".to_string()));
    }

    #[test]
    fn test_find_values() {
        let items = vec![
            create_test_item("0001", vec![("env", "prod"), ("app", "api")]),
            create_test_item("0002", vec![("env", "dev"), ("app", "api")]),
            create_test_item("0003", vec![("env", "prod"), ("app", "web")]),
        ];

        let index = SimpleTagIndex::new(items);
        let query = TagQuery {
            query: None,
            key: Some("env".to_string()),
            offset: String::new(),
            limit: 10,
        };

        let values = index.find_values(&query);
        assert_eq!(values.len(), 2);
        assert!(values.contains(&"dev".to_string()));
        assert!(values.contains(&"prod".to_string()));
    }

    #[test]
    fn test_find_items_with_query() {
        let items = vec![
            create_test_item("0001", vec![("env", "prod"), ("app", "api")]),
            create_test_item("0002", vec![("env", "dev"), ("app", "api")]),
            create_test_item("0003", vec![("env", "prod"), ("app", "web")]),
        ];

        let index = SimpleTagIndex::new(items);

        // Find items where env=prod
        let query = TagQuery {
            query: Some(Query::Equal("env".to_string(), "prod".to_string())),
            key: None,
            offset: String::new(),
            limit: 10,
        };

        let items = index.find_items(&query);
        assert_eq!(items.len(), 2);

        // Verify both items have env=prod
        for item in items {
            assert_eq!(item.tags.get("env").unwrap(), "prod");
        }
    }

    #[test]
    fn test_complex_query() {
        let items = vec![
            create_test_item("0001", vec![("env", "prod"), ("app", "api"), ("region", "us")]),
            create_test_item("0002", vec![("env", "dev"), ("app", "api"), ("region", "eu")]),
            create_test_item("0003", vec![("env", "prod"), ("app", "web"), ("region", "us")]),
        ];

        let index = SimpleTagIndex::new(items);

        // Find items where env=prod AND app=api
        let q1 = Query::Equal("env".to_string(), "prod".to_string());
        let q2 = Query::Equal("app".to_string(), "api".to_string());
        let query = TagQuery {
            query: Some(Query::And(Box::new(q1), Box::new(q2))),
            key: None,
            offset: String::new(),
            limit: 10,
        };

        let items = index.find_items(&query);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id.to_string(), "00000000000000000000000000000001");
    }
}