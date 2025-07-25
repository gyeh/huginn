/// Represents a key/value pair and its associated count. The count is the number of items that
/// are marked with the tag.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tag {
    /// Key for the tag
    pub key: String,
    /// Value associated with the key
    pub value: String,
    /// Number of items with this tag or -1 if unknown
    pub count: i32,
}

impl Tag {
    /// Creates a new Tag with the given key and value, with count set to -1
    pub fn new(key: String, value: String) -> Self {
        Tag {
            key,
            value,
            count: -1,
        }
    }

    /// Creates a new Tag with the given key, value, and count
    pub fn with_count(key: String, value: String, count: i32) -> Self {
        Tag { key, value, count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tag = Tag::new("env".to_string(), "prod".to_string());
        assert_eq!(tag.key, "env");
        assert_eq!(tag.value, "prod");
        assert_eq!(tag.count, -1);
    }

    #[test]
    fn test_with_count() {
        let tag = Tag::with_count("region".to_string(), "us-east-1".to_string(), 42);
        assert_eq!(tag.key, "region");
        assert_eq!(tag.value, "us-east-1");
        assert_eq!(tag.count, 42);
    }

    #[test]
    fn test_ordering() {
        let tag1 = Tag::with_count("a".to_string(), "1".to_string(), 10);
        let tag2 = Tag::with_count("a".to_string(), "2".to_string(), 5);
        let tag3 = Tag::with_count("b".to_string(), "1".to_string(), 10);
        let tag4 = Tag::with_count("a".to_string(), "1".to_string(), 20);

        // Test ordering by key
        assert!(tag1 < tag3);
        assert!(tag3 > tag1);

        // Test ordering by value (same key)
        assert!(tag1 < tag2);
        assert!(tag2 > tag1);

        // Test ordering by count (same key and value)
        assert!(tag1 < tag4);
        assert!(tag4 > tag1);

        // Test equality
        let tag5 = Tag::with_count("a".to_string(), "1".to_string(), 10);
        assert_eq!(tag1, tag5);
    }

    #[test]
    fn test_comparison_operators() {
        let tag1 = Tag::new("app".to_string(), "api".to_string());
        let tag2 = Tag::new("app".to_string(), "web".to_string());

        assert!(tag1 < tag2);
        assert!(!(tag1 > tag2));
        assert!(tag1 <= tag2);
        assert!(!(tag1 >= tag2));
        assert!(tag1 != tag2);
    }
}