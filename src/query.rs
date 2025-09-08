use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};

/// Wrapper for regex pattern matching with caching
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternMatcher {
    pattern: RegexWrapper,
    ignore_case: bool,
}

impl PatternMatcher {
    fn compile(pattern: &str) -> Self {
        Self {
            pattern: crate::query::RegexWrapper {regex: Regex::new(pattern).expect("Invalid regex pattern")},
            ignore_case: false,
        }
    }

    fn ignore_case(mut self) -> Self {
        // For simplicity, we'd need to recompile with (?i) flag
        let pattern_str = format!("(?i){}", self.pattern.regex.as_str());
        self.pattern = RegexWrapper {regex: Regex::new(&pattern_str).expect("Invalid regex pattern")};
        self.ignore_case = true;
        self
    }

    fn matches(&self, text: &str) -> bool {
        self.pattern.regex.is_match(text)
    }
}

struct RegexWrapper {
    regex: Regex,
}

impl PartialEq for RegexWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.regex.as_str() == other.regex.as_str()
    }
}

impl Eq for RegexWrapper {}

impl Clone for RegexWrapper {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl Debug for RegexWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

/// Main Query trait implemented as an enum in Rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Query {
    True,
    False,
    HasKey(String),
    Equal(String, String),
    LessThan(String, String),
    LessThanEqual(String, String),
    GreaterThan(String, String),
    GreaterThanEqual(String, String),
    Regex(String, String, PatternMatcher),
    RegexIgnoreCase(String, String, PatternMatcher),
    In(String, Vec<String>),
    And(Box<Query>, Box<Query>),
    Or(Box<Query>, Box<Query>),
    Not(Box<Query>),
}

impl Hash for Query {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Query::True => 0u8.hash(state),
            Query::False => 1u8.hash(state),
            Query::HasKey(k) => {
                2u8.hash(state);
                k.hash(state);
            }
            Query::Equal(k, v) => {
                3u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::LessThan(k, v) => {
                4u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::LessThanEqual(k, v) => {
                5u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::GreaterThan(k, v) => {
                6u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::GreaterThanEqual(k, v) => {
                7u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::Regex(k, v, _) => {
                8u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::RegexIgnoreCase(k, v, _) => {
                9u8.hash(state);
                k.hash(state);
                v.hash(state);
            }
            Query::In(k, vs) => {
                10u8.hash(state);
                k.hash(state);
                vs.hash(state);
            }
            Query::And(q1, q2) => {
                11u8.hash(state);
                q1.hash(state);
                q2.hash(state);
            }
            Query::Or(q1, q2) => {
                12u8.hash(state);
                q1.hash(state);
                q2.hash(state);
            }
            Query::Not(q) => {
                13u8.hash(state);
                q.hash(state);
            }
        }
    }
}

impl Query {
    /// Returns true if the query expression matches the tags provided by the function
    pub fn matches_fn<F>(&self, tags: F) -> bool
    where
        F: Fn(&str) -> Option<&str>,
    {
        match self {
            Query::True => true,
            Query::False => false,
            Query::HasKey(k) => tags(k).is_some(),
            Query::Equal(k, v) => tags(k).map_or(false, |tag_v| tag_v == v),
            Query::LessThan(k, v) => tags(k).map_or(false, |tag_v| tag_v < v),
            Query::LessThanEqual(k, v) => tags(k).map_or(false, |tag_v| tag_v <= v),
            Query::GreaterThan(k, v) => tags(k).map_or(false, |tag_v| tag_v > v),
            Query::GreaterThanEqual(k, v) => tags(k).map_or(false, |tag_v| tag_v >= v),
            Query::Regex(k, _, pattern) => tags(k).map_or(false, |tag_v| pattern.matches(tag_v)),
            Query::RegexIgnoreCase(k, _, pattern) => {
                tags(k).map_or(false, |tag_v| pattern.matches(tag_v))
            }
            Query::In(k, vs) => tags(k).map_or(false, |tag_v| vs.contains(&tag_v.to_string())),
            Query::And(q1, q2) => q1.matches_fn(&tags) && q2.matches_fn(&tags),
            Query::Or(q1, q2) => q1.matches_fn(&tags) || q2.matches_fn(&tags),
            Query::Not(q) => !q.matches_fn(&tags),
        }
    }

    /// Returns true if the query expression matches the tags
    pub fn matches(&self, tags: &HashMap<String, String>) -> bool {
        match self {
            Query::True => true,
            Query::False => false,
            Query::HasKey(k) => tags.contains_key(k),
            Query::Equal(k, v) => tags.get(k).map_or(false, |tag_v| tag_v == v),
            Query::LessThan(k, v) => tags.get(k).map_or(false, |tag_v| tag_v < v),
            Query::LessThanEqual(k, v) => tags.get(k).map_or(false, |tag_v| tag_v <= v),
            Query::GreaterThan(k, v) => tags.get(k).map_or(false, |tag_v| tag_v > v),
            Query::GreaterThanEqual(k, v) => tags.get(k).map_or(false, |tag_v| tag_v >= v),
            Query::Regex(k, _, pattern) => tags.get(k).map_or(false, |tag_v| pattern.matches(tag_v)),
            Query::RegexIgnoreCase(k, _, pattern) => {
                tags.get(k).map_or(false, |tag_v| pattern.matches(tag_v))
            }
            Query::In(k, vs) => tags.get(k).map_or(false, |tag_v| vs.contains(tag_v)),
            Query::And(q1, q2) => q1.matches(tags) && q2.matches(tags),
            Query::Or(q1, q2) => q1.matches(tags) || q2.matches(tags),
            Query::Not(q) => !q.matches(tags),
        }
    }

    /// Returns true if the query matches for any of the items in the list
    pub fn matches_any(&self, tags: &HashMap<String, Vec<String>>) -> bool {
        match self {
            Query::True => true,
            Query::False => false,
            Query::HasKey(k) => tags.contains_key(k),
            Query::Equal(k, v) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| tag_v == v)),
            Query::LessThan(k, v) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| tag_v < v)),
            Query::LessThanEqual(k, v) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| tag_v <= v)),
            Query::GreaterThan(k, v) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| tag_v > v)),
            Query::GreaterThanEqual(k, v) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| tag_v >= v)),
            Query::Regex(k, _, pattern) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| pattern.matches(tag_v))),
            Query::RegexIgnoreCase(k, _, pattern) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| pattern.matches(tag_v))),
            Query::In(k, vs) => tags
                .get(k)
                .map_or(false, |tag_vs| tag_vs.iter().any(|tag_v| vs.contains(tag_v))),
            Query::And(q1, q2) => q1.matches_any(tags) && q2.matches_any(tags),
            Query::Or(q1, q2) => q1.matches_any(tags) || q2.matches_any(tags),
            Query::Not(q) => !q.matches_any(tags),
        }
    }

    /// Returns true if the query expression could match if additional tags were added
    pub fn could_match(&self, tags: &HashMap<String, String>) -> bool {
        match self {
            Query::True => true,
            Query::False => false,
            Query::HasKey(_) => true,
            Query::Equal(k, v) => tags.get(k).map_or(true, |tag_v| tag_v == v),
            Query::LessThan(k, v) => tags.get(k).map_or(true, |tag_v| tag_v < v),
            Query::LessThanEqual(k, v) => tags.get(k).map_or(true, |tag_v| tag_v <= v),
            Query::GreaterThan(k, v) => tags.get(k).map_or(true, |tag_v| tag_v > v),
            Query::GreaterThanEqual(k, v) => tags.get(k).map_or(true, |tag_v| tag_v >= v),
            Query::Regex(k, _, pattern) => {
                tags.get(k).map_or(true, |tag_v| pattern.matches(tag_v))
            }
            Query::RegexIgnoreCase(k, _, pattern) => {
                tags.get(k).map_or(true, |tag_v| pattern.matches(tag_v))
            }
            Query::In(k, vs) => tags.get(k).map_or(true, |tag_v| vs.contains(tag_v)),
            Query::And(q1, q2) => q1.could_match(tags) && q2.could_match(tags),
            Query::Or(q1, q2) => q1.could_match(tags) || q2.could_match(tags),
            Query::Not(q) => !q.matches(tags),
        }
    }

    /// Returns a string that summarizes the query expression in a human readable format
    pub fn label_string(&self) -> String {
        match self {
            Query::True => "true".to_string(),
            Query::False => "false".to_string(),
            Query::HasKey(k) => format!("has({})", k),
            Query::Equal(k, v) => format!("{}={}", k, v),
            Query::LessThan(k, v) => format!("{}<{}", k, v),
            Query::LessThanEqual(k, v) => format!("{}<={}", k, v),
            Query::GreaterThan(k, v) => format!("{}>{}", k, v),
            Query::GreaterThanEqual(k, v) => format!("{}>={}", k, v),
            Query::Regex(k, v, _) => format!("{}~/^{}/", k, v),
            Query::RegexIgnoreCase(k, v, _) => format!("{}~/^{}/i", k, v),
            Query::In(k, vs) => format!("{} in ({})", k, vs.join(",")),
            Query::And(q1, q2) => format!("({}) and ({})", q1.label_string(), q2.label_string()),
            Query::Or(q1, q2) => format!("({}) or ({})", q1.label_string(), q2.label_string()),
            Query::Not(q) => format!("not({})", q.label_string()),
        }
    }

    pub fn and(self, query: Query) -> Query {
        match query {
            Query::True => self,
            Query::False => Query::False,
            q => Query::And(Box::new(self), Box::new(q)),
        }
    }

    pub fn or(self, query: Query) -> Query {
        match query {
            Query::True => Query::True,
            Query::False => self,
            q => Query::Or(Box::new(self), Box::new(q)),
        }
    }

    pub fn not(self) -> Query {
        Query::Not(Box::new(self))
    }

    /// Helper constructor for regex queries
    pub fn regex(k: String, v: String) -> Query {
        let pattern = PatternMatcher::compile(&format!("^{}", v));
        Query::Regex(k, v, pattern)
    }

    /// Helper constructor for case-insensitive regex queries
    pub fn regex_ignore_case(k: String, v: String) -> Query {
        let pattern = PatternMatcher::compile(&format!("^{}", v)).ignore_case();
        Query::RegexIgnoreCase(k, v, pattern)
    }
}

/// Query utility functions
impl Query {
    /// Return the set of keys explicitly referenced in the query
    pub fn exact_keys(&self) -> HashSet<String> {
        match self {
            Query::And(q1, q2) => {
                let mut keys = q1.exact_keys();
                keys.extend(q2.exact_keys());
                keys
            }
            Query::Or(_, _) => HashSet::new(),
            Query::Not(_) => HashSet::new(),
            Query::Equal(k, _) => {
                let mut set = HashSet::new();
                set.insert(k.clone());
                set
            }
            _ => HashSet::new(),
        }
    }

    /// Return the set of keys referenced in the query
    pub fn all_keys(&self) -> HashSet<String> {
        match self {
            Query::And(q1, q2) | Query::Or(q1, q2) => {
                let mut keys = q1.all_keys();
                keys.extend(q2.all_keys());
                keys
            }
            Query::Not(q) => q.all_keys(),
            Query::HasKey(k)
            | Query::Equal(k, _)
            | Query::LessThan(k, _)
            | Query::LessThanEqual(k, _)
            | Query::GreaterThan(k, _)
            | Query::GreaterThanEqual(k, _)
            | Query::Regex(k, _, _)
            | Query::RegexIgnoreCase(k, _, _)
            | Query::In(k, _) => {
                let mut set = HashSet::new();
                set.insert(k.clone());
                set
            }
            _ => HashSet::new(),
        }
    }

    /// Extract a set of tags for the query based on the :eq clauses
    pub fn tags(&self) -> HashMap<String, String> {
        match self {
            Query::And(q1, q2) => {
                let mut tags = q1.tags();
                tags.extend(q2.tags());
                tags
            }
            Query::Equal(k, v) => {
                let mut map = HashMap::new();
                map.insert(k.clone(), v.clone());
                map
            }
            _ => HashMap::new(),
        }
    }

    /// Converts the input query into conjunctive normal form
    pub fn cnf(&self) -> Query {
        let cnf_list = self.cnf_list();
        cnf_list
            .into_iter()
            .reduce(|q1, q2| Query::And(Box::new(q1), Box::new(q2)))
            .unwrap_or(Query::True)
    }

    /// Converts the input query into a list of sub-queries that should be ANDed together
    pub fn cnf_list(&self) -> Vec<Query> {
        match self {
            Query::And(q1, q2) => {
                let mut list = q1.cnf_list();
                list.extend(q2.cnf_list());
                list
            }
            Query::Or(q1, q2) => cross_or(&q1.cnf_list(), &q2.cnf_list()),
            Query::Not(q) => match q.as_ref() {
                Query::And(q1, q2) => {
                    Query::Or(Box::new(q1.as_ref().clone().not()), Box::new(q2.as_ref().clone().not()))
                        .cnf_list()
                }
                Query::Or(q1, q2) => {
                    let mut list = q1.as_ref().clone().not().cnf_list();
                    list.extend(q2.as_ref().clone().not().cnf_list());
                    list
                }
                Query::Not(inner) => vec![inner.as_ref().clone()],
                _ => vec![self.clone()],
            },
            q => vec![q.clone()],
        }
    }

    /// Converts the input query into disjunctive normal form
    pub fn dnf(&self) -> Query {
        let dnf_list = self.dnf_list();
        dnf_list
            .into_iter()
            .reduce(|q1, q2| Query::Or(Box::new(q1), Box::new(q2)))
            .unwrap_or(Query::False)
    }

    /// Converts the input query into a list of sub-queries that should be ORed together
    pub fn dnf_list(&self) -> Vec<Query> {
        match self {
            Query::And(q1, q2) => cross_and(&q1.dnf_list(), &q2.dnf_list()),
            Query::Or(q1, q2) => {
                let mut list = q1.dnf_list();
                list.extend(q2.dnf_list());
                list
            }
            Query::Not(q) => match q.as_ref() {
                Query::And(q1, q2) => {
                    let mut list = q1.as_ref().clone().not().dnf_list();
                    list.extend(q2.as_ref().clone().not().dnf_list());
                    list
                }
                Query::Or(q1, q2) => {
                    Query::And(
                        Box::new(q1.as_ref().clone().not()),
                        Box::new(q2.as_ref().clone().not()),
                    )
                        .dnf_list()
                }
                Query::Not(inner) => vec![inner.as_ref().clone()],
                _ => vec![self.clone()],
            },
            q => vec![q.clone()],
        }
    }

    /// Split :in queries into a list of queries using :eq
    pub fn expand_in_clauses(&self, limit: usize) -> Vec<Query> {
        match self {
            Query::And(q1, q2) => {
                let mut result = Vec::new();
                for a in q1.expand_in_clauses(limit) {
                    for b in q2.expand_in_clauses(limit) {
                        result.push(Query::And(Box::new(a.clone()), Box::new(b.clone())));
                    }
                }
                result
            }
            Query::In(k, vs) if vs.len() <= limit => vs
                .iter()
                .map(|v| Query::Equal(k.clone(), v.clone()))
                .collect(),
            _ => vec![self.clone()],
        }
    }

    /// Simplify a query expression that contains True and False constants
    pub fn simplify(&self, ignore: bool) -> Query {
        let new_query = match self {
            Query::And(q1, q2) => match (q1.as_ref(), q2.as_ref()) {
                (Query::True, q) => q.simplify(ignore),
                (q, Query::True) => q.simplify(ignore),
                (Query::False, _) | (_, Query::False) => Query::False,
                _ => Query::And(
                    Box::new(q1.simplify(ignore)),
                    Box::new(q2.simplify(ignore)),
                ),
            },

            Query::Or(q1, q2) => match (q1.as_ref(), q2.as_ref()) {
                (Query::True, _) | (_, Query::True) => Query::True,
                (Query::False, q) => q.simplify(ignore),
                (q, Query::False) => q.simplify(ignore),
                _ => Query::Or(Box::new(q1.simplify(ignore)), Box::new(q2.simplify(ignore))),
            },

            Query::Not(q) => match q.as_ref() {
                Query::True => {
                    if ignore {
                        Query::True
                    } else {
                        Query::False
                    }
                }
                Query::False => Query::True,
                _ => Query::Not(Box::new(q.simplify(ignore))),
            },

            q => q.clone(),
        };

        if new_query != *self {
            new_query.simplify(ignore)
        } else {
            new_query
        }
    }

    /// Convert In query to a sequence of OR'd together equal queries
    pub fn to_or_query(&self) -> Query {
        match self {
            Query::In(k, vs) => {
                if vs.is_empty() {
                    Query::False
                } else {
                    vs.iter()
                        .skip(1)
                        .fold(Query::Equal(k.clone(), vs[0].clone()), |acc, v| {
                            Query::Or(Box::new(acc), Box::new(Query::Equal(k.clone(), v.clone())))
                        })
                }
            }
            _ => self.clone(),
        }
    }
}

// Helper functions
fn cross_or(qs1: &[Query], qs2: &[Query]) -> Vec<Query> {
    let mut result = Vec::new();
    for q1 in qs1 {
        for q2 in qs2 {
            result.push(Query::Or(Box::new(q1.clone()), Box::new(q2.clone())));
        }
    }
    result
}

fn cross_and(qs1: &[Query], qs2: &[Query]) -> Vec<Query> {
    let mut result = Vec::new();
    for q1 in qs1 {
        for q2 in qs2 {
            result.push(Query::And(Box::new(q1.clone()), Box::new(q2.clone())));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_queries() {
        let mut tags = HashMap::new();
        tags.insert("name".to_string(), "test".to_string());
        tags.insert("value".to_string(), "42".to_string());

        assert!(Query::True.matches(&tags));
        assert!(!Query::False.matches(&tags));
        assert!(Query::HasKey("name".to_string()).matches(&tags));
        assert!(!Query::HasKey("missing".to_string()).matches(&tags));
        assert!(Query::Equal("name".to_string(), "test".to_string()).matches(&tags));
        assert!(!Query::Equal("name".to_string(), "wrong".to_string()).matches(&tags));
    }

    #[test]
    fn test_and_or_not() {
        let mut tags = HashMap::new();
        tags.insert("a".to_string(), "1".to_string());
        tags.insert("b".to_string(), "2".to_string());

        let q1 = Query::Equal("a".to_string(), "1".to_string());
        let q2 = Query::Equal("b".to_string(), "2".to_string());
        let q3 = Query::Equal("c".to_string(), "3".to_string());

        assert!(q1.clone().and(q2.clone()).matches(&tags));
        assert!(!q1.clone().and(q3.clone()).matches(&tags));
        assert!(q1.clone().or(q3.clone()).matches(&tags));
        assert!(!q1.not().matches(&tags));
    }

    #[test]
    fn test_simplify() {
        let q1 = Query::True.and(Query::Equal("a".to_string(), "1".to_string()));
        let simplified = q1.simplify(false);
        assert_eq!(simplified, Query::Equal("a".to_string(), "1".to_string()));

        let q2 = Query::False.or(Query::Equal("a".to_string(), "1".to_string()));
        let simplified = q2.simplify(false);
        assert_eq!(simplified, Query::Equal("a".to_string(), "1".to_string()));
    }
}