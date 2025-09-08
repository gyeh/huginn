//! Helper functions for working with strings.
use chrono::{DateTime, Duration, TimeZone, Utc};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::fmt::Write;

/// Error type for string parsing operations
#[derive(Debug, thiserror::Error)]
pub enum StringError {
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Invalid hex encoding")]
    InvalidHexEncoding,
    #[error("Invalid date: {0}")]
    InvalidDate(String),
    #[error("Invalid duration: {0}")]
    InvalidDuration(String),
    #[error("Invalid color: {0}")]
    InvalidColor(String),
    #[error("Chrono error: {0}")]
    ChronoError(#[from] chrono::ParseError),
}

type Result<T> = std::result::Result<T, StringError>;

lazy_static! {
    /// URL query parameter pattern
    static ref QUERY_PARAM: Regex = Regex::new(r"^([^=]+)=(.*)$").unwrap();

    /// Period following conventions of unix `at` command
    static ref AT_PERIOD: Regex = Regex::new(r"^(\d+)([a-zμ]+)$").unwrap();

    /// Period following the ISO8601 conventions
    static ref ISO_PERIOD: Regex = Regex::new(r"^(P.*)$").unwrap();

    /// Date relative to a given reference point
    static ref RELATIVE_DATE: Regex = Regex::new(r"^([a-z]+)([-+])([^-+]+)$").unwrap();

    /// Named date such as `epoch` or `now`
    static ref NAMED_DATE: Regex = Regex::new(r"^([a-z]+)$").unwrap();

    /// Unix date in seconds since the epoch
    static ref UNIX_DATE: Regex = Regex::new(r"^([0-9]+)$").unwrap();

    /// Unix date with operation
    static ref UNIX_DATE_WITH_OP: Regex = Regex::new(r"^([0-9]+)([-+])([^-+]+)$").unwrap();
}

/// When parsing a timestamp string, timestamps after this point will be treated as
/// milliseconds rather than seconds.
const SECONDS_CUTOFF: i64 = i32::MAX as i64;

/// When parsing a timestamp string, timestamps after this point will be treated as
/// microseconds rather than milliseconds.
const MILLIS_CUTOFF: i64 = {
    // Approximate value for year 2400
    13_569_465_600_000
};

/// When parsing a timestamp string, timestamps after this point will be treated as
/// nanoseconds rather than microseconds.
const MICROS_CUTOFF: i64 = MILLIS_CUTOFF * 1000;

/// Color type (RGBA)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b, a: 255 }
    }

    pub fn from_rgba_u32(rgba: u32) -> Self {
        Color {
            a: ((rgba >> 24) & 0xFF) as u8,
            r: ((rgba >> 16) & 0xFF) as u8,
            g: ((rgba >> 8) & 0xFF) as u8,
            b: (rgba & 0xFF) as u8,
        }
    }

    pub fn from_rgb_u32(rgb: u32) -> Self {
        Color {
            a: 255,
            r: ((rgb >> 16) & 0xFF) as u8,
            g: ((rgb >> 8) & 0xFF) as u8,
            b: (rgb & 0xFF) as u8,
        }
    }
}

/// Escape special characters in the input string to unicode escape sequences (\uXXXX).
pub fn escape<F>(input: &str, is_special: F) -> String
where
    F: Fn(char) -> bool,
{
    let mut result = String::with_capacity(input.len());
    for ch in input.chars() {
        if is_special(ch) {
            write!(&mut result, "\\u{:04x}", ch as u32).unwrap();
        } else {
            result.push(ch);
        }
    }
    result
}

/// Unescape unicode characters in the input string.
pub fn unescape(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '\\' && i + 5 < chars.len() && chars[i + 1] == 'u' {
            // Try to parse the next 4 characters as hex
            if let Ok(code_point) = u32::from_str_radix(&input[i + 2..i + 6], 16) {
                if let Some(ch) = char::from_u32(code_point) {
                    result.push(ch);
                    i += 6;
                    continue;
                }
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// URI escape table for ASCII characters
fn get_uri_escapes() -> [String; 128] {
    let mut array = std::array::from_fn(|i| {
        let c = i as u8 as char;
        if c.is_control() {
            format!("%{:02X}", i)
        } else {
            c.to_string()
        }
    });

    // Special characters that need encoding
    array[b' ' as usize] = "%20".to_string();
    array[b'+' as usize] = "%2B".to_string();
    array[b'#' as usize] = "%23".to_string();
    array[b'"' as usize] = "%22".to_string();
    array[b'%' as usize] = "%25".to_string();
    array[b'&' as usize] = "%26".to_string();
    array[b';' as usize] = "%3B".to_string();
    array[b'<' as usize] = "%3C".to_string();
    array[b'=' as usize] = "%3D".to_string();
    array[b'>' as usize] = "%3E".to_string();
    array[b'?' as usize] = "%3F".to_string();
    array[b'[' as usize] = "%5B".to_string();
    array[b'\\' as usize] = "%5C".to_string();
    array[b']' as usize] = "%5D".to_string();
    array[b'^' as usize] = "%5E".to_string();
    array[b'{' as usize] = "%7B".to_string();
    array[b'|' as usize] = "%7C".to_string();
    array[b'}' as usize] = "%7D".to_string();

    array
}

lazy_static! {
    static ref URI_ESCAPES: [String; 128] = get_uri_escapes();
}

/// Lenient URL encoder
pub fn url_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);

    for ch in s.chars() {
        if (ch as u32) < 128 {
            result.push_str(&URI_ESCAPES[ch as usize]);
        } else {
            // For non-ASCII, use percent encoding of UTF-8 bytes
            for byte in ch.to_string().as_bytes() {
                write!(&mut result, "%{:02X}", byte).unwrap();
            }
        }
    }

    result
}

/// Lenient URL decoder
pub fn url_decode(s: &str) -> String {
    hex_decode(s, '%')
}

/// Hex decode an input string
pub fn hex_decode(input: &str, escape_char: char) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == escape_char && i + 2 < chars.len() {
            if let (Some(c1), Some(c2)) = (hex_value(chars[i + 1]), hex_value(chars[i + 2])) {
                let byte = (c1 << 4) | c2;
                result.push(byte as char);
                i += 3;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Convert hex character to value
fn hex_value(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some((c as u8) - b'0'),
        'A'..='F' => Some((c as u8) - b'A' + 10),
        'a'..='f' => Some((c as u8) - b'a' + 10),
        _ => None,
    }
}

/// Parse query string into a map
pub fn parse_query_string(query: &str) -> HashMap<String, Vec<String>> {
    let mut result = HashMap::new();

    if query.is_empty() {
        return result;
    }

    for param in query.split(&['&', ';'][..]) {
        if let Some(caps) = QUERY_PARAM.captures(param) {
            let key = url_decode(&caps[1]);
            let value = url_decode(&caps[2]);
            result.entry(key).or_insert_with(Vec::new).push(value);
        } else if !param.is_empty() {
            let key = url_decode(param);
            result.entry(key).or_insert_with(Vec::new).push("1".to_string());
        }
    }

    result
}

/// Characters allowed in variable names
fn is_allowed_in_var_name(c: char) -> bool {
    matches!(c, '.' | '-' | '_' | 'a'..='z' | 'A'..='Z' | '0'..='9')
}

/// Substitute variables in a string
pub fn substitute<F>(str: &str, vars: F) -> String
where
    F: Fn(&str) -> String,
{
    let mut result = String::with_capacity(str.len() * 2);
    let chars: Vec<char> = str.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            i += 1;
            let key = if chars[i] == '(' {
                // Parenthesized variable
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != ')' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // Skip closing paren
                    chars[start..i-1].iter().collect()
                } else {
                    String::new()
                }
            } else {
                // Simple variable
                let start = i;
                while i < chars.len() && is_allowed_in_var_name(chars[i]) {
                    i += 1;
                }
                chars[start..i].iter().collect()
            };

            if key.is_empty() {
                result.push('$');
            } else {
                result.push_str(&vars(&key));
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Substitute variables from a map
pub fn substitute_map(str: &str, vars: &HashMap<String, String>) -> String {
    substitute(str, |k| vars.get(k).cloned().unwrap_or_else(|| k.to_string()))
}

/// Check if a date string is relative
pub fn is_relative_date(str: &str, custom_ref: bool) -> bool {
    if let Some(caps) = RELATIVE_DATE.captures(str) {
        !custom_ref || (&caps[1] != "now" && &caps[1] != "epoch")
    } else {
        false
    }
}

/// Extract reference point from date string
pub fn extract_reference_point_date(str: &str) -> Option<String> {
    if let Some(caps) = RELATIVE_DATE.captures(str) {
        Some(caps[1].to_string())
    } else if let Some(caps) = NAMED_DATE.captures(str) {
        Some(caps[1].to_string())
    } else {
        None
    }
}

/// Parse a date string
pub fn parse_date(
    str: &str,
    refs: &HashMap<String, DateTime<Utc>>,
) -> Result<DateTime<Utc>> {
    if let Some(caps) = RELATIVE_DATE.captures(str) {
        let reference = parse_ref_var(refs, &caps[1])?;
        let op = &caps[2];
        let period = parse_duration(&caps[3])?;
        apply_date_offset(reference, op, period)
    } else if let Some(caps) = NAMED_DATE.captures(str) {
        parse_ref_var(refs, &caps[1])
    } else if let Some(caps) = UNIX_DATE.captures(str) {
        parse_unix_date(&caps[1])
    } else if let Some(caps) = UNIX_DATE_WITH_OP.captures(str) {
        let date = parse_unix_date(&caps[1])?;
        let op = &caps[2];
        let period = parse_duration(&caps[3])?;
        apply_date_offset(date, op, period)
    } else {
        // Try to parse as ISO date
        DateTime::parse_from_rfc3339(str)
            .map(|dt| dt.with_timezone(&Utc))
            .or_else(|_| DateTime::parse_from_str(str, "%Y-%m-%dT%H:%M:%S%.fZ")
                .map(|dt| dt.with_timezone(&Utc)))
            .map_err(|e| StringError::InvalidDate(format!("invalid date {}: {}", str, e)))
    }
}

fn parse_unix_date(d: &str) -> Result<DateTime<Utc>> {
    // let timestamp = d.parse::<i64>()
    //     .map_err(|_| StringError::InvalidDate(format!("invalid unix timestamp: {}", d)))?;
    //
    // let dt = match timestamp {
    //     t if t <= SECONDS_CUTOFF => Utc.timestamp_opt(t, 0),
    //     t if t <= MILLIS_CUTOFF => Utc.timestamp_millis_opt(t),
    //     t if t <= MICROS_CUTOFF => Utc.timestamp_micros(t),
    //     t => Utc.timestamp_nanos(t),
    // };
    //
    // dt.single()
    //     .ok_or_else(|| StringError::InvalidDate(format!("invalid timestamp: {}", d)))
    unimplemented!()
}

fn apply_date_offset(
    dt: DateTime<Utc>,
    op: &str,
    duration: Duration,
) -> Result<DateTime<Utc>> {
    match op {
        "-" => Ok(dt - duration),
        "+" => Ok(dt + duration),
        _ => Err(StringError::InvalidFormat(format!("invalid operation: {}", op))),
    }
}

fn parse_ref_var(
    refs: &HashMap<String, DateTime<Utc>>,
    v: &str,
) -> Result<DateTime<Utc>> {
    if let Some(dt) = refs.get(v) {
        Ok(*dt)
    } else {
        match v {
            "epoch" => Ok(Utc.timestamp_opt(0, 0).unwrap()),
            "now" | _ => Ok(Utc::now()),
        }
    }
}

/// Parse duration string
pub fn parse_duration(str: &str) -> Result<Duration> {
    if let Some(caps) = AT_PERIOD.captures(str) {
        parse_at_duration(&caps[1], &caps[2])
    } else if ISO_PERIOD.is_match(str) {
        // For simplicity, we'll parse basic ISO durations manually
        // In a real implementation, you'd want a more complete ISO8601 parser
        parse_iso_duration(str)
    } else {
        Err(StringError::InvalidDuration(format!("invalid period: {}", str)))
    }
}

fn parse_at_duration(amount: &str, unit: &str) -> Result<Duration> {
    let v = amount.parse::<i64>()
        .map_err(|_| StringError::InvalidDuration(format!("invalid amount: {}", amount)))?;

    let duration = match unit {
        "ns" => Duration::nanoseconds(v),
        "us" | "μs" => Duration::microseconds(v),
        "ms" => Duration::milliseconds(v),
        "seconds" | "second" | "s" => Duration::seconds(v),
        "minutes" | "minute" | "min" | "m" => Duration::minutes(v),
        "hours" | "hour" | "h" => Duration::hours(v),
        "days" | "day" | "d" => Duration::days(v),
        "weeks" | "week" | "wk" | "w" => Duration::weeks(v),
        "months" | "month" => Duration::days(v * 30),
        "years" | "year" | "y" => Duration::days(v * 365),
        _ => return Err(StringError::InvalidDuration(format!("unknown unit: {}", unit))),
    };

    Ok(duration)
}

fn parse_iso_duration(str: &str) -> Result<Duration> {
    // Simplified ISO duration parser - handles basic cases like PT1H, P1D, etc.
    if !str.starts_with('P') {
        return Err(StringError::InvalidDuration("ISO duration must start with P".to_string()));
    }

    // This is a simplified implementation. A full implementation would need
    // to handle all ISO8601 duration formats
    let mut duration = Duration::zero();
    let mut current_num = String::new();
    let mut in_time = false;

    for ch in str[1..].chars() {
        match ch {
            'T' => in_time = true,
            '0'..='9' => current_num.push(ch),
            'Y' => {
                let n: i64 = current_num.parse()
                    .map_err(|_| StringError::InvalidDuration("invalid year value".to_string()))?;
                duration = duration + Duration::days(n * 365);
                current_num.clear();
            }
            'M' => {
                let n: i64 = current_num.parse()
                    .map_err(|_| StringError::InvalidDuration("invalid month/minute value".to_string()))?;
                duration = duration + if in_time {
                    Duration::minutes(n)
                } else {
                    Duration::days(n * 30)
                };
                current_num.clear();
            }
            'W' => {
                let n: i64 = current_num.parse()
                    .map_err(|_| StringError::InvalidDuration("invalid week value".to_string()))?;
                duration = duration + Duration::weeks(n);
                current_num.clear();
            }
            'D' => {
                let n: i64 = current_num.parse()
                    .map_err(|_| StringError::InvalidDuration("invalid day value".to_string()))?;
                duration = duration + Duration::days(n);
                current_num.clear();
            }
            'H' => {
                let n: i64 = current_num.parse()
                    .map_err(|_| StringError::InvalidDuration("invalid hour value".to_string()))?;
                duration = duration + Duration::hours(n);
                current_num.clear();
            }
            'S' => {
                let n: i64 = current_num.parse()
                    .map_err(|_| StringError::InvalidDuration("invalid second value".to_string()))?;
                duration = duration + Duration::seconds(n);
                current_num.clear();
            }
            _ => return Err(StringError::InvalidDuration(format!("unexpected character: {}", ch))),
        }
    }

    Ok(duration)
}

/// Parse start and end time strings
pub fn time_range(
    s: &str,
    e: &str,
    refs: &HashMap<String, DateTime<Utc>>,
) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
    let (start, end) = if is_relative_date(s, true) || s == "e" {
        if is_relative_date(e, true) {
            return Err(StringError::InvalidDate("start and end are both relative".to_string()));
        }
        let mut new_refs = refs.clone();
        let end = parse_date(e, &new_refs)?;
        new_refs.insert("e".to_string(), end);
        let start = parse_date(s, &new_refs)?;
        (start, end)
    } else {
        let mut new_refs = refs.clone();
        let start = parse_date(s, &new_refs)?;
        new_refs.insert("s".to_string(), start);
        let end = parse_date(e, &new_refs)?;
        (start, end)
    };

    if start > end {
        return Err(StringError::InvalidDate("end time is before start time".to_string()));
    }

    Ok((start, end))
}

/// Parse a color from hex string
pub fn parse_color(str: &str) -> Result<Color> {
    let len = str.len();
    if len != 3 && len != 6 && len != 8 {
        return Err(StringError::InvalidColor("color must be hex string [AA]RRGGBB".to_string()));
    }

    let expanded = if len == 3 {
        // Expand 3-char format (RGB) to 6-char (RRGGBB)
        str.chars()
            .flat_map(|c| [c, c])
            .collect::<String>()
    } else {
        str.to_string()
    };

    if expanded.len() <= 6 {
        let rgb = u32::from_str_radix(&expanded, 16)
            .map_err(|_| StringError::InvalidColor("invalid hex color".to_string()))?;
        Ok(Color::from_rgb_u32(rgb))
    } else {
        let rgba = u32::from_str_radix(&expanded, 16)
            .map_err(|_| StringError::InvalidColor("invalid hex color".to_string()))?;
        Ok(Color::from_rgba_u32(rgba))
    }
}

/// Duration constants
const ONE_SECOND: i64 = 1000;
const ONE_MINUTE: i64 = ONE_SECOND * 60;
const ONE_HOUR: i64 = ONE_MINUTE * 60;
const ONE_DAY: i64 = ONE_HOUR * 24;
const ONE_WEEK: i64 = ONE_DAY * 7;

/// Convert duration to string
pub fn duration_to_string(d: &Duration) -> String {
    let millis = d.num_milliseconds();

    match millis {
        t if t % ONE_WEEK == 0 => format!("{}w", t / ONE_WEEK),
        t if t % ONE_DAY == 0 => format!("{}d", t / ONE_DAY),
        t if t % ONE_HOUR == 0 => format!("{}h", t / ONE_HOUR),
        t if t % ONE_MINUTE == 0 => format!("{}m", t / ONE_MINUTE),
        t if t % ONE_SECOND == 0 => format!("{}s", t / ONE_SECOND),
        _ => format!("PT{}S", d.num_seconds()),
    }
}

/// Strip margin from multi-line strings
pub fn strip_margin(str: &str) -> String {
    let trimmed = str.trim();
    let lines: Vec<&str> = trimmed.lines().map(|line| line.trim_start()).collect();
    lines.join(" ").replace("  ", "\n\n")
}

/// Zero pad a string
pub fn zero_pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{:0>width$}", s, width = width)
    }
}

/// Zero pad integer as hex
pub fn zero_pad_hex(v: u32, width: usize) -> String {
    format!("{:0>width$x}", v, width = width)
}

/// Zero pad long as hex
pub fn zero_pad_hex_u64(v: u64, width: usize) -> String {
    format!("{:0>width$x}", v, width = width)
}

/// Zero pad bytes as hex string
pub fn zero_pad_bytes(bytes: &[u8], width: usize) -> String {
    let hex_len = bytes.len() * 2;
    let padding = if width > hex_len { width - hex_len } else { 0 };

    let mut result = String::with_capacity(width.max(hex_len));
    for _ in 0..padding {
        result.push('0');
    }
    for byte in bytes {
        write!(&mut result, "{:02x}", byte).unwrap();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_unescape() {
        let original = "Hello\nWorld\t!";
        let escaped = escape(original, |c| c.is_control());
        assert_eq!(escaped, "Hello\\u000aWorld\\u0009!");
        let unescaped = unescape(&escaped);
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_url_encode_decode() {
        let original = "hello world & friends";
        let encoded = url_encode(original);
        assert_eq!(encoded, "hello%20world%20%26%20friends");
        let decoded = url_decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_parse_query_string() {
        let query = "foo=bar&baz=qux&foo=second";
        let params = parse_query_string(query);
        assert_eq!(params.get("foo").unwrap(), &vec!["bar", "second"]);
        assert_eq!(params.get("baz").unwrap(), &vec!["qux"]);
    }

    #[test]
    fn test_substitute() {
        let template = "Hello $name, you are $age years old";
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("age".to_string(), "30".to_string());
        let result = substitute_map(template, &vars);
        assert_eq!(result, "Hello Alice, you are 30 years old");
    }

    #[test]
    fn test_parse_color() {
        assert_eq!(parse_color("fff").unwrap(), Color::from_rgb(255, 255, 255));
        assert_eq!(parse_color("ff0000").unwrap(), Color::from_rgb(255, 0, 0));
        assert_eq!(parse_color("80ff0000").unwrap(), Color::new(255, 0, 0, 128));
    }

    #[test]
    fn test_zero_pad() {
        assert_eq!(zero_pad("123", 5), "00123");
        assert_eq!(zero_pad("12345", 3), "12345");
        assert_eq!(zero_pad_hex(0x1234, 8), "00001234");
    }
}
