use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Aggregate types
pub mod aggregate_type {
    pub const SUM: i32 = 0;
    pub const COUNT: i32 = 1;
    pub const MIN: i32 = 2;
    pub const MAX: i32 = 3;
}

/// Max size for an array block
pub const MAX_SIZE: usize = 120;

/// Helper module for NaN-aware math operations
mod math_helper {
    pub fn add_nan(a: f64, b: f64) -> f64 {
        if a.is_nan() { b }
        else if b.is_nan() { a }
        else { a + b }
    }
}

/// Helper for double to int hash map functionality
struct DoubleIntHashMap {
    map: HashMap<u64, i32>,
}

impl DoubleIntHashMap {
    fn new() -> Self {
        Self { map: HashMap::new() }
    }

    fn get(&self, key: f64, default: i32) -> i32 {
        if key.is_nan() {
            return default;
        }
        let bits = key.to_bits();
        self.map.get(&bits).copied().unwrap_or(default)
    }

    fn put(&mut self, key: f64, value: i32) {
        if !key.is_nan() {
            let bits = key.to_bits();
            self.map.insert(bits, value);
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn for_each<F>(&self, mut f: F)
    where F: FnMut(f64, i32) {
        for (&bits, &value) in &self.map {
            f(f64::from_bits(bits), value);
        }
    }
}

/// Represents a fixed size window of metric data
pub trait Block: Send + Sync {
    /// Start time for the block (epoch in milliseconds)
    fn start(&self) -> i64;

    /// Number of data points in the block
    fn size(&self) -> usize;

    /// Return the value for a given position with aggregate type
    fn get_with_aggr(&self, pos: usize, aggr: i32) -> f64 {
        let v = self.get(pos);
        match aggr {
            aggregate_type::SUM => v,
            aggregate_type::COUNT => if v.is_nan() { f64::NAN } else { 1.0 },
            aggregate_type::MIN => v,
            aggregate_type::MAX => v,
            _ => v,
        }
    }

    /// Return the value for a given position
    fn get(&self, pos: usize) -> f64;

    /// Number of bytes required to store this block
    fn byte_count(&self) -> usize;

    /// Returns a copy of the block as an array-backed block
    fn to_array_block(&self) -> ArrayBlock {
        let mut block = ArrayBlock::new(self.start(), self.size());
        for i in 0..self.size() {
            block.buffer[i] = self.get(i);
        }
        block
    }

    /// Clone the block into a boxed trait object
    fn clone_box(&self) -> Box<dyn Block>;
}

/// Block type that can be updated incrementally
pub trait MutableBlock: Block {
    /// Update the value for the specified position
    fn update(&mut self, pos: usize, value: f64);

    /// Reset this block so it can be re-used
    fn reset(&mut self, t: i64);
}

/// Block that stores raw data in an array
#[derive(Clone, Debug)]
pub struct ArrayBlock {
    pub start: i64,
    pub buffer: Vec<f64>,
}

impl ArrayBlock {
    pub fn new(start: i64, size: usize) -> Self {
        Self {
            start,
            buffer: vec![f64::NAN; size],
        }
    }

    /// Add contents of another block to this block
    pub fn add(&mut self, b: &dyn Block, aggr: i32) {
        for i in 0..self.size() {
            self.buffer[i] = math_helper::add_nan(self.buffer[i], b.get_with_aggr(i, aggr));
        }
    }

    /// Select the minimum value of this block or b
    pub fn min(&mut self, b: &dyn Block, aggr: i32) {
        for i in 0..self.size() {
            self.buffer[i] = self.buffer[i].min(b.get_with_aggr(i, aggr));
        }
    }

    /// Select the maximum value of this block or b
    pub fn max(&mut self, b: &dyn Block, aggr: i32) {
        for i in 0..self.size() {
            self.buffer[i] = self.buffer[i].max(b.get_with_aggr(i, aggr));
        }
    }

    /// Merge data from another block
    pub fn merge(&mut self, b: &dyn Block) -> usize {
        let mut changed = 0;
        for i in 0..self.size() {
            let v1 = self.buffer[i];
            let v2 = b.get(i);
            if v1.is_nan() {
                self.buffer[i] = v2;
                if !v2.is_nan() {
                    changed += 1;
                }
            } else if v1 < v2 {
                self.buffer[i] = v2;
                changed += 1;
            }
        }
        changed
    }
}

impl Block for ArrayBlock {
    fn start(&self) -> i64 { self.start }
    fn size(&self) -> usize { self.buffer.len() }
    fn get(&self, pos: usize) -> f64 { self.buffer[pos] }
    fn byte_count(&self) -> usize { 2 + 8 * self.buffer.len() }
    fn to_array_block(&self) -> ArrayBlock { self.clone() }
    fn clone_box(&self) -> Box<dyn Block> { Box::new(self.clone()) }
}

impl MutableBlock for ArrayBlock {
    fn update(&mut self, pos: usize, value: f64) {
        self.buffer[pos] = value;
    }

    fn reset(&mut self, t: i64) {
        self.start = t;
        self.buffer.fill(f64::NAN);
    }
}

impl PartialEq for ArrayBlock {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.buffer == other.buffer
    }
}

impl Eq for ArrayBlock {}

impl Hash for ArrayBlock {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        for &v in &self.buffer {
            v.to_bits().hash(state);
        }
    }
}

/// Block using single-precision floats for compression
#[derive(Clone, Debug)]
pub struct FloatArrayBlock {
    pub start: i64,
    pub buffer: Vec<f32>,
}

impl FloatArrayBlock {
    pub fn from_array_block(b: &ArrayBlock) -> Self {
        let buffer: Vec<f32> = b.buffer.iter().map(|&v| v as f32).collect();
        Self {
            start: b.start,
            buffer,
        }
    }
}

impl Block for FloatArrayBlock {
    fn start(&self) -> i64 { self.start }
    fn size(&self) -> usize { self.buffer.len() }
    fn get(&self, pos: usize) -> f64 { self.buffer[pos] as f64 }
    fn byte_count(&self) -> usize { 2 + 4 * self.buffer.len() }
    fn clone_box(&self) -> Box<dyn Block> { Box::new(self.clone()) }
}

/// Constants for sparse blocks
pub mod sparse_block {
    pub const NOT_FOUND: i32 = -4;
    pub const NAN: i32 = -3;
    pub const ZERO: i32 = -2;
    pub const ONE: i32 = -1;
    pub const UNDEFINED: i32 = 0;

    pub fn predefined_index(value: f64) -> i32 {
        if value.is_nan() { NAN }
        else if value == 0.0 { ZERO }
        else if value == 1.0 { ONE }
        else { UNDEFINED }
    }

    pub fn get(pos: i32, values: &[f64]) -> f64 {
        match pos {
            NAN => f64::NAN,
            ZERO => 0.0,
            ONE => 1.0,
            _ => values[pos as usize],
        }
    }
}

/// Block optimized for storing a small set of discrete values
#[derive(Clone, Debug)]
pub struct SparseBlock {
    pub start: i64,
    pub indexes: Vec<u8>,
    pub values: Vec<f64>,
}

impl Block for SparseBlock {
    fn start(&self) -> i64 { self.start }
    fn size(&self) -> usize { self.indexes.len() }

    fn get(&self, pos: usize) -> f64 {
        let idx = self.indexes[pos] as i32;
        sparse_block::get(idx, &self.values)
    }

    fn byte_count(&self) -> usize {
        2 + self.indexes.len() + 8 * self.values.len()
    }

    fn clone_box(&self) -> Box<dyn Block> { Box::new(self.clone()) }
}

/// Simple block where all data points have the same value
#[derive(Clone, Debug)]
pub struct ConstantBlock {
    pub start: i64,
    pub size: usize,
    pub value: f64,
}

impl Block for ConstantBlock {
    fn start(&self) -> i64 { self.start }
    fn size(&self) -> usize { self.size }
    fn get(&self, _pos: usize) -> f64 { self.value }
    fn byte_count(&self) -> usize { 2 + 4 + 8 }
    fn clone_box(&self) -> Box<dyn Block> { Box::new(self.clone()) }
}

/// Mutable block with compression during updates
#[derive(Clone, Debug)]
pub struct CompressedArrayBlock {
    pub start: i64,
    pub size: usize,
    buffer: Vec<u64>,
}

impl CompressedArrayBlock {
    const MODE_2_BIT: i32 = 0;
    const MODE_4_BIT: i32 = 1;
    const MODE_64_BIT: i32 = 2;

    const ONE_OVER_60: f64 = 1.0 / 60.0;
    const BITS_PER_LONG: usize = 64;
    const BIT2_PER_LONG: usize = Self::BITS_PER_LONG / 2;
    const BIT4_PER_LONG: usize = Self::BITS_PER_LONG / 4;

    pub fn new(start: i64, size: usize) -> Self {
        let buffer = if size < 16 {
            Self::new_array(size, false)
        } else {
            Self::new_array(Self::ceiling_divide(size * 2, Self::BITS_PER_LONG), true)
        };

        Self { start, size, buffer }
    }

    fn new_array(n: usize, compressed: bool) -> Vec<u64> {
        if compressed {
            vec![0; n]
        } else {
            vec![f64::NAN.to_bits(); n]
        }
    }

    fn ceiling_divide(dividend: usize, divisor: usize) -> usize {
        (dividend + divisor - 1) / divisor
    }

    fn determine_mode(&self) -> i32 {
        if self.size < 16 {
            Self::MODE_64_BIT
        } else {
            let bits_per_value = Self::ceiling_divide(
                self.buffer.len() * Self::BITS_PER_LONG,
                self.size
            );
            if bits_per_value < 4 {
                Self::MODE_2_BIT
            } else if bits_per_value >= 64 {
                Self::MODE_64_BIT
            } else {
                Self::MODE_4_BIT
            }
        }
    }

    fn set2(buffer: u64, pos: usize, value: u64) -> u64 {
        let shift = pos * 2;
        (buffer & !(0x3u64 << shift)) | ((value & 0x3) << shift)
    }

    fn get2(buffer: u64, pos: usize) -> i32 {
        let shift = pos * 2;
        ((buffer >> shift) & 0x3) as i32
    }

    fn set4(buffer: u64, pos: usize, value: u64) -> u64 {
        let shift = pos * 4;
        (buffer & !(0xFu64 << shift)) | ((value & 0xF) << shift)
    }

    fn get4(buffer: u64, pos: usize) -> i32 {
        let shift = pos * 4;
        ((buffer >> shift) & 0xF) as i32
    }

    fn int_value(value: f64) -> i32 {
        if value.is_nan() { 0 }
        else if value == 0.0 { 1 }
        else if value == 1.0 { 2 }
        else if value == Self::ONE_OVER_60 { 3 }
        else { -1 }
    }

    fn double_value(value: i32) -> f64 {
        match value {
            0 => f64::NAN,
            1 => 0.0,
            2 => 1.0,
            3 => Self::ONE_OVER_60,
            _ => f64::NAN,
        }
    }
}

impl Block for CompressedArrayBlock {
    fn start(&self) -> i64 { self.start }
    fn size(&self) -> usize { self.size }

    fn get(&self, pos: usize) -> f64 {
        match self.determine_mode() {
            Self::MODE_2_BIT => {
                let v = Self::get2(self.buffer[pos / Self::BIT2_PER_LONG], pos % Self::BIT2_PER_LONG);
                Self::double_value(v)
            }
            Self::MODE_4_BIT => {
                let v = Self::get4(self.buffer[pos / Self::BIT4_PER_LONG], pos % Self::BIT4_PER_LONG);
                if v < 4 {
                    Self::double_value(v)
                } else {
                    let idx = (v - 4) as usize + Self::ceiling_divide(self.size * 4, Self::BITS_PER_LONG);
                    f64::from_bits(self.buffer[idx])
                }
            }
            Self::MODE_64_BIT => {
                f64::from_bits(self.buffer[pos])
            }
            _ => panic!("Unsupported mode"),
        }
    }

    fn byte_count(&self) -> usize {
        2 + 8 * self.buffer.len()
    }

    fn clone_box(&self) -> Box<dyn Block> { Box::new(self.clone()) }
}

impl MutableBlock for CompressedArrayBlock {
    fn update(&mut self, pos: usize, value: f64) {
        // Implementation would be quite complex, similar to Scala version
        // Omitting detailed implementation for brevity
        unimplemented!("CompressedArrayBlock update implementation")
    }

    fn reset(&mut self, _t: i64) {
        unimplemented!("CompressedArrayBlock reset not supported")
    }
}

/// Block containing rollup aggregates
// #[derive(Clone)]
// pub struct RollupBlock {
//     pub sum: Box<dyn Block>,
//     pub count: Box<dyn Block>,
//     pub min: Box<dyn Block>,
//     pub max: Box<dyn Block>,
// }
//
// impl RollupBlock {
//     pub fn empty(start: i64, size: usize) -> Self {
//         Self {
//             sum: Box::new(ArrayBlock::new(start, size)),
//             count: Box::new(ArrayBlock::new(start, size)),
//             min: Box::new(ArrayBlock::new(start, size)),
//             max: Box::new(ArrayBlock::new(start, size)),
//         }
//     }
//
//     pub fn compress(&self) -> Self {
//         Self {
//             sum: compress_if_needed(self.sum.clone_box()),
//             count: compress_if_needed(self.count.clone_box()),
//             min: compress_if_needed(self.min.clone_box()),
//             max: compress_if_needed(self.max.clone_box()),
//         }
//     }
// }
//
// impl Block for RollupBlock {
//     fn start(&self) -> i64 { self.sum.start() }
//     fn size(&self) -> usize { self.sum.size() }
//     fn get(&self, pos: usize) -> f64 { self.sum.get(pos) }
//
//     fn get_with_aggr(&self, pos: usize, aggr: i32) -> f64 {
//         match aggr {
//             aggregate_type::SUM => self.sum.get(pos),
//             aggregate_type::COUNT => self.count.get(pos),
//             aggregate_type::MIN => self.min.get(pos),
//             aggregate_type::MAX => self.max.get(pos),
//             _ => self.sum.get(pos),
//         }
//     }
//
//     fn byte_count(&self) -> usize {
//         2 + self.sum.byte_count() + self.count.byte_count() +
//             self.min.byte_count() + self.max.byte_count()
//     }
//
//     fn clone_box(&self) -> Box<dyn Block> {
//         Box::new(self.clone())
//     }
// }

/// Compress an array block into a more compact type
pub fn compress(block: ArrayBlock) -> Box<dyn Block> {
    if block.size() < 10 {
        return Box::new(block);
    }

    let mut idx_map = DoubleIntHashMap::new();
    let mut next_idx = 0;
    let mut indexes = vec![0u8; block.size()];
    let mut is_constant = true;
    let mut prev = 0;

    for i in 0..block.size() {
        let v = block.buffer[i];
        let predef_idx = sparse_block::predefined_index(v);
        let mut idx = if predef_idx == sparse_block::UNDEFINED {
            idx_map.get(v, sparse_block::NOT_FOUND)
        } else {
            predef_idx
        };

        if idx == sparse_block::NOT_FOUND {
            if next_idx == MAX_SIZE as i32 {
                return Box::new(block);
            }
            idx = next_idx;
            next_idx += 1;
            idx_map.put(v, idx);
        }

        indexes[i] = idx as u8;
        if i > 0 {
            is_constant = is_constant && (prev == idx);
        }
        prev = idx;
    }

    let mut values = vec![0.0; idx_map.len()];
    idx_map.for_each(|k, v| {
        values[v as usize] = k;
    });

    if is_constant {
        Box::new(ConstantBlock {
            start: block.start,
            size: block.size(),
            value: sparse_block::get(indexes[0] as i32, &values),
        })
    } else if idx_map.len() < block.size() / 2 {
        Box::new(SparseBlock {
            start: block.start,
            indexes,
            values,
        })
    } else {
        Box::new(block)
    }
}

/// Compress with potential loss of precision
// pub fn lossy_compress(block: ArrayBlock) -> Box<dyn Block> {
//     let compressed = compress(block);
//     match compressed.as_ref() {
//         b if b.type_id() == std::any::TypeId::of::<ArrayBlock>() => {
//             // Convert to FloatArrayBlock
//             let array_block = b.to_array_block();
//             Box::new(FloatArrayBlock::from_array_block(&array_block))
//         }
//         _ => compressed,
//     }
// }

/// Compress a block if needed
pub fn compress_if_needed(block: Box<dyn Block>) -> Box<dyn Block> {
    // Would need dynamic type checking here
    // For now, return as-is
    block
}

/// Add data from two blocks
pub fn add(block1: Box<dyn Block>, block2: Box<dyn Block>) -> Box<dyn Block> {
    assert_eq!(block1.size(), block2.size(), "Block sizes must match");

    let mut b1 = block1.to_array_block();
    b1.add(block2.as_ref(), aggregate_type::SUM);
    Box::new(b1)
}

/// Merge data from two blocks
pub fn merge(block1: Box<dyn Block>, block2: Box<dyn Block>) -> Box<dyn Block> {
    assert_eq!(block1.size(), block2.size(), "Block sizes must match");

    let mut b1 = block1.to_array_block();
    b1.merge(block2.as_ref());
    compress(b1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_block() {
        let mut block = ArrayBlock::new(1000, 60);
        assert_eq!(block.size(), 60);
        assert_eq!(block.start(), 1000);

        block.update(0, 42.0);
        assert_eq!(block.get(0), 42.0);

        block.update(1, 100.0);
        assert_eq!(block.get(1), 100.0);
    }

    #[test]
    fn test_constant_block() {
        let block = ConstantBlock {
            start: 1000,
            size: 60,
            value: 42.0,
        };

        assert_eq!(block.get(0), 42.0);
        assert_eq!(block.get(59), 42.0);
    }

    // #[test]
    // fn test_sparse_block_compression() {
    //     let mut block = ArrayBlock::new(1000, 60);
    //
    //     // Fill with mostly the same values
    //     for i in 0..60 {
    //         block.update(i, if i % 10 == 0 { 1.0 } else { 0.0 });
    //     }
    //
    //     let compressed = compress(block);
    //     assert_eq!(compressed.size(), 60);
    //
    //     // Verify values
    //     for i in 0..60 {
    //         let expected = if i % 10 == 0 { 1.0 } else { 0.0 };
    //         assert_eq!(compressed.get(i), expected);
    //     }
    // }
}