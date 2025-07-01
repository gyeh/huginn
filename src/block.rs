use std::cmp;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggrType {
    Sum,
    Count,
    Min,
    Max,
}

pub trait Block: fmt::Debug + Send + Sync {
    fn size(&self) -> usize;
    fn get(&self, pos: usize) -> f64;

    fn as_array_block(&self) -> Option<ArrayBlock> {
        if self.size() <= 1 || self.bytesize() > 3 * self.size() * 8 {
            return None;
        }
        let mut data = vec![0.0; self.size()];
        for i in 0..self.size() {
            data[i] = self.get(i);
        }
        Some(ArrayBlock::new(data))
    }

    fn bytesize(&self) -> usize;

    fn to_array(&self) -> Vec<f64> {
        let mut result = vec![0.0; self.size()];
        for i in 0..self.size() {
            result[i] = self.get(i);
        }
        result
    }

    fn merge(&self, aggr: AggrType, other: &dyn Block) -> Box<dyn Block> where Self: Sized {
        match aggr {
            AggrType::Sum => merge_blocks(self, other, |v1, v2| {
                if v1.is_nan() || v2.is_nan() { f64::NAN } else { v1 + v2 }
            }),
            AggrType::Count => merge_blocks(self, other, |v1, v2| {
                if v1.is_nan() || v2.is_nan() { f64::NAN } else { 1.0 }
            }),
            AggrType::Min => merge_blocks(self, other, |v1, v2| {
                if v1.is_nan() || v2.is_nan() { f64::NAN } else { v1.min(v2) }
            }),
            AggrType::Max => merge_blocks(self, other, |v1, v2| {
                if v1.is_nan() || v2.is_nan() { f64::NAN } else { v1.max(v2) }
            }),
        }
    }
}

pub trait MutableBlock: Block {
    fn update(&mut self, pos: usize, value: f64);
    fn reset(&mut self, pos: usize);
}

#[derive(Debug, Clone)]
pub struct ArrayBlock {
    values: Vec<f64>,
}

impl ArrayBlock {
    pub fn new(values: Vec<f64>) -> Self {
        ArrayBlock { values }
    }

    pub fn fill(size: usize, value: f64) -> Self {
        ArrayBlock {
            values: vec![value; size],
        }
    }
}

impl Block for ArrayBlock {
    fn size(&self) -> usize {
        self.values.len()
    }

    fn get(&self, pos: usize) -> f64 {
        self.values[pos]
    }

    fn bytesize(&self) -> usize {
        16 + self.values.len() * 8
    }
}

impl MutableBlock for ArrayBlock {
    fn update(&mut self, pos: usize, value: f64) {
        self.values[pos] = value;
    }

    fn reset(&mut self, pos: usize) {
        self.values[pos] = f64::NAN;
    }
}

#[derive(Debug, Clone)]
pub struct FloatArrayBlock {
    values: Vec<f32>,
}

impl FloatArrayBlock {
    pub fn new(size: usize, block: &dyn Block) -> Self {
        let mut values = vec![0.0f32; size];
        for i in 0..size {
            values[i] = block.get(i) as f32;
        }
        FloatArrayBlock { values }
    }
}

impl Block for FloatArrayBlock {
    fn size(&self) -> usize {
        self.values.len()
    }

    fn get(&self, pos: usize) -> f64 {
        self.values[pos] as f64
    }

    fn bytesize(&self) -> usize {
        16 + self.values.len() * 4
    }
}

#[derive(Debug, Clone)]
pub struct ConstantBlock {
    size: usize,
    value: f64,
}

impl ConstantBlock {
    pub fn new(size: usize, value: f64) -> Self {
        ConstantBlock { size, value }
    }
}

impl Block for ConstantBlock {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, _pos: usize) -> f64 {
        self.value
    }

    fn bytesize(&self) -> usize {
        32
    }
}

#[derive(Debug, Clone)]
pub struct SparseBlock {
    size: usize,
    values: Vec<f64>,
    indices: Vec<u16>,
}

impl SparseBlock {
    pub fn new(size: usize, block: &dyn Block) -> Option<Self> {
        let mut value_set = std::collections::HashSet::new();
        for i in 0..size {
            let v = block.get(i);
            if !v.is_nan() {
                value_set.insert(v.to_bits());
            }
        }

        if value_set.len() > 16 {
            return None;
        }

        let mut values: Vec<f64> = value_set.iter().map(|&bits| f64::from_bits(bits)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut indices = vec![255u16; size];
        for i in 0..size {
            let v = block.get(i);
            if !v.is_nan() {
                if let Some(pos) = values.iter().position(|&x| x == v) {
                    indices[i] = pos as u16;
                }
            }
        }

        Some(SparseBlock { size, values, indices })
    }
}

impl Block for SparseBlock {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, pos: usize) -> f64 {
        let idx = self.indices[pos];
        if idx == 255 {
            f64::NAN
        } else {
            self.values[idx as usize]
        }
    }

    fn bytesize(&self) -> usize {
        32 + self.values.len() * 8 + self.indices.len() * 2
    }
}

#[derive(Debug)]
pub struct RollupBlock {
    size: usize,
    sum: Box<dyn Block>,
    count: Box<dyn Block>,
    min: Box<dyn Block>,
    max: Box<dyn Block>,
}

impl RollupBlock {
    pub fn new(size: usize, blocks: &[&dyn Block]) -> Self {
        let mut sum_block = ArrayBlock::fill(size, 0.0);
        let mut count_block = ArrayBlock::fill(size, 0.0);
        let mut min_block = ArrayBlock::fill(size, f64::INFINITY);
        let mut max_block = ArrayBlock::fill(size, f64::NEG_INFINITY);

        for i in 0..size {
            let mut sum = 0.0;
            let mut count = 0.0;
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let mut has_value = false;

            for block in blocks {
                let v = block.get(i);
                if !v.is_nan() {
                    sum += v;
                    count += 1.0;
                    min = min.min(v);
                    max = max.max(v);
                    has_value = true;
                }
            }

            if has_value {
                sum_block.update(i, sum);
                count_block.update(i, count);
                min_block.update(i, min);
                max_block.update(i, max);
            } else {
                sum_block.update(i, f64::NAN);
                count_block.update(i, f64::NAN);
                min_block.update(i, f64::NAN);
                max_block.update(i, f64::NAN);
            }
        }

        RollupBlock {
            size,
            sum: Box::new(sum_block),
            count: Box::new(count_block),
            min: Box::new(min_block),
            max: Box::new(max_block),
        }
    }

    pub fn get_aggr(&self, aggr: AggrType, pos: usize) -> f64 {
        match aggr {
            AggrType::Sum => self.sum.get(pos),
            AggrType::Count => self.count.get(pos),
            AggrType::Min => self.min.get(pos),
            AggrType::Max => self.max.get(pos),
        }
    }
}

impl Block for RollupBlock {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, pos: usize) -> f64 {
        self.sum.get(pos)
    }

    fn bytesize(&self) -> usize {
        24 + self.sum.bytesize() + self.count.bytesize() + self.min.bytesize() + self.max.bytesize()
    }
}

#[derive(Debug)]
pub struct CompressedArrayBlock {
    size: usize,
    buffer: Vec<u8>,
    bits_per_value: u8,
}

impl CompressedArrayBlock {
    pub fn new(size: usize, block: &dyn Block) -> Option<Self> {
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        let mut values = vec![0i64; size];

        for i in 0..size {
            let v = block.get(i);
            if v.is_nan() {
                values[i] = i64::MAX;
            } else {
                let long_val = v.to_bits() as i64;
                values[i] = long_val;
                if long_val != i64::MAX {
                    min = min.min(long_val);
                    max = max.max(long_val);
                }
            }
        }

        if min == i64::MAX {
            min = 0;
            max = 0;
        }

        let diff = (max - min) as u64;
        let bits_per_value = if diff == 0 {
            1
        } else if diff < 4 {
            2
        } else if diff < 16 {
            4
        } else {
            return None;
        };

        let buffer_size = (size * bits_per_value as usize + 7) / 8 + 16;
        let mut buffer = vec![0u8; buffer_size];

        buffer[0..8].copy_from_slice(&min.to_le_bytes());
        buffer[8..16].copy_from_slice(&max.to_le_bytes());

        for (i, &value) in values.iter().enumerate() {
            let normalized = if value == i64::MAX {
                (1 << bits_per_value) - 1
            } else {
                ((value - min) as u64) as u8
            };
            set_bits(&mut buffer[16..], i, bits_per_value, normalized);
        }

        Some(CompressedArrayBlock {
            size,
            buffer,
            bits_per_value,
        })
    }
}

impl Block for CompressedArrayBlock {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, pos: usize) -> f64 {
        let min = i64::from_le_bytes(self.buffer[0..8].try_into().unwrap());
        let normalized = get_bits(&self.buffer[16..], pos, self.bits_per_value);

        if normalized == (1 << self.bits_per_value) - 1 {
            f64::NAN
        } else {
            f64::from_bits((min + normalized as i64) as u64)
        }
    }

    fn bytesize(&self) -> usize {
        24 + self.buffer.len()
    }
}

fn set_bits(buffer: &mut [u8], index: usize, bits_per_value: u8, value: u8) {
    let bit_index = index * bits_per_value as usize;
    let byte_index = bit_index / 8;
    let bit_offset = bit_index % 8;

    if bit_offset + bits_per_value as usize <= 8 {
        let mask = ((1 << bits_per_value) - 1) << bit_offset;
        buffer[byte_index] = (buffer[byte_index] & !mask) | ((value << bit_offset) & mask);
    } else {
        let first_bits = 8 - bit_offset;
        let second_bits = bits_per_value as usize - first_bits;

        let first_mask = ((1 << first_bits) - 1) << bit_offset;
        buffer[byte_index] = (buffer[byte_index] & !first_mask) | ((value << bit_offset) & first_mask);

        let second_mask = (1 << second_bits) - 1;
        buffer[byte_index + 1] = (buffer[byte_index + 1] & !second_mask) | ((value >> first_bits) & second_mask);
    }
}

fn get_bits(buffer: &[u8], index: usize, bits_per_value: u8) -> u8 {
    let bit_index = index * bits_per_value as usize;
    let byte_index = bit_index / 8;
    let bit_offset = bit_index % 8;

    if bit_offset + bits_per_value as usize <= 8 {
        (buffer[byte_index] >> bit_offset) & ((1 << bits_per_value) - 1)
    } else {
        let first_bits = 8 - bit_offset;
        let second_bits = bits_per_value as usize - first_bits;

        let first_part = (buffer[byte_index] >> bit_offset) & ((1 << first_bits) - 1);
        let second_part = buffer[byte_index + 1] & ((1 << second_bits) - 1);

        first_part | (second_part << first_bits)
    }
}

fn merge_blocks(b1: &dyn Block, b2: &dyn Block, f: impl Fn(f64, f64) -> f64) -> Box<dyn Block> {
    let size = cmp::min(b1.size(), b2.size());
    let mut result = ArrayBlock::fill(size, f64::NAN);

    for i in 0..size {
        result.update(i, f(b1.get(i), b2.get(i)));
    }

    Box::new(result)
}

pub fn compress(block: &dyn Block) -> Box<dyn Block> {
    let size = block.size();

    if size <= 16 {
        return Box::new(ArrayBlock::new(block.to_array()));
    }

    let mut constant_value = None;
    for i in 0..size {
        let v = block.get(i);
        match constant_value {
            None => constant_value = Some(v),
            Some(cv) => {
                if (v.is_nan() && !cv.is_nan()) || (!v.is_nan() && cv.is_nan()) ||
                    (!v.is_nan() && !cv.is_nan() && v != cv) {
                    constant_value = None;
                    break;
                }
            }
        }
    }

    if let Some(cv) = constant_value {
        return Box::new(ConstantBlock::new(size, cv));
    }

    if let Some(sparse) = SparseBlock::new(size, block) {
        if sparse.bytesize() < block.bytesize() {
            return Box::new(sparse);
        }
    }

    if let Some(compressed) = CompressedArrayBlock::new(size, block) {
        if compressed.bytesize() < block.bytesize() {
            return Box::new(compressed);
        }
    }

    if let Some(array_block) = block.as_array_block() {
        Box::new(array_block)
    } else {
        Box::new(ArrayBlock::new(block.to_array()))
    }
}

pub fn lossy_compress(block: &dyn Block) -> Box<dyn Block> {
    let compressed = compress(block);
    let float_block = FloatArrayBlock::new(block.size(), block);

    if float_block.bytesize() < compressed.bytesize() {
        Box::new(float_block)
    } else {
        compressed
    }
}

pub const COMMON_VALUES: [f64; 4] = [f64::NAN, 0.0, 1.0, 1.0 / 60.0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_block_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let block = ArrayBlock::new(values.clone());

        assert_eq!(block.size(), 5);
        for i in 0..5 {
            assert_eq!(block.get(i), values[i]);
        }
    }

    #[test]
    fn test_array_block_mutable() {
        let mut block = ArrayBlock::fill(5, 0.0);

        block.update(2, 42.0);
        assert_eq!(block.get(2), 42.0);

        block.reset(2);
        assert!(block.get(2).is_nan());
    }

    #[test]
    fn test_constant_block() {
        let block = ConstantBlock::new(100, 42.0);

        assert_eq!(block.size(), 100);
        for i in 0..100 {
            assert_eq!(block.get(i), 42.0);
        }
    }

    #[test]
    fn test_sparse_block() {
        let mut array = ArrayBlock::fill(10, f64::NAN);
        array.update(2, 1.0);
        array.update(5, 2.0);
        array.update(8, 1.0);

        let sparse = SparseBlock::new(10, &array).unwrap();

        assert_eq!(sparse.size(), 10);
        assert!(sparse.get(0).is_nan());
        assert_eq!(sparse.get(2), 1.0);
        assert_eq!(sparse.get(5), 2.0);
        assert_eq!(sparse.get(8), 1.0);
    }

    #[test]
    fn test_float_array_block() {
        let values = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        let array = ArrayBlock::new(values.clone());
        let float_block = FloatArrayBlock::new(5, &array);

        assert_eq!(float_block.size(), 5);
        for i in 0..5 {
            assert!((float_block.get(i) - values[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_compressed_array_block_2bit() {
        let values = vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0];
        let array = ArrayBlock::new(values);
        let compressed = CompressedArrayBlock::new(6, &array).unwrap();

        assert_eq!(compressed.size(), 6);
        assert_eq!(compressed.get(0), 1.0);
        assert_eq!(compressed.get(1), 2.0);
        assert_eq!(compressed.get(2), 3.0);
        assert_eq!(compressed.get(3), 2.0);
        assert_eq!(compressed.get(4), 1.0);
        assert_eq!(compressed.get(5), 3.0);
    }

    #[test]
    fn test_rollup_block() {
        let block1 = ArrayBlock::new(vec![1.0, 2.0, f64::NAN, 4.0]);
        let block2 = ArrayBlock::new(vec![5.0, f64::NAN, 3.0, 6.0]);

        let rollup = RollupBlock::new(4, &[&block1 as &dyn Block, &block2 as &dyn Block]);

        assert_eq!(rollup.get_aggr(AggrType::Sum, 0), 6.0);
        assert_eq!(rollup.get_aggr(AggrType::Count, 0), 2.0);
        assert_eq!(rollup.get_aggr(AggrType::Min, 0), 1.0);
        assert_eq!(rollup.get_aggr(AggrType::Max, 0), 5.0);

        assert_eq!(rollup.get_aggr(AggrType::Sum, 1), 2.0);
        assert_eq!(rollup.get_aggr(AggrType::Count, 1), 1.0);
        assert_eq!(rollup.get_aggr(AggrType::Min, 1), 2.0);
        assert_eq!(rollup.get_aggr(AggrType::Max, 1), 2.0);
    }

    #[test]
    fn test_merge_sum() {
        let block1 = ArrayBlock::new(vec![1.0, 2.0, f64::NAN]);
        let block2 = ArrayBlock::new(vec![3.0, f64::NAN, 5.0]);

        let merged = block1.merge(AggrType::Sum, &block2);

        assert_eq!(merged.get(0), 4.0);
        assert!(merged.get(1).is_nan());
        assert!(merged.get(2).is_nan());
    }

    #[test]
    fn test_compress_constant() {
        let block = ConstantBlock::new(100, 42.0);
        let compressed = compress(&block);

        assert_eq!(compressed.size(), 100);
        for i in 0..100 {
            assert_eq!(compressed.get(i), 42.0);
        }
    }

    #[test]
    fn test_compress_sparse() {
        let mut array = ArrayBlock::fill(100, f64::NAN);
        for i in (0..100).step_by(10) {
            array.update(i, (i / 10) as f64);
        }

        let compressed = compress(&array);

        assert_eq!(compressed.size(), 100);
        for i in 0..100 {
            if i % 10 == 0 {
                assert_eq!(compressed.get(i), (i / 10) as f64);
            } else {
                assert!(compressed.get(i).is_nan());
            }
        }
    }

    #[test]
    fn test_lossy_compress() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let block = ArrayBlock::new(values);

        let compressed = lossy_compress(&block);
        assert_eq!(compressed.size(), 100);

        for i in 0..100 {
            let expected = i as f64 * 0.1;
            let actual = compressed.get(i);
            assert!((actual - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_compressed_array_block_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let array = ArrayBlock::new(values);
        let compressed = CompressedArrayBlock::new(5, &array).unwrap();

        assert_eq!(compressed.size(), 5);
        assert_eq!(compressed.get(0), 1.0);
        assert!(compressed.get(1).is_nan());
        assert_eq!(compressed.get(2), 3.0);
        assert!(compressed.get(3).is_nan());
        assert_eq!(compressed.get(4), 5.0);
    }

    #[test]
    fn test_merge_min_max() {
        let block1 = ArrayBlock::new(vec![1.0, 5.0, 3.0]);
        let block2 = ArrayBlock::new(vec![2.0, 4.0, 6.0]);

        let min_merged = block1.merge(AggrType::Min, &block2);
        assert_eq!(min_merged.get(0), 1.0);
        assert_eq!(min_merged.get(1), 4.0);
        assert_eq!(min_merged.get(2), 3.0);

        let max_merged = block1.merge(AggrType::Max, &block2);
        assert_eq!(max_merged.get(0), 2.0);
        assert_eq!(max_merged.get(1), 5.0);
        assert_eq!(max_merged.get(2), 6.0);
    }

    #[test]
    fn test_merge_count() {
        let block1 = ArrayBlock::new(vec![1.0, f64::NAN, 3.0]);
        let block2 = ArrayBlock::new(vec![2.0, 4.0, f64::NAN]);

        let count_merged = block1.merge(AggrType::Count, &block2);
        assert_eq!(count_merged.get(0), 1.0);
        assert!(count_merged.get(1).is_nan());
        assert!(count_merged.get(2).is_nan());
    }

    #[test]
    fn test_sparse_block_too_many_values() {
        let mut array = ArrayBlock::fill(20, 0.0);
        for i in 0..20 {
            array.update(i, i as f64);
        }

        let sparse = SparseBlock::new(20, &array);
        assert!(sparse.is_none());
    }

    #[test]
    fn test_compressed_array_block_4bit() {
        let values = vec![0.0, 5.0, 10.0, 15.0, 7.0, 12.0];
        let array = ArrayBlock::new(values);
        let compressed = CompressedArrayBlock::new(6, &array).unwrap();

        assert_eq!(compressed.bits_per_value, 4);
        assert_eq!(compressed.get(0), 0.0);
        assert_eq!(compressed.get(1), 5.0);
        assert_eq!(compressed.get(2), 10.0);
        assert_eq!(compressed.get(3), 15.0);
        assert_eq!(compressed.get(4), 7.0);
        assert_eq!(compressed.get(5), 12.0);
    }

    #[test]
    fn test_common_values() {
        assert!(COMMON_VALUES[0].is_nan());
        assert_eq!(COMMON_VALUES[1], 0.0);
        assert_eq!(COMMON_VALUES[2], 1.0);
        assert_eq!(COMMON_VALUES[3], 1.0 / 60.0);
    }

    #[test]
    fn test_to_array() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let block = ArrayBlock::new(values.clone());
        let array = block.to_array();

        assert_eq!(array, values);
    }

    #[test]
    fn test_bytesize() {
        let array = ArrayBlock::fill(10, 0.0);
        assert_eq!(array.bytesize(), 16 + 10 * 8);

        let constant = ConstantBlock::new(100, 42.0);
        assert_eq!(constant.bytesize(), 32);

        let float_array = FloatArrayBlock::new(10, &array);
        assert_eq!(float_array.bytesize(), 16 + 10 * 4);
    }

    #[test]
    fn test_compress_small_block() {
        let block = ArrayBlock::new(vec![1.0, 2.0, 3.0]);
        let compressed = compress(&block);

        assert_eq!(compressed.size(), 3);
        assert_eq!(compressed.get(0), 1.0);
        assert_eq!(compressed.get(1), 2.0);
        assert_eq!(compressed.get(2), 3.0);
    }

    #[test]
    fn test_rollup_block_empty() {
        let blocks: Vec<&dyn Block> = vec![];
        let rollup = RollupBlock::new(5, &blocks);

        for i in 0..5 {
            assert!(rollup.get_aggr(AggrType::Sum, i).is_nan());
            assert!(rollup.get_aggr(AggrType::Count, i).is_nan());
            assert!(rollup.get_aggr(AggrType::Min, i).is_nan());
            assert!(rollup.get_aggr(AggrType::Max, i).is_nan());
        }
    }
}