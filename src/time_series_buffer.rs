// use std::collections::HashMap;
// use crate::time_series::{time_series, ItemId, TimeSeries};
// use crate::block::{Block, ConstantBlock, aggregate_type};
// use crate::consolidation_fn::ConsolidationFunction;
// use crate::ds_type::DsType;
// use crate::time_seq::{ArrayTimeSeq, TimeSeq};
// use crate::time_seq::math::{add_nan, max_nan, min_nan};
//
// #[derive(Debug)]
// pub struct TimeSeriesBuffer {
//     pub tags: HashMap<String, String>,
//     pub data: ArrayTimeSeq,
//     id: Option<ItemId>,
// }
//
// impl TimeSeriesBuffer {
//     pub fn new_with_values(
//         tags: HashMap<String, String>,
//         step: i64,
//         start: i64,
//         values: Vec<f64>,
//     ) -> Self {
//         let data = ArrayTimeSeq::new(DsType::from_tags(&tags), start, step, values);
//         Self {
//             tags,
//             data,
//             id: None,
//         }
//     }
//
//     pub fn new_empty(
//         tags: HashMap<String, String>,
//         step: i64,
//         start: i64,
//         end: i64,
//     ) -> Self {
//         let s = start / step;
//         let e = end / step;
//         let size = ((e - s) as usize) + 1;
//         let values = vec![f64::NAN; size];
//         Self::new_with_values(tags, step, s * step, values)
//     }
//
//     pub fn new_with_blocks(
//         tags: HashMap<String, String>,
//         step: i64,
//         start: i64,
//         end: i64,
//         blocks: &[Box<dyn Block>],
//         aggr: i32,
//     ) -> Self {
//         let s = start / step;
//         let e = end / step;
//         let size = ((e - s) as usize) + 1;
//         let mut buffer = vec![f64::NAN; size];
//
//         for block in blocks {
//             Self::fill_buffer(&mut buffer, block.as_ref(), step, s, e, aggr);
//         }
//
//         Self::new_with_values(tags, step, start, buffer)
//     }
//
//     fn fill_buffer(
//         buf: &mut [f64],
//         blk: &dyn Block,
//         step: i64,
//         s: i64,
//         e: i64,
//         aggr: i32,
//     ) {
//         let bs = blk.start() / step;
//         let be = bs + (blk.size() as i64) - 1;
//
//         if e >= bs && s <= be {
//             let spos = if s > bs { s } else { bs };
//             let epos = if e < be { e } else { be };
//
//             let mut i = spos;
//             while i <= epos {
//                 let buf_idx = (i - s) as usize;
//                 let blk_idx = (i - bs) as usize;
//                 if buf_idx < buf.len() && blk_idx < blk.size() {
//                     buf[buf_idx] = blk.get_with_aggr(blk_idx, aggr);
//                 }
//                 i += 1;
//             }
//         }
//     }
//
//     pub fn is_all_nan(&self) -> bool {
//         self.data.data.iter().all(|&v| v.is_nan())
//     }
//
//     pub fn copy_buffer(&self) -> Self {
//         Self {
//             tags: self.tags.clone(),
//             data: ArrayTimeSeq::new(
//                 self.data.ds_type(),
//                 self.data.start,
//                 self.data.step,
//                 self.data.data.clone()
//             ),
//             id: self.id.clone(),
//         }
//     }
//
//     // Aggregate the data from a block into this buffer
//     pub fn aggr_block(
//         &mut self,
//         block: &dyn Block,
//         aggr: i32,
//         cf: ConsolidationFunction,
//         multiple: usize,
//         op: fn(f64, f64) -> f64,
//     ) -> usize {
//         let s = self.data.start / self.data.step();
//         let e = self.data.data.len() as i64 + s - 1;
//         let bs = block.start() / self.data.step();
//         let be = bs + (block.size() / multiple) as i64 - 1;
//         let mut value_count = 0;
//
//         if e >= bs && s <= be {
//             let spos = if s > bs { s } else { bs };
//             let epos = if e < be { e } else { be };
//             let mut i = spos;
//             let mut j = ((i - bs) as usize) * multiple;
//
//             while i <= epos {
//                 let pos = (i - s) as usize;
//                 if pos < self.data.data.len() && j < block.size() {
//                     let v = self.compute_cf_value(&cf, block, j, aggr, multiple);
//                     self.data.data[pos] = op(self.data.data[pos], v);
//                     if !self.data.data[pos].is_nan() {
//                         value_count += 1;
//                     }
//                 }
//                 i += 1;
//                 j += multiple;
//             }
//         }
//         value_count
//     }
//
//     fn compute_cf_value(
//         &self,
//         cf: &ConsolidationFunction,
//         block: &dyn Block,
//         j: usize,
//         aggr: i32,
//         multiple: usize,
//     ) -> f64 {
//         match cf {
//             ConsolidationFunction::Sum => {
//                 let mut sum = 0.0;
//                 let mut has_value = false;
//                 for k in 0..multiple {
//                     if j + k < block.size() {
//                         let v = block.get_with_aggr(j + k, aggr);
//                         if !v.is_nan() {
//                             sum += v;
//                             has_value = true;
//                         }
//                     }
//                 }
//                 if has_value { sum } else { f64::NAN }
//             }
//             ConsolidationFunction::Max => {
//                 let mut max = f64::NEG_INFINITY;
//                 let mut has_value = false;
//                 for k in 0..multiple {
//                     if j + k < block.size() {
//                         let v = block.get_with_aggr(j + k, aggr);
//                         if !v.is_nan() {
//                             max = max.max(v);
//                             has_value = true;
//                         }
//                     }
//                 }
//                 if has_value { max } else { f64::NAN }
//             }
//             ConsolidationFunction::Min => {
//                 let mut min = f64::INFINITY;
//                 let mut has_value = false;
//                 for k in 0..multiple {
//                     if j + k < block.size() {
//                         let v = block.get_with_aggr(j + k, aggr);
//                         if !v.is_nan() {
//                             min = min.min(v);
//                             has_value = true;
//                         }
//                     }
//                 }
//                 if has_value { min } else { f64::NAN }
//             }
//             ConsolidationFunction::Avg => {
//                 let mut sum = 0.0;
//                 let mut count = 0;
//                 for k in 0..multiple {
//                     if j + k < block.size() {
//                         let v = block.get_with_aggr(j + k, aggr);
//                         if !v.is_nan() {
//                             sum += v;
//                             count += 1;
//                         }
//                     }
//                 }
//                 if count > 0 { sum / count as f64 } else { f64::NAN }
//             }
//         }
//     }
//
//     pub fn add_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             self.data.data[i] = add_nan(self.data.data[i], normalized.data.data[i]);
//         }
//     }
//
//     pub fn add_block(&mut self, block: &dyn Block) -> usize {
//         self.aggr_block(block, aggregate_type::SUM, ConsolidationFunction::Sum, 1, add_nan)
//     }
//
//     pub fn add_constant(&mut self, value: f64) {
//         for v in &mut self.data.data {
//             *v = add_nan(*v, value);
//         }
//     }
//
//     pub fn subtract_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             self.data.data[i] = self.data.data[i] - normalized.data.data[i];
//         }
//     }
//
//     pub fn subtract_constant(&mut self, value: f64) {
//         for v in &mut self.data.data {
//             *v = *v - value;
//         }
//     }
//
//     pub fn multiply_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             self.data.data[i] = self.data.data[i] * normalized.data.data[i];
//         }
//     }
//
//     pub fn multiply_constant(&mut self, value: f64) {
//         for v in &mut self.data.data {
//             *v = *v * value;
//         }
//     }
//
//     pub fn divide_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             self.data.data[i] = self.data.data[i] / normalized.data.data[i];
//         }
//     }
//
//     pub fn divide_constant(&mut self, value: f64) {
//         for v in &mut self.data.data {
//             *v = *v / value;
//         }
//     }
//
//     pub fn max_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             self.data.data[i] = max_nan(self.data.data[i], normalized.data.data[i]);
//         }
//     }
//
//     pub fn max_block(&mut self, block: &dyn Block) -> usize {
//         self.aggr_block(block, aggregate_type::MAX, ConsolidationFunction::Sum, 1, max_nan)
//     }
//
//     pub fn min_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             self.data.data[i] = min_nan(self.data.data[i], normalized.data.data[i]);
//         }
//     }
//
//     pub fn min_block(&mut self, block: &dyn Block) -> usize {
//         self.aggr_block(block, aggregate_type::MIN, ConsolidationFunction::Sum, 1, min_nan)
//     }
//
//     pub fn init_count(&mut self) {
//         for v in &mut self.data.data {
//             *v = if v.is_nan() { 0.0 } else { 1.0 };
//         }
//     }
//
//     pub fn count_buffer(&mut self, other: &TimeSeriesBuffer) {
//         let normalized = other.normalize(self.data.step(), self.data.start, self.data.data.len());
//         let length = self.data.data.len().min(normalized.data.data.len());
//
//         for i in 0..length {
//             let v2 = if normalized.data.data[i].is_nan() { 0.0 } else { 1.0 };
//             self.data.data[i] = self.data.data[i] + v2;
//         }
//     }
//
//     pub fn count_block(&mut self, block: &dyn Block) -> usize {
//         self.aggr_block(block, aggregate_type::COUNT, ConsolidationFunction::Sum, 1, add_nan)
//     }
//
//     pub fn merge(&mut self, other: &TimeSeriesBuffer) {
//         if self.data.step() != other.data.step() {
//             panic!("step sizes must be the same");
//         }
//         if self.data.start != other.data.start {
//             panic!("start times must be the same");
//         }
//
//         let length = self.data.data.len().min(other.data.data.len());
//         for i in 0..length {
//             let v1 = self.data.data[i];
//             let v2 = other.data.data[i];
//             if v1.is_nan() || v1 < v2 {
//                 self.data.data[i] = v2;
//             }
//         }
//     }
//
//     // // Returns new buf with values consolidate to a larger step size
//     // pub fn consolidate(&self, multiple: usize, cf: ConsolidationFunction) -> TimeSeriesBuffer {
//     //     let new_step = self.data.step() * multiple as i64;
//     //     let consolidated_data = ArrayTimeSeq::new(&self.data, new_step, cf);
//     //
//     //     TimeSeriesBuffer {
//     //         tags: self.tags.clone(),
//     //         data: consolidated_data,
//     //         id: self.id.clone(),
//     //     }
//     // }
//     //
//     // pub fn normalize(&self, step: i64, start: i64, size: usize) -> TimeSeriesBuffer {
//     //     let buf = if step > self.data.step() {
//     //         self.consolidate((step / self.data.step()) as usize, ConsolidationFunction::Avg)
//     //     } else {
//     //         self.clone()
//     //     };
//     //
//     //     if buf.data.start == start && buf.data.step() == step {
//     //         buf
//     //     } else {
//     //         let aligned_start = start / step * step;
//     //         let mut buffer = vec![f64::NAN; size];
//     //         for i in 0..size {
//     //             buffer[i] = buf.get_value(aligned_start + i as i64 * step);
//     //         }
//     //         TimeSeriesBuffer {
//     //             tags: self.tags.clone(),
//     //             data: ArrayTimeSeq::new(
//     //                 self.data.ds_type(),
//     //                 aligned_start,
//     //                 step,
//     //                 buffer
//     //             ),
//     //             id: None,
//     //         }
//     //     }
//     // }
//
//     pub fn get_value(&self, timestamp: i64) -> f64 {
//         let offset = timestamp - self.data.start;
//         let pos = (offset / self.data.step()) as usize;
//         if offset < 0 || pos >= self.data.data.len() {
//             f64::NAN
//         } else {
//             self.data.data[pos]
//         }
//     }
//
//     pub fn values(&self) -> &Vec<f64> {
//         &self.data.data
//     }
//
//     pub fn step(&self) -> i64 {
//         self.data.step()
//     }
//
//     pub fn start(&self) -> i64 {
//         self.data.start
//     }
//
//     pub fn ds_type(&self) -> DsType {
//         self.data.ds_type()
//     }
//
//     pub fn label(&self) -> String {
//         time_series::to_label(&self.tags)
//     }
// }
//
// impl Clone for TimeSeriesBuffer {
//     fn clone(&self) -> Self {
//         Self {
//             tags: self.tags.clone(),
//             data: self.data.clone(),
//             id: self.id.clone(),
//         }
//     }
// }
//
// impl PartialEq for TimeSeriesBuffer {
//     fn eq(&self, other: &Self) -> bool {
//         self.tags == other.tags && self.data.data == other.data.data &&
//             self.data.start == other.data.start && self.data.step() == other.data.step() &&
//             self.data.ds_type() == other.data.ds_type()
//     }
// }
//
// // Static methods for aggregating buffers
// impl TimeSeriesBuffer {
//     pub fn sum(mut buffers: Vec<TimeSeriesBuffer>) -> Option<TimeSeriesBuffer> {
//         if buffers.is_empty() {
//             return None;
//         }
//
//         let mut result = buffers.remove(0);
//         for buffer in buffers {
//             result.add_buffer(&buffer);
//         }
//         Some(result)
//     }
//
//     pub fn max(mut buffers: Vec<TimeSeriesBuffer>) -> Option<TimeSeriesBuffer> {
//         if buffers.is_empty() {
//             return None;
//         }
//
//         let mut result = buffers.remove(0);
//         for buffer in buffers {
//             result.max_buffer(&buffer);
//         }
//         Some(result)
//     }
//
//     pub fn min(mut buffers: Vec<TimeSeriesBuffer>) -> Option<TimeSeriesBuffer> {
//         if buffers.is_empty() {
//             return None;
//         }
//
//         let mut result = buffers.remove(0);
//         for buffer in buffers {
//             result.min_buffer(&buffer);
//         }
//         Some(result)
//     }
//
//     pub fn count(mut buffers: Vec<TimeSeriesBuffer>) -> Option<TimeSeriesBuffer> {
//         if buffers.is_empty() {
//             return None;
//         }
//
//         let mut result = buffers.remove(0);
//         result.init_count();
//
//         for buffer in buffers {
//             result.count_buffer(&buffer);
//         }
//         Some(result)
//     }
// }
//
// fn empty_tags() -> HashMap<String, String> {
//     HashMap::new()
// }
//
// fn new_buffer(value: f64, step: i64, start: i64, n: usize) -> TimeSeriesBuffer {
//     TimeSeriesBuffer::new_with_values(empty_tags(), step, start, vec![value; n])
// }
//
// fn new_buffer_simple(value: f64) -> TimeSeriesBuffer {
//     new_buffer(value, 60000, 0, 1)
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_apply_list_block() {
//         let tags = empty_tags();
//         let step = 60000i64;
//         let blocks = vec![
//             Box::new(ConstantBlock::new(0 * step, 6, 1.0)),
//             Box::new(ConstantBlock::new(6 * step, 6, 2.0)),
//             Box::new(ConstantBlock::new(18 * step, 6, 4.0)),
//         ];
//
//         let buffer = TimeSeriesBuffer::new_with_blocks(
//             tags,
//             step,
//             1 * step,
//             19 * step,
//             &blocks,
//             aggregate_type::SUM,
//         );
//
//         assert_eq!(buffer.step(), step);
//         assert_eq!(buffer.start(), step);
//
//         let values = buffer.values();
//         assert!(values[0..5].iter().all(|&v| v == 1.0));
//         assert!(values[5..11].iter().all(|&v| v == 2.0));
//         assert!(values[11..17].iter().all(|&v| v.is_nan()));
//         assert!(values[17..].iter().all(|&v| v == 4.0));
//     }
//
//     #[test]
//     fn test_add_block() {
//         let tags = empty_tags();
//         let step = 60000i64;
//         let blocks = vec![
//             Box::new(ConstantBlock::new(0 * step, 6, 1.0)),
//             Box::new(ConstantBlock::new(6 * step, 6, 2.0)),
//             Box::new(ConstantBlock::new(18 * step, 6, 4.0)),
//         ];
//
//         let mut buffer = TimeSeriesBuffer::new_empty(tags, step, 1 * step, 19 * step);
//         for block in &blocks {
//             buffer.add_block(block.as_ref());
//         }
//
//         assert_eq!(buffer.step(), step);
//         assert_eq!(buffer.start(), step);
//
//         let values = buffer.values();
//         assert!(values[0..5].iter().all(|&v| v == 1.0));
//         assert!(values[5..11].iter().all(|&v| v == 2.0));
//         assert!(values[11..17].iter().all(|&v| v.is_nan()));
//         assert!(values[17..].iter().all(|&v| v == 4.0));
//     }
//
//     #[test]
//     fn test_add_buffer() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(42.0);
//         b1.add_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 84.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_add_buffer_b1_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(42.0);
//         b1.add_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_add_buffer_b2_nan() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.add_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_add_buffer_both_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.add_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v.is_nan()));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_add_constant() {
//         let mut b1 = new_buffer_simple(42.0);
//         b1.add_constant(42.0);
//         assert!(b1.values().iter().all(|&v| v == 84.0));
//     }
//
//     #[test]
//     fn test_add_constant_b1_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         b1.add_constant(42.0);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_add_constant_v_nan() {
//         let mut b1 = new_buffer_simple(42.0);
//         b1.add_constant(f64::NAN);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_add_constant_both_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         b1.add_constant(f64::NAN);
//         assert!(b1.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_subtract_buffer() {
//         let mut b1 = new_buffer_simple(84.0);
//         let b2 = new_buffer_simple(42.0);
//         b1.subtract_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_subtract_constant() {
//         let mut b1 = new_buffer_simple(84.0);
//         b1.subtract_constant(42.0);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_multiply_buffer() {
//         let mut b1 = new_buffer_simple(84.0);
//         let b2 = new_buffer_simple(2.0);
//         b1.multiply_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 168.0));
//         assert!(b2.values().iter().all(|&v| v == 2.0));
//     }
//
//     #[test]
//     fn test_multiply_constant() {
//         let mut b1 = new_buffer_simple(84.0);
//         b1.multiply_constant(2.0);
//         assert!(b1.values().iter().all(|&v| v == 168.0));
//     }
//
//     #[test]
//     fn test_divide_buffer() {
//         let mut b1 = new_buffer_simple(84.0);
//         let b2 = new_buffer_simple(2.0);
//         b1.divide_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 2.0));
//     }
//
//     #[test]
//     fn test_divide_constant() {
//         let mut b1 = new_buffer_simple(84.0);
//         b1.divide_constant(2.0);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_max_buffer_b1_greater() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(21.0);
//         b1.max_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 21.0));
//     }
//
//     #[test]
//     fn test_max_buffer_b1_less() {
//         let mut b1 = new_buffer_simple(21.0);
//         let b2 = new_buffer_simple(42.0);
//         b1.max_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_max_buffer_b1_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(42.0);
//         b1.max_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_max_buffer_b2_nan() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.max_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_max_buffer_both_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.max_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v.is_nan()));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_min_buffer_b1_greater() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(21.0);
//         b1.min_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 21.0));
//         assert!(b2.values().iter().all(|&v| v == 21.0));
//     }
//
//     #[test]
//     fn test_min_buffer_b1_less() {
//         let mut b1 = new_buffer_simple(21.0);
//         let b2 = new_buffer_simple(42.0);
//         b1.min_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 21.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_min_buffer_b1_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(42.0);
//         b1.min_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_min_buffer_b2_nan() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.min_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 42.0));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_min_buffer_both_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.min_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v.is_nan()));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_count_buffer() {
//         let mut b1 = new_buffer_simple(21.0);
//         let b2 = new_buffer_simple(42.0);
//         b1.init_count();
//         b1.count_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 2.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_count_buffer_b1_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(42.0);
//         b1.init_count();
//         b1.count_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 1.0));
//         assert!(b2.values().iter().all(|&v| v == 42.0));
//     }
//
//     #[test]
//     fn test_count_buffer_b2_nan() {
//         let mut b1 = new_buffer_simple(42.0);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.init_count();
//         b1.count_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 1.0));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_count_buffer_both_nan() {
//         let mut b1 = new_buffer_simple(f64::NAN);
//         let b2 = new_buffer_simple(f64::NAN);
//         b1.init_count();
//         b1.count_buffer(&b2);
//         assert!(b1.values().iter().all(|&v| v == 0.0));
//         assert!(b2.values().iter().all(|&v| v.is_nan()));
//     }
//
//     #[test]
//     fn test_get_value_prior_to_start() {
//         let b1 = new_buffer(42.0, 60000, 120000, 1);
//         assert!(b1.get_value(60000).is_nan());
//     }
//
//     #[test]
//     fn test_get_value_after_end() {
//         let b1 = new_buffer(42.0, 60000, 120000, 1);
//         assert!(b1.get_value(240000).is_nan());
//     }
//
//     #[test]
//     fn test_get_value_with_match() {
//         let b1 = new_buffer(42.0, 60000, 120000, 1);
//         assert_eq!(b1.get_value(120000), 42.0);
//     }
//
//     #[test]
//     fn test_sum() {
//         let buffers = vec![
//             new_buffer_simple(1.0),
//             new_buffer_simple(f64::NAN),
//             new_buffer_simple(2.0),
//         ];
//         let result = TimeSeriesBuffer::sum(buffers).unwrap();
//         assert_eq!(result.values()[0], 3.0);
//     }
//
//     #[test]
//     fn test_sum_empty() {
//         let buffers = vec![];
//         assert!(TimeSeriesBuffer::sum(buffers).is_none());
//     }
//
//     #[test]
//     fn test_max() {
//         let buffers = vec![
//             new_buffer_simple(1.0),
//             new_buffer_simple(f64::NAN),
//             new_buffer_simple(2.0),
//         ];
//         let result = TimeSeriesBuffer::max(buffers).unwrap();
//         assert_eq!(result.values()[0], 2.0);
//     }
//
//     #[test]
//     fn test_max_empty() {
//         let buffers = vec![];
//         assert!(TimeSeriesBuffer::max(buffers).is_none());
//     }
//
//     #[test]
//     fn test_min() {
//         let buffers = vec![
//             new_buffer_simple(1.0),
//             new_buffer_simple(f64::NAN),
//             new_buffer_simple(2.0),
//         ];
//         let result = TimeSeriesBuffer::min(buffers).unwrap();
//         assert_eq!(result.values()[0], 1.0);
//     }
//
//     #[test]
//     fn test_min_empty() {
//         let buffers = vec![];
//         assert!(TimeSeriesBuffer::min(buffers).is_none());
//     }
//
//     #[test]
//     fn test_count() {
//         let buffers = vec![
//             new_buffer_simple(1.0),
//             new_buffer_simple(f64::NAN),
//             new_buffer_simple(2.0),
//         ];
//         let result = TimeSeriesBuffer::count(buffers).unwrap();
//         assert_eq!(result.values()[0], 2.0);
//     }
//
//     #[test]
//     fn test_count_empty() {
//         let buffers = vec![];
//         assert!(TimeSeriesBuffer::count(buffers).is_none());
//     }
//
//     #[test]
//     fn test_merge_diff_sizes_b1_smaller() {
//         let mut b1 = new_buffer(1.0, 60000, 0, 1);
//         let b2 = new_buffer(2.0, 60000, 0, 2);
//         b1.merge(&b2);
//         assert_eq!(b1.values()[0], 2.0);
//     }
//
//     #[test]
//     fn test_merge_diff_sizes_b1_larger() {
//         let b1 = new_buffer(7.0, 60000, 0, 1);
//         let mut b2 = new_buffer(2.0, 60000, 0, 2);
//         b2.merge(&b1);
//         assert_eq!(b2.values()[0], 7.0);
//         assert_eq!(b2.values()[1], 2.0);
//     }
//
//     // #[test]
//     // fn test_consolidate() {
//     //     let start = 1366746900000i64;
//     //     let b = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         60000,
//     //         start,
//     //         vec![1.0, 2.0, 3.0, 4.0, 5.0],
//     //     );
//     //
//     //     let b2 = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         120000,
//     //         start,
//     //         vec![1.0, 5.0, 9.0],
//     //     );
//     //     assert_eq!(b.consolidate(2, ConsolidationFunction::Sum), b2);
//     //
//     //     let b3 = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         180000,
//     //         start,
//     //         vec![3.0, 12.0],
//     //     );
//     //     assert_eq!(b.consolidate(3, ConsolidationFunction::Sum), b3);
//     //
//     //     let b4 = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         240000,
//     //         start,
//     //         vec![1.0, 14.0],
//     //     );
//     //     assert_eq!(b.consolidate(4, ConsolidationFunction::Sum), b4);
//     //
//     //     let b5 = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         300000,
//     //         start,
//     //         vec![15.0],
//     //     );
//     //     assert_eq!(b.consolidate(5, ConsolidationFunction::Sum), b5);
//     // }
//
//     // #[test]
//     // fn test_consolidate_nan_avg_with_rate() {
//     //     let start = 1366746900000i64;
//     //     let mut tags = HashMap::new();
//     //     tags.insert("dsType".to_string(), "rate".to_string());
//     //
//     //     let b = TimeSeriesBuffer::new_with_values(
//     //         tags.clone(),
//     //         60000,
//     //         start,
//     //         vec![1.0, 2.0, f64::NAN, 4.0, 5.0],
//     //     );
//     //
//     //     let b5 = TimeSeriesBuffer::new_with_values(
//     //         tags,
//     //         300000,
//     //         start,
//     //         vec![12.0 / 5.0],
//     //     );
//     //     assert_eq!(b.consolidate(5, ConsolidationFunction::Avg), b5);
//     // }
//
//     // #[test]
//     // fn test_consolidate_nan_avg_with_gauge() {
//     //     let start = 1366746900000i64;
//     //     let mut tags = HashMap::new();
//     //     tags.insert("dsType".to_string(), "gauge".to_string());
//     //
//     //     let b = TimeSeriesBuffer::new_with_values(
//     //         tags.clone(),
//     //         60000,
//     //         start,
//     //         vec![1.0, 2.0, f64::NAN, 4.0, 5.0],
//     //     );
//     //
//     //     let b5 = TimeSeriesBuffer::new_with_values(
//     //         tags,
//     //         300000,
//     //         start,
//     //         vec![12.0 / 4.0],
//     //     );
//     //     assert_eq!(b.consolidate(5, ConsolidationFunction::Avg), b5);
//     // }
//
//     // #[test]
//     // fn test_normalize() {
//     //     let start = 1366746900000i64;
//     //     let b1 = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         60000,
//     //         start,
//     //         vec![1.0, 2.0, 3.0, 4.0, 5.0],
//     //     );
//     //     let b1e = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         120000,
//     //         start,
//     //         vec![0.5, 2.5, 4.5],
//     //     );
//     //     assert_eq!(b1.normalize(60000, start, 5), b1);
//     //     assert_eq!(b1.normalize(120000, start, 3), b1e);
//     //
//     //     let b2 = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         120000,
//     //         start,
//     //         vec![3.0, 7.0],
//     //     );
//     //     let b2e = TimeSeriesBuffer::new_with_values(
//     //         empty_tags(),
//     //         60000,
//     //         start,
//     //         vec![3.0, 7.0, 7.0, f64::NAN, f64::NAN],
//     //     );
//     //     assert_eq!(b2.normalize(60000, start, 5), b2e);
//     // }
//
//     // #[test]
//     // fn test_normalize_gauge() {
//     //     let start = 1366746900000i64;
//     //     let mut tags = HashMap::new();
//     //     tags.insert("dsType".to_string(), "gauge".to_string());
//     //
//     //     let b1 = TimeSeriesBuffer::new_with_values(
//     //         tags.clone(),
//     //         60000,
//     //         start,
//     //         vec![1.0, 2.0, 3.0, 4.0, 5.0],
//     //     );
//     //     let b1e = TimeSeriesBuffer::new_with_values(
//     //         tags.clone(),
//     //         120000,
//     //         start,
//     //         vec![1.0, 2.5, 4.5],
//     //     );
//     //     assert_eq!(b1.normalize(60000, start, 5), b1);
//     //     assert_eq!(b1.normalize(120000, start, 3), b1e);
//     //
//     //     let b2 = TimeSeriesBuffer::new_with_values(
//     //         tags.clone(),
//     //         120000,
//     //         start,
//     //         vec![3.0, 7.0],
//     //     );
//     //     let b2e = TimeSeriesBuffer::new_with_values(
//     //         tags,
//     //         60000,
//     //         start,
//     //         vec![3.0, 7.0, 7.0, f64::NAN, f64::NAN],
//     //     );
//     //     assert_eq!(b2.normalize(60000, start, 5), b2e);
//     // }
//
//     #[test]
//     fn test_is_all_nan() {
//         let b1 = new_buffer_simple(f64::NAN);
//         assert!(b1.is_all_nan());
//
//         let b2 = new_buffer_simple(42.0);
//         assert!(!b2.is_all_nan());
//     }
//
//     #[test]
//     fn test_copy_buffer() {
//         let b1 = new_buffer_simple(42.0);
//         let b2 = b1.copy_buffer();
//         assert_eq!(b1, b2);
//         assert_eq!(b1.values(), b2.values());
//     }
// }
