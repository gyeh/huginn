use std::collections::HashMap;
use crate::block::{ArrayBlock, Block, MutableBlock};
use crate::item_id::ItemId;

pub struct BlockStoreItem {
    pub id: ItemId,
    pub tags: HashMap<String, String>,
    pub blocks: MemoryBlockStore,
}

/// Memory-based block store for time series data
pub struct MemoryBlockStore {
    /// Time step in milliseconds
    step: i64,
    /// Number of data points per block
    block_size: usize,
    /// Total number blocks in circular block
    num_blocks: usize,
    /// Total amount of time (in ms) encapsulated by single block
    block_step: i64,
    /// Circular buffer
    blocks: Vec<Option<Box<ArrayBlock>>>,
    /// Current block index within circular buffer
    current_pos: usize,
    /// Does store have data?
    has_data: bool,
}

impl MemoryBlockStore {
    /// Create a new MemoryBlockStore
    ///
    /// # Arguments
    /// * `step` - Time step in milliseconds
    /// * `block_size` - Number of data points per block
    /// * `num_blocks` - Total number of blocks in the circular buffer
    pub fn new(step: i64, block_size: usize, num_blocks: usize) -> Self {
        let block_step = step * block_size as i64;
        let blocks = vec![None; num_blocks];

        Self {
            step,
            block_size,
            num_blocks,
            block_step,
            blocks,
            current_pos: 0,
            has_data: false,
        }
    }

    fn current_block(&self) -> &Option<Box<ArrayBlock>> {
        &self.blocks[self.current_pos]
    }

    /// Get the next position in the circular buffer
    fn next(&self, pos: usize) -> usize {
        (pos + 1) % self.num_blocks
    }

    /// Reposition start time to block boundary
    fn reposition_start(&self, start: i64) -> i64 {
        start - start % self.block_step
    }

    /// Create a new block and update the circular buffer
    fn new_block(&mut self, start: i64) {
        assert!(
            start % self.block_step == 0,
            "start time {} is not on block boundary",
            start
        );

        let new_block= Box::new(ArrayBlock::new(start, self.block_size));

        // self.blocks[self.current_pos] = old_block;
        // self.current_block = Some(new_block);
        self.current_pos = self.next(self.current_pos);
        self.blocks[self.current_pos] = Some(new_block);
        self.has_data = true;
    }

    /// Clean up blocks older than the cutoff time
    pub fn cleanup(&mut self, cutoff: i64) {
        let mut non_empty = false;

        for pos in 0..self.num_blocks {
            if let Some(ref block) = self.blocks[pos] {
                if block.start() < cutoff {
                    self.blocks[pos] = None;
                } else {
                    non_empty = true;
                }
            }
        }
        self.has_data = non_empty;
    }

    /// Update the store with a new data point
    ///
    /// # Arguments
    /// * `timestamp` - Timestamp in milliseconds
    /// * `value` - Value to store
    /// * `rollup` - Whether this is rollup data (currently unused)
    pub fn update(&mut self, timestamp: i64, value: f64, _rollup: bool) {
        // Create block if none exists
        if self.has_data == false {
            let repositioned_start = self.reposition_start(timestamp);
            let current_block = Some(Box::new(ArrayBlock::new(repositioned_start, self.block_size)));
            // self.current_pos = self.next(self.current_pos);
            self.blocks[self.current_pos] = current_block;
            self.has_data = true;
        }

        if let Some(ref mut current) = &mut self.blocks[self.current_pos] {
            // Calculate position to update
            let mut pos = ((timestamp - current.start()) / self.step) as i32;

            if pos >= self.block_size as i32 {
                // Exceeded window of current block, create a new one
                self.new_block(self.reposition_start(timestamp));

                // Recalculate position in the new block
                if let Some(ref mut new_current) =  &mut self.blocks[self.current_pos] {
                    pos = ((timestamp - new_current.start()) / self.step) as i32;
                    new_current.update(pos as usize, value);
                }
            } else if pos < 0 {
                // Out of order update for an older block
                let previous_pos = if self.current_pos == 0 {
                    self.num_blocks - 1
                } else {
                    self.current_pos - 1
                };

                if let Some(ref mut previous_block) = self.blocks[previous_pos] {
                    pos = ((timestamp - previous_block.start()) / self.step) as i32;
                    if pos >= 0 && (pos as usize) < self.block_size {
                        previous_block.update(pos as usize, value);
                    }
                }
            } else {
                current.update(pos as usize, value);
            }
        }
    }

    /// Update with a list of values starting at a given time
    pub fn update_values(&mut self, start: i64, values: &[f64]) {
        let mut t = start;
        for &v in values {
            self.update(t, v, false);
            t += self.step;
        }
    }

    /// Fill buffer with data from a block
    fn fill(&self, blk: &dyn Block, buf: &mut [f64], start: i64, end: i64) {
        let s = start / self.step;
        let e = end / self.step;
        let bs = blk.start() / self.step;
        let be = bs + self.block_size as i64 - 1;

        if e >= bs && s <= be {
            let spos = s.max(bs);
            let epos = e.min(be);

            for i in spos..=epos {
                buf[(i - s) as usize] = blk.get((i - bs) as usize);
            }
        }
    }

    /// Get list of all non-null blocks
    pub fn block_list(&self) -> Vec<Box<dyn Block>> {
        self.blocks
            .iter()
            .filter_map(|b| b.as_ref().map(|block| block.clone_box()))
            .collect()
    }

    /// Fetch data for a time range
    ///
    /// # Arguments
    /// * `start` - Start time in milliseconds
    /// * `end` - End time in milliseconds
    /// * `aggr` - Aggregation type (passed to Block::get_with_aggr)
    pub fn fetch(&self, start: i64, end: i64, _aggr: i32) -> Vec<f64> {
        let size = ((end / self.step - start / self.step) + 1) as usize;
        let mut buffer = vec![f64::NAN; size];

        for block_opt in &self.blocks {
            if let Some(ref block) = block_opt {
                self.fill(block.as_ref(), &mut buffer, start, end);
            }
        }

        buffer
    }

    /// Get a block by its start time
    pub fn get(&self, start: i64) -> Option<Box<dyn Block>> {
        self.blocks
            .iter()
            .find_map(|block_opt| {
                block_opt.as_ref().and_then(|block| {
                    if block.start() == start {
                        Some(block.clone_box())
                    } else {
                        None
                    }
                })
            })
    }

    /// Check if the store has any data
    pub fn has_data(&self) -> bool {
        self.has_data
    }

    /// Get the time step
    pub fn step(&self) -> i64 {
        self.step
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl std::fmt::Display for MemoryBlockStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, block_opt) in self.blocks.iter().enumerate() {
            match block_opt {
                Some(block) => {
                    write!(f, "{} => Block(start={})", i, block.start())?;
                }
                None => {
                    write!(f, "{} => None", i)?;
                }
            }

            if i == self.current_pos {
                write!(f, " (current)")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_block_store_basic() {
        let mut store = MemoryBlockStore::new(1000, 60, 5);

        // Add some data
        store.update(0, 42.0, false);
        store.update(1000, 43.0, false);
        store.update(2000, 44.0, false);

        assert!(store.has_data());

        // Fetch data
        let data = store.fetch(0, 3000, 0);
        assert_eq!(data[0], 42.0);
        assert_eq!(data[1], 43.0);
        assert_eq!(data[2], 44.0);
    }

    #[test]
    fn test_circular_buffer() {
        let mut store = MemoryBlockStore::new(1000, 2, 3); // Small blocks and buffer for testing

        // Fill up blocks
        store.update(0, 1.0, false);
        store.update(2000, 2.0, false); // New block
        store.update(4000, 3.0, false); // New block
        store.update(6000, 4.0, false); // Should wrap around

        // Check that we can still access recent data
        let data = store.fetch(4000, 6000, 0);
        assert_eq!(data[0], 3.0);
        assert_eq!(data[2], 4.0);
    }

    #[test]
    fn test_cleanup() {
        let mut store = MemoryBlockStore::new(1000, 60, 5);

        store.update(0, 1.0, false);
        store.update(60000, 2.0, false);
        store.update(120000, 3.0, false);

        // Cleanup old blocks
        store.cleanup(60000);

        // First block should be gone
        assert!(store.get(0).is_none());
        assert!(store.get(60000).is_some());
        assert!(store.get(120000).is_some());
    }

    #[test]
    fn test_out_of_order_updates() {
        let mut store = MemoryBlockStore::new(1000, 60, 5);

        // Create first block
        store.update(0, 1.0, false);
        store.update(1000, 2.0, false);

        // Create second block
        store.update(60000, 3.0, false);

        // Try to update previous block
        store.update(2000, 2.5, false);

        let data = store.fetch(0, 3000, 0);
        assert_eq!(data[2], 2.5);
    }

    #[test]
    fn test_update_values() {
        let mut store = MemoryBlockStore::new(1000, 60, 5);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        store.update_values(0, &values);

        let data = store.fetch(0, 4000, 0);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 5.0);
    }

    #[test]
    fn test_block_alignment() {
        let store = MemoryBlockStore::new(1000, 60, 5);

        // block_step = 1000 * 60 = 60000
        assert_eq!(store.reposition_start(0), 0);
        assert_eq!(store.reposition_start(30000), 0);
        assert_eq!(store.reposition_start(60000), 60000);
        assert_eq!(store.reposition_start(90000), 60000);
        assert_eq!(store.reposition_start(120000), 120000);
    }
}