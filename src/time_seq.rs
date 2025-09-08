use std::cmp;
use std::fmt;
use std::hash::{Hash, Hasher};
use crate::ds_type::DsType;

pub type BinaryOp = fn(f64, f64) -> f64;

pub trait TimeSeq {
    fn ds_type(&self) -> DsType;
    fn step(&self) -> i64;
    fn apply(&self, timestamp: i64) -> f64;

    fn map_values(&self, f: Box<dyn Fn(f64) -> f64>) -> Box<dyn TimeSeq> {
        Box::new(UnaryOpTimeSeq::new(Box::new(self.clone_box()), f))
    }

    /// Fast loop with no intermediate object creation
    fn foreach<F>(&self, s: i64, e: i64, mut f: F)
    where
        F: FnMut(i64, f64)
    {
        assert!(s <= e, "start must be <= end");
        let step = self.step();
        let end = e / step * step;
        let mut t = s / step * step;
        while t < end {
            f(t, self.apply(t));
            t += step;
        }
    }

    // fn foreach(&self, s: i64, e: i64, f: &mut dyn FnMut(i64, f64))
    // {
    //     assert!(s <= e, "start must be <= end");
    //     let step = self.step();
    //     let end = e / step * step;
    //     let mut t = s / step * step;
    //     while t < end {
    //         f(t, self.apply(t));
    //         t += step;
    //     }
    // }

    fn bounded(&self, s: i64, e: i64) -> ArrayTimeSeq {
        assert!(s <= e, "start must be <= end");
        let step = self.step();
        let end = e / step * step;
        let start = s / step * step;
        let length = ((end - start) / step) as usize;
        let mut data = vec![0.0; length];
        let mut i = 0;
        self.foreach(start, end, |_, v| {
            data[i] = v;
            i += 1;
        });
        ArrayTimeSeq::new(self.ds_type(), start, step, data)
    }

    fn clone_box(&self) -> Box<dyn TimeSeq>;
}

#[derive(Clone, Debug)]
pub struct ArrayTimeSeq {
    pub ds_type: DsType,
    pub start: i64,
    pub step: i64,
    pub data: Vec<f64>,
}

impl ArrayTimeSeq {
    pub fn new(ds_type: DsType, start: i64, step: i64, data: Vec<f64>) -> Self {
        assert!(start % step == 0, "start time must be on step boundary");
        ArrayTimeSeq {
            ds_type,
            start,
            step,
            data,
        }
    }

    pub fn end(&self) -> i64 {
        self.start + (self.data.len() as i64) * self.step
    }

    /// Update with another ArrayTimeSeq using a binary operation
    pub fn update_with_array(&mut self, ts: &ArrayTimeSeq, op: BinaryOp) {
        assert_eq!(self.step, ts.step, "step sizes must be the same");
        let s = cmp::max(self.start, ts.start);
        let e = cmp::min(self.end(), ts.end());
        if s < e {
            let mut i1 = ((s - self.start) / self.step) as usize;
            let mut i2 = ((s - ts.start) / self.step) as usize;
            let epos = ((e - self.start) / self.step) as usize;
            while i1 < epos {
                self.data[i1] = op(self.data[i1], ts.data[i2]);
                i1 += 1;
                i2 += 1;
            }
        }
    }

    /// Update with any TimeSeq using a binary operation
    pub fn update_with_seq(&mut self, ts: &dyn TimeSeq, op: BinaryOp) {
        assert_eq!(self.step, ts.step(), "step sizes must be the same");
        let mut i = 0;
        ts.foreach(self.start, self.end(), |_, v| {
            self.data[i] = op(self.data[i], v);
            i += 1;
        });
    }

    /// Update all values in place
    pub fn update<F>(&mut self, op: F)
    where
        F: Fn(f64) -> f64
    {
        for i in 0..self.data.len() {
            self.data[i] = op(self.data[i]);
        }
    }
}

impl TimeSeq for ArrayTimeSeq {
    fn ds_type(&self) -> DsType {
        self.ds_type
    }

    fn step(&self) -> i64 {
        self.step
    }

    fn apply(&self, timestamp: i64) -> f64 {
        let i = (timestamp - self.start) / self.step;
        if timestamp < self.start || timestamp >= self.end() {
            f64::NAN
        } else {
            self.data[i as usize]
        }
    }

    fn clone_box(&self) -> Box<dyn TimeSeq> {
        Box::new(self.clone())
    }
}

impl PartialEq for ArrayTimeSeq {
    fn eq(&self, other: &Self) -> bool {
        self.ds_type == other.ds_type
            && self.step == other.step
            && self.start == other.start
            && self.data == other.data
    }
}

impl Eq for ArrayTimeSeq {}

impl Hash for ArrayTimeSeq {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ds_type.hash(state);
        self.step.hash(state);
        self.start.hash(state);
        for &val in &self.data {
            val.to_bits().hash(state);
        }
    }
}

impl fmt::Display for ArrayTimeSeq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values: Vec<String> = self.data.iter().map(|v| v.to_string()).collect();
        write!(
            f,
            "ArrayTimeSeq({:?}, {}, {}, [{}])",
            self.ds_type,
            self.start,
            self.step,
            values.join(",")
        )
    }
}

pub struct FunctionTimeSeq<F>
where
    F: Fn(i64) -> f64
{
    pub ds_type: DsType,
    pub step: i64,
    pub f: F,
}

impl<F> FunctionTimeSeq<F>
where
    F: Fn(i64) -> f64
{
    pub fn new(ds_type: DsType, step: i64, f: F) -> Self {
        FunctionTimeSeq { ds_type, step, f }
    }
}

impl<F> TimeSeq for FunctionTimeSeq<F>
where
    F: Fn(i64) -> f64 + Clone + 'static
{
    fn ds_type(&self) -> DsType {
        self.ds_type
    }

    fn step(&self) -> i64 {
        self.step
    }

    fn apply(&self, timestamp: i64) -> f64 {
        (self.f)(timestamp / self.step * self.step)
    }

    fn clone_box(&self) -> Box<dyn TimeSeq> {
        Box::new(FunctionTimeSeq::new(self.ds_type, self.step, self.f.clone()))
    }
}

pub struct OffsetTimeSeq {
    seq: Box<dyn TimeSeq>,
    offset: i64,
}

impl OffsetTimeSeq {
    pub fn new(seq: Box<dyn TimeSeq>, offset: i64) -> Self {
        OffsetTimeSeq { seq, offset }
    }
}

impl TimeSeq for OffsetTimeSeq {
    fn ds_type(&self) -> DsType {
        self.seq.ds_type()
    }

    fn step(&self) -> i64 {
        self.seq.step()
    }

    fn apply(&self, timestamp: i64) -> f64 {
        self.seq.apply(timestamp - self.offset)
    }

    fn clone_box(&self) -> Box<dyn TimeSeq> {
        Box::new(OffsetTimeSeq::new(self.seq.clone_box(), self.offset))
    }
}

pub struct UnaryOpTimeSeq {
    ts: Box<dyn TimeSeq>,
    f: Box<dyn Fn(f64) -> f64>,
}

impl UnaryOpTimeSeq {
    pub fn new(ts: Box<dyn TimeSeq>, f: Box<dyn Fn(f64) -> f64>) -> Self {
        UnaryOpTimeSeq { ts, f }
    }
}

impl TimeSeq for UnaryOpTimeSeq {
    fn ds_type(&self) -> DsType {
        self.ts.ds_type()
    }

    fn step(&self) -> i64 {
        self.ts.step()
    }

    fn apply(&self, timestamp: i64) -> f64 {
        (self.f)(self.ts.apply(timestamp))
    }

    fn clone_box(&self) -> Box<dyn TimeSeq> {
        // Note: This requires the function to be clonable, which might need
        // Arc<dyn Fn> instead of Box<dyn Fn> for proper cloning
        panic!("UnaryOpTimeSeq cloning not fully implemented")
    }
}

pub struct BinaryOpTimeSeq {
    ts1: Box<dyn TimeSeq>,
    ts2: Box<dyn TimeSeq>,
    op: Box<dyn Fn(f64, f64) -> f64>,
}

impl BinaryOpTimeSeq {
    pub fn new(
        ts1: Box<dyn TimeSeq>,
        ts2: Box<dyn TimeSeq>,
        op: Box<dyn Fn(f64, f64) -> f64>
    ) -> Self {
        assert_eq!(
            ts1.step(),
            ts2.step(),
            "time series must have the same step size"
        );
        BinaryOpTimeSeq { ts1, ts2, op }
    }
}

impl TimeSeq for BinaryOpTimeSeq {
    fn ds_type(&self) -> DsType {
        self.ts1.ds_type()
    }

    fn step(&self) -> i64 {
        self.ts1.step()
    }

    fn apply(&self, timestamp: i64) -> f64 {
        (self.op)(self.ts1.apply(timestamp), self.ts2.apply(timestamp))
    }

    fn clone_box(&self) -> Box<dyn TimeSeq> {
        // Note: This requires the function to be clonable
        panic!("BinaryOpTimeSeq cloning not fully implemented")
    }
}

// Math utility functions that might be needed
pub mod math {
    pub fn add_nan(a: f64, b: f64) -> f64 {
        if a.is_nan() { b }
        else if b.is_nan() { a }
        else { a + b }
    }

    pub fn max_nan(a: f64, b: f64) -> f64 {
        if a.is_nan() { b }
        else if b.is_nan() { a }
        else { a.max(b) }
    }

    pub fn min_nan(a: f64, b: f64) -> f64 {
        if a.is_nan() { b }
        else if b.is_nan() { a }
        else { a.min(b) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_time_seq() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = ArrayTimeSeq::new(DsType::Gauge, 0, 60, data);

        assert_eq!(ts.apply(0), 1.0);
        assert_eq!(ts.apply(60), 2.0);
        assert_eq!(ts.apply(120), 3.0);
        assert!(ts.apply(-60).is_nan());
        assert!(ts.apply(300).is_nan());
    }

    #[test]
    fn test_foreach() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = ArrayTimeSeq::new(DsType::Gauge, 0, 60, data);

        let mut sum = 0.0;
        ts.foreach(0, 300, |_, v| {
            if !v.is_nan() {
                sum += v;
            }
        });
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_bounded() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = ArrayTimeSeq::new(DsType::Gauge, 0, 60, data);

        let bounded = ts.bounded(60, 180);
        assert_eq!(bounded.data.len(), 2);
        assert_eq!(bounded.data[0], 2.0);
        assert_eq!(bounded.data[1], 3.0);
    }
}