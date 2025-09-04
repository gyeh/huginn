use crate::block::Block;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsolidationFunction {
    Avg,
    Sum,
    Min,
    Max,
}

impl ConsolidationFunction {
    pub fn name(&self) -> &'static str {
        match self {
            ConsolidationFunction::Avg => "avg",
            ConsolidationFunction::Sum => "sum",
            ConsolidationFunction::Min => "min",
            ConsolidationFunction::Max => "max",
        }
    }

    /// - `avg`: average over the entire window length (including NaN slots in the denominator).
    ///          If *all* values are NaN, returns NaN.
    /// - `sum`: sum of non-NaN values; if *all* values are NaN, returns NaN.
    /// - `min`/`max`: min/max over non-NaN values; if *all* values are NaN, returns NaN.
    pub fn compute(
        &self,
        b: &dyn Block,
        pos: usize,
        aggr: i32,
        multiple: usize,
    ) -> f64 {
        let end = pos + multiple;

        match self {
            ConsolidationFunction::Avg => {
                let mut total = 0.0f64;
                let mut count_non_nan = 0usize;
                let mut count_nan = 0usize;

                for i in pos..end {
                    let v = b.get_with_aggr(i, aggr);
                    if v.is_nan() {
                        count_nan += 1;
                    } else {
                        total += v;
                        count_non_nan += 1;
                    }
                }

                if count_non_nan == 0 {
                    f64::NAN
                } else {
                    // Note: denominator includes NaN slots (same as Scala: total / (count + nanCount))
                    let denom = (count_non_nan + count_nan) as f64;
                    total / denom
                }
            }

            ConsolidationFunction::Sum => {
                let mut total = 0.0f64;
                let mut count_non_nan = 0usize;

                for i in pos..end {
                    let v = b.get_with_aggr(i, aggr);
                    if !v.is_nan() {
                        total += v;
                        count_non_nan += 1;
                    }
                }

                if count_non_nan == 0 { f64::NAN } else { total }
            }

            ConsolidationFunction::Min => {
                let mut m = f64::MAX;
                let mut count_non_nan = 0usize;

                for i in pos..end {
                    let v = b.get_with_aggr(i, aggr);
                    if !v.is_nan() {
                        if v < m { m = v; }
                        count_non_nan += 1;
                    }
                }

                if count_non_nan == 0 { f64::NAN } else { m }
            }

            ConsolidationFunction::Max => {
                let mut m = -f64::MAX;
                let mut count_non_nan = 0usize;

                for i in pos..end {
                    let v = b.get_with_aggr(i, aggr);
                    if !v.is_nan() {
                        if v > m { m = v; }
                        count_non_nan += 1;
                    }
                }

                if count_non_nan == 0 { f64::NAN } else { m }
            }
        }
    }
}

impl std::fmt::Display for ConsolidationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ":cf-{}", self.name())
    }
}
