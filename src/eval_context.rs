use chrono::{DateTime, Duration, Timelike, Utc};

/// Evaluation context for time-based computations
#[derive(Clone, Debug)]
pub struct EvalContext {
    pub start: i64,
    pub end: i64,
    pub step: i64,
}

impl EvalContext {
    /// Creates a new EvalContext with the given parameters
    pub fn new(start: i64, end: i64, step: i64) -> Result<Self, String> {
        if start >= end {
            return Err(format!(
                "start time must be less than end time ({} >= {})",
                start, end
            ));
        }

        Ok(EvalContext {
            start,
            end,
            step,
        })
    }

    /// Creates a new EvalContext with state
    pub fn with_state(
        start: i64,
        end: i64,
        step: i64,
    ) -> Result<Self, String> {
        if start >= end {
            return Err(format!(
                "start time must be less than end time ({} >= {})",
                start, end
            ));
        }

        Ok(EvalContext {
            start,
            end,
            step,
        })
    }

    /// Buffer size that would be needed to represent the result set based on the start time,
    /// end time, and step size.
    pub fn buffer_size(&self) -> usize {
        ((self.end - self.start) / self.step) as usize + 1
    }

    /// Partitions the context by the given duration and truncation unit
    pub fn partition(&self, one_step: Duration, truncate_to_hour: bool) -> Vec<EvalContext> {
        let mut result = Vec::new();

        // Convert start time to DateTime and truncate
        let start_dt = DateTime::<Utc>::from_timestamp_millis(self.start)
            .expect("Invalid start timestamp");

        let mut t = if truncate_to_hour {
            // Truncate to hour
            let truncated = start_dt
                .with_minute(0).unwrap()
                .with_second(0).unwrap()
                .with_nanosecond(0).unwrap();
            truncated.timestamp_millis()
        } else {
            // Truncate to day
            let truncated = start_dt
                .with_hour(0).unwrap()
                .with_minute(0).unwrap()
                .with_second(0).unwrap()
                .with_nanosecond(0).unwrap();
            truncated.timestamp_millis()
        };

        let step_millis = one_step.num_milliseconds();

        while t < self.end {
            let e = t + step_millis;
            let stime = self.start.max(t);
            let etime = self.end.min(e);

            if let Ok(ctx) = EvalContext::new(stime, etime, self.step) {
                result.push(ctx);
            }
            t = e;
        }

        result
    }

    /// Creates a new context with the given offset
    pub fn with_offset(&self, offset: i64) -> EvalContext {
        let dur = (offset / self.step) * self.step;

        if dur < self.step {
            self.clone()
        } else {
            EvalContext::with_state(
                self.start - dur,
                self.end - dur,
                self.step,
            ).unwrap_or_else(|_| self.clone())
        }
    }
}