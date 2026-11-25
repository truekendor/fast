//! Scalar and runtime-dispatched SIMD kernels for the crossing filter.

/// Chooses the CPU implementation once per screenshot.
#[derive(Clone, Copy)]
pub(super) struct Filter {
    #[cfg(target_arch = "x86_64")]
    avx: bool,
}

impl Filter {
    /// Uses AVX on compatible x86-64 processors and the portable filter elsewhere.
    pub fn automatic() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx: std::is_x86_feature_detected!("avx"),
        }
    }

    /// Uses the portable filter for testing and diagnostics.
    pub fn scalar() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx: false,
        }
    }

    /// Computes the alternating three-pixel block responses for a row.
    pub fn correlate(self, row: &[f32], differences: &mut [f32], output: &mut [f32]) {
        assert!(differences.len() <= row.len().saturating_sub(7));
        assert!(output.len() <= differences.len().saturating_sub(66));
        #[cfg(target_arch = "x86_64")]
        if self.avx {
            // The constructor checks AVX support and the assertions bound each lane's loads.
            unsafe { avx::correlate(row, differences, output) };
            return;
        }
        differences_scalar(row, differences);
        crossing_sum_scalar(differences, output);
    }
}

/// Computes the difference between the left and right three-pixel blocks.
fn differences_scalar(row: &[f32], differences: &mut [f32]) {
    for (delta, window) in differences.iter_mut().zip(row.windows(8)) {
        *delta = (window[0] + window[1] + window[2]) - (window[5] + window[6] + window[7]);
    }
}

/// Combines seven alternating block differences into a crossing response.
fn crossing_sum_scalar(differences: &[f32], output: &mut [f32]) {
    for (value, deltas) in output.iter_mut().zip(differences[6..].windows(61)) {
        *value =
            deltas[0] - deltas[10] + deltas[20] - deltas[30] + deltas[40] - deltas[50] + deltas[60];
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};

    /// Applies the same filter to eight adjacent pixels at a time, with a scalar tail.
    /// The caller must check AVX support and reserve seven and 66 input pixels for each pass.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn correlate(row: &[f32], differences: &mut [f32], output: &mut [f32]) {
        let (chunks, tail) = differences.as_chunks_mut::<8>();
        for (index, chunk) in chunks.iter_mut().enumerate() {
            // Each chunk has eight lanes and the input reserves seven additional pixels.
            unsafe {
                let input = row.as_ptr().add(index * 8);
                let left = _mm256_add_ps(
                    _mm256_add_ps(_mm256_loadu_ps(input), _mm256_loadu_ps(input.add(1))),
                    _mm256_loadu_ps(input.add(2)),
                );
                let right = _mm256_add_ps(
                    _mm256_add_ps(_mm256_loadu_ps(input.add(5)), _mm256_loadu_ps(input.add(6))),
                    _mm256_loadu_ps(input.add(7)),
                );
                _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_sub_ps(left, right));
            }
        }
        super::differences_scalar(&row[chunks.len() * 8..], tail);
        let (chunks, tail) = output.as_chunks_mut::<8>();
        for (index, chunk) in chunks.iter_mut().enumerate() {
            // Each eight-lane load fits within the differences plus the reserved pixels.
            unsafe {
                let input = differences.as_ptr().add(index * 8 + 6);
                let value = _mm256_sub_ps(_mm256_loadu_ps(input), _mm256_loadu_ps(input.add(10)));
                let value = _mm256_add_ps(value, _mm256_loadu_ps(input.add(20)));
                let value = _mm256_sub_ps(value, _mm256_loadu_ps(input.add(30)));
                let value = _mm256_add_ps(value, _mm256_loadu_ps(input.add(40)));
                let value = _mm256_sub_ps(value, _mm256_loadu_ps(input.add(50)));
                let value = _mm256_add_ps(value, _mm256_loadu_ps(input.add(60)));
                _mm256_storeu_ps(chunk.as_mut_ptr(), value);
            }
        }
        super::crossing_sum_scalar(&differences[chunks.len() * 8..], tail);
    }
}

#[cfg(test)]
mod tests {
    use super::Filter;

    #[test]
    /// Confirms that runtime dispatch preserves every response, including partial vectors.
    fn automatic_and_scalar_match() {
        let row: Vec<_> = (0..121).map(|n| (n * n % 251) as f32 / 13.0).collect();
        let mut differences = vec![0.0; row.len() - 7];
        let mut scalar = vec![0.0; row.len() - 79];
        let mut automatic = scalar.clone();
        Filter::scalar().correlate(&row, &mut differences, &mut scalar);
        Filter::automatic().correlate(&row, &mut differences, &mut automatic);
        assert_eq!(scalar, automatic);
    }
}
