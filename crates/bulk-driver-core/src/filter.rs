//! Selective filtering for k-points and bands.
//!
//! This module provides the `SelectiveFilter` type for filtering band structure
//! results to include only specific k-points and bands. This is used for:
//!
//! - Selective output mode (merged CSV with specific data)
//! - Stream filtering (reduce bandwidth by filtering at emission point)

use crate::config::SelectiveSpec;
use crate::result::{CompactBandResult, CompactResultType, MaxwellResult};

// ============================================================================
// Selective Filter
// ============================================================================

/// High-performance filter for selective streaming.
///
/// Pre-computes index sets for O(1) lookup during filtering.
/// This is applied at the emission point to minimize data transfer.
///
/// # Example
///
/// ```ignore
/// // Filter to only Gamma, X, M points and first 4 bands
/// let filter = SelectiveFilter::new(
///     vec![0, 10, 15],  // k-indices for Γ, X, M
///     vec![0, 1, 2, 3], // bands 1-4 (0-based)
/// );
///
/// let filtered = filter.apply(&result);
/// assert_eq!(filtered.num_k_points(), 3);
/// assert_eq!(filtered.num_bands(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct SelectiveFilter {
    /// K-point indices to include (sorted, deduplicated)
    k_indices: Vec<usize>,

    /// Band indices to include (0-based internally, sorted, deduplicated)
    band_indices: Vec<usize>,

    /// Whether filtering is actually needed
    is_active: bool,
}

impl SelectiveFilter {
    /// Create a filter from a SelectiveSpec.
    ///
    /// Note: `k_labels` are not resolved here - they should be converted
    /// to indices by the caller using the k-path information.
    pub fn from_spec(spec: &SelectiveSpec) -> Self {
        let mut k_indices: Vec<usize> = spec.k_indices.clone();
        k_indices.sort_unstable();
        k_indices.dedup();

        // Convert 1-based band indices to 0-based
        let mut band_indices: Vec<usize> = spec
            .bands
            .iter()
            .filter(|&&b| b > 0)
            .map(|&b| b - 1)
            .collect();
        band_indices.sort_unstable();
        band_indices.dedup();

        let is_active = !k_indices.is_empty() || !band_indices.is_empty();

        Self {
            k_indices,
            band_indices,
            is_active,
        }
    }

    /// Create a filter with explicit k-point and band indices.
    ///
    /// Band indices are 0-based here (unlike SelectiveSpec which is 1-based).
    pub fn new(k_indices: Vec<usize>, band_indices: Vec<usize>) -> Self {
        let mut k_indices = k_indices;
        k_indices.sort_unstable();
        k_indices.dedup();

        let mut band_indices = band_indices;
        band_indices.sort_unstable();
        band_indices.dedup();

        let is_active = !k_indices.is_empty() || !band_indices.is_empty();

        Self {
            k_indices,
            band_indices,
            is_active,
        }
    }

    /// Create a pass-through filter (no filtering).
    pub fn none() -> Self {
        Self {
            k_indices: Vec::new(),
            band_indices: Vec::new(),
            is_active: false,
        }
    }

    /// Check if this filter is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Get the k-point indices (empty means all).
    pub fn k_indices(&self) -> &[usize] {
        &self.k_indices
    }

    /// Get the band indices (empty means all).
    pub fn band_indices(&self) -> &[usize] {
        &self.band_indices
    }

    /// Apply the filter to a result, returning a filtered copy.
    ///
    /// If the filter is not active, returns a clone of the original.
    /// This is optimized for the common case where filtering is applied.
    ///
    /// Note: Filtering only applies to Maxwell results. EA results are returned unchanged.
    pub fn apply(&self, result: &CompactBandResult) -> CompactBandResult {
        if !self.is_active {
            return result.clone();
        }

        // Filtering only applies to Maxwell results
        let maxwell = match &result.result_type {
            CompactResultType::Maxwell(m) => m,
            CompactResultType::EA(_) => return result.clone(),
        };

        // Determine which k-indices to include
        let k_filter: Vec<usize> = if self.k_indices.is_empty() {
            // No k-filter: include all
            (0..maxwell.k_path.len()).collect()
        } else {
            // Filter to valid indices
            self.k_indices
                .iter()
                .copied()
                .filter(|&i| i < maxwell.k_path.len())
                .collect()
        };

        // Determine which band indices to include
        let num_bands = maxwell.bands.first().map(|b| b.len()).unwrap_or(0);
        let band_filter: Vec<usize> = if self.band_indices.is_empty() {
            // No band filter: include all
            (0..num_bands).collect()
        } else {
            // Filter to valid indices
            self.band_indices
                .iter()
                .copied()
                .filter(|&i| i < num_bands)
                .collect()
        };

        // Build filtered result
        let k_path: Vec<[f64; 2]> = k_filter.iter().map(|&i| maxwell.k_path[i]).collect();
        let distances: Vec<f64> = k_filter.iter().map(|&i| maxwell.distances[i]).collect();
        let bands: Vec<Vec<f64>> = k_filter
            .iter()
            .map(|&k_idx| {
                band_filter
                    .iter()
                    .map(|&b_idx| maxwell.bands[k_idx][b_idx])
                    .collect()
            })
            .collect();

        CompactBandResult {
            job_index: result.job_index,
            params: result.params.clone(),
            result_type: CompactResultType::Maxwell(MaxwellResult {
                k_path,
                distances,
                bands,
            }),
        }
    }

    /// Get the number of k-points that will be in filtered output.
    pub fn k_count(&self) -> Option<usize> {
        if self.k_indices.is_empty() {
            None // All k-points
        } else {
            Some(self.k_indices.len())
        }
    }

    /// Get the number of bands that will be in filtered output.
    pub fn band_count(&self) -> Option<usize> {
        if self.band_indices.is_empty() {
            None // All bands
        } else {
            Some(self.band_indices.len())
        }
    }
}

impl Default for SelectiveFilter {
    fn default() -> Self {
        Self::none()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expansion::{AtomParams, JobParams};
    use mpb2d_core::polarization::Polarization;

    fn make_test_result(index: usize) -> CompactBandResult {
        CompactBandResult {
            job_index: index,
            params: JobParams {
                eps_bg: 12.0,
                resolution: 32,
                polarization: Polarization::TM,
                lattice_type: Some("square".to_string()),
                atoms: vec![AtomParams {
                    index: 0,
                    pos: [0.5, 0.5],
                    radius: 0.3,
                    eps_inside: 1.0,
                }],
                sweep_values: vec![],
            },
            result_type: CompactResultType::Maxwell(MaxwellResult {
                k_path: vec![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
                distances: vec![0.0, 0.5, 1.0],
                bands: vec![
                    vec![0.1, 0.2, 0.3],
                    vec![0.15, 0.25, 0.35],
                    vec![0.2, 0.3, 0.4],
                ],
            }),
        }
    }

    #[test]
    fn test_selective_filter_none() {
        let filter = SelectiveFilter::none();
        assert!(!filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        // Should be identical
        assert_eq!(
            filtered.k_path().unwrap().len(),
            result.k_path().unwrap().len()
        );
        assert_eq!(
            filtered.bands().unwrap().len(),
            result.bands().unwrap().len()
        );
    }

    #[test]
    fn test_selective_filter_k_only() {
        // Filter to only k-index 0 and 2
        let filter = SelectiveFilter::new(vec![0, 2], vec![]);
        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        assert_eq!(filtered.k_path().unwrap().len(), 2);
        assert_eq!(filtered.k_path().unwrap()[0], result.k_path().unwrap()[0]);
        assert_eq!(filtered.k_path().unwrap()[1], result.k_path().unwrap()[2]);
        assert_eq!(filtered.bands().unwrap().len(), 2);
        // All bands preserved
        assert_eq!(
            filtered.bands().unwrap()[0].len(),
            result.bands().unwrap()[0].len()
        );
    }

    #[test]
    fn test_selective_filter_bands_only() {
        // Filter to only bands 0 and 2 (0-based)
        let filter = SelectiveFilter::new(vec![], vec![0, 2]);
        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        // All k-points preserved
        assert_eq!(
            filtered.k_path().unwrap().len(),
            result.k_path().unwrap().len()
        );
        // Only 2 bands
        assert_eq!(filtered.bands().unwrap()[0].len(), 2);
        assert_eq!(
            filtered.bands().unwrap()[0][0],
            result.bands().unwrap()[0][0]
        );
        assert_eq!(
            filtered.bands().unwrap()[0][1],
            result.bands().unwrap()[0][2]
        );
    }

    #[test]
    fn test_selective_filter_both() {
        // Filter to k-index 1 and bands 1,2 (0-based)
        let filter = SelectiveFilter::new(vec![1], vec![1, 2]);
        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        assert_eq!(filtered.k_path().unwrap().len(), 1);
        assert_eq!(filtered.k_path().unwrap()[0], result.k_path().unwrap()[1]);
        assert_eq!(filtered.bands().unwrap().len(), 1);
        assert_eq!(filtered.bands().unwrap()[0].len(), 2);
        assert_eq!(
            filtered.bands().unwrap()[0][0],
            result.bands().unwrap()[1][1]
        );
        assert_eq!(
            filtered.bands().unwrap()[0][1],
            result.bands().unwrap()[1][2]
        );
    }

    #[test]
    fn test_selective_filter_from_spec() {
        let spec = SelectiveSpec {
            k_indices: vec![0, 2],
            k_labels: vec![], // Not resolved here
            bands: vec![1, 3], // 1-based
        };
        let filter = SelectiveFilter::from_spec(&spec);

        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        assert_eq!(filtered.k_path().unwrap().len(), 2);
        assert_eq!(filtered.bands().unwrap()[0].len(), 2);
        // Band 1 -> index 0, Band 3 -> index 2
        assert_eq!(
            filtered.bands().unwrap()[0][0],
            result.bands().unwrap()[0][0]
        );
        assert_eq!(
            filtered.bands().unwrap()[0][1],
            result.bands().unwrap()[0][2]
        );
    }

    #[test]
    fn test_selective_filter_out_of_bounds() {
        // Request indices that don't exist
        let filter = SelectiveFilter::new(vec![0, 100], vec![0, 100]);

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        // Should only include valid indices
        assert_eq!(filtered.k_path().unwrap().len(), 1); // Only index 0 is valid
        assert_eq!(filtered.bands().unwrap()[0].len(), 1); // Only band 0 is valid
    }
}
