//! Compact result types for efficient transfer and storage.
//!
//! These types represent the output of band structure and eigenvalue calculations
//! in a serializable, platform-agnostic format suitable for streaming to consumers.

use crate::expansion::JobParams;

/// A complex number represented as a pair of f64 (real, imaginary).
/// Used for serializable representation of eigenvectors.
pub type ComplexPair = [f64; 2];

// ============================================================================
// Compact Band Result
// ============================================================================

/// Serializable band structure result optimized for efficient transfer.
///
/// This is a self-contained representation of a single band structure calculation,
/// including all metadata needed for output and analysis.
///
/// ## Memory Layout
///
/// For a typical calculation with 10 bands and 100 k-points:
/// - `k_path`: 100 × 2 × 8 = 1,600 bytes
/// - `distances`: 100 × 8 = 800 bytes
/// - `bands`: 100 × 10 × 8 = 8,000 bytes
/// - Metadata: ~200 bytes
/// - **Total**: ~10.6 KB per result
#[derive(Debug, Clone)]
pub struct CompactBandResult {
    /// Job index (matches ExpandedJob.index)
    pub job_index: usize,

    /// Parameter values used for this job
    pub params: JobParams,

    /// The result type (Maxwell with k-path data, or EA with eigenvalues only)
    pub result_type: CompactResultType,
}

/// Type of compact result.
#[derive(Debug, Clone)]
pub enum CompactResultType {
    /// Maxwell result with full band structure
    Maxwell(MaxwellResult),
    /// EA result with eigenvalues only (no k-path)
    EA(EAResult),
}

/// Maxwell band structure result.
#[derive(Debug, Clone)]
pub struct MaxwellResult {
    /// K-path in fractional coordinates
    pub k_path: Vec<[f64; 2]>,

    /// Cumulative distance along k-path
    pub distances: Vec<f64>,

    /// Computed eigenfrequencies organized as bands[k_index][band_index]
    /// Values are normalized frequencies (ω/2π)
    pub bands: Vec<Vec<f64>>,
}

/// EA eigenvalue result.
#[derive(Debug, Clone)]
pub struct EAResult {
    /// Computed eigenvalues
    pub eigenvalues: Vec<f64>,

    /// Computed eigenvectors as flattened complex arrays.
    /// Each inner Vec represents one eigenvector, with complex values as [re, im] pairs.
    /// Shape: [n_bands][grid_size], where each element is [f64; 2] (real, imag).
    pub eigenvectors: Vec<Vec<ComplexPair>>,

    /// Grid dimensions [nx, ny] for reconstructing the 2D field structure
    pub grid_dims: [usize; 2],

    /// Number of iterations taken
    pub n_iterations: usize,

    /// Whether convergence was achieved
    pub converged: bool,
}

impl CompactBandResult {
    /// Approximate size in bytes for buffer management.
    pub fn approx_size(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let params_size = 200; // rough estimate

        let result_size = match &self.result_type {
            CompactResultType::Maxwell(m) => {
                let k_path_size = m.k_path.len() * 16;
                let distances_size = m.distances.len() * 8;
                let bands_size: usize = m.bands.iter().map(|b| b.len() * 8 + 24).sum();
                k_path_size + distances_size + bands_size
            }
            CompactResultType::EA(ea) => {
                let eigenvalues_size = ea.eigenvalues.len() * 8;
                // Each eigenvector: n_elements * 2 floats (complex) * 8 bytes
                let eigenvectors_size: usize = ea.eigenvectors.iter()
                    .map(|v| v.len() * 16 + 24)
                    .sum();
                eigenvalues_size + eigenvectors_size + 24
            }
        };

        base + params_size + result_size
    }

    /// Number of k-points in this result (Maxwell only).
    pub fn num_k_points(&self) -> usize {
        match &self.result_type {
            CompactResultType::Maxwell(m) => m.k_path.len(),
            CompactResultType::EA(_) => 1, // EA has no k-path concept
        }
    }

    /// Number of bands computed.
    pub fn num_bands(&self) -> usize {
        match &self.result_type {
            CompactResultType::Maxwell(m) => m.bands.first().map(|b| b.len()).unwrap_or(0),
            CompactResultType::EA(ea) => ea.eigenvalues.len(),
        }
    }

    /// Get k_path if this is a Maxwell result.
    pub fn k_path(&self) -> Option<&Vec<[f64; 2]>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.k_path),
            CompactResultType::EA(_) => None,
        }
    }

    /// Get distances if this is a Maxwell result.
    pub fn distances(&self) -> Option<&Vec<f64>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.distances),
            CompactResultType::EA(_) => None,
        }
    }

    /// Get bands if this is a Maxwell result.
    pub fn bands(&self) -> Option<&Vec<Vec<f64>>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.bands),
            CompactResultType::EA(_) => None,
        }
    }

    /// Get eigenvalues if this is an EA result.
    pub fn eigenvalues(&self) -> Option<&Vec<f64>> {
        match &self.result_type {
            CompactResultType::Maxwell(_) => None,
            CompactResultType::EA(ea) => Some(&ea.eigenvalues),
        }
    }

    /// Check if this is a Maxwell result.
    pub fn is_maxwell(&self) -> bool {
        matches!(self.result_type, CompactResultType::Maxwell(_))
    }

    /// Check if this is an EA result.
    pub fn is_ea(&self) -> bool {
        matches!(self.result_type, CompactResultType::EA(_))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expansion::AtomParams;
    use mpb2d_core::polarization::Polarization;

    fn make_test_maxwell_result(index: usize) -> CompactBandResult {
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
                k_path: (0..100).map(|i| [i as f64 / 100.0, 0.0]).collect(),
                distances: (0..100).map(|i| i as f64 / 100.0).collect(),
                bands: (0..100)
                    .map(|_| (0..10).map(|b| 0.1 * b as f64).collect())
                    .collect(),
            }),
        }
    }

    fn make_test_ea_result(index: usize) -> CompactBandResult {
        CompactBandResult {
            job_index: index,
            params: JobParams {
                eps_bg: 0.0,
                resolution: 64,
                polarization: Polarization::TM,
                lattice_type: None,
                atoms: vec![],
                sweep_values: vec![],
            },
            result_type: CompactResultType::EA(EAResult {
                eigenvalues: vec![0.1, 0.2, 0.3, 0.4, 0.5],
                eigenvectors: (0..5)
                    .map(|_| (0..64*64).map(|i| [i as f64 * 0.01, i as f64 * 0.001]).collect())
                    .collect(),
                grid_dims: [64, 64],
                n_iterations: 50,
                converged: true,
            }),
        }
    }

    #[test]
    fn test_compact_result_size() {
        let result = make_test_maxwell_result(0);
        let size = result.approx_size();

        // Should be approximately 10-11 KB for 10 bands × 100 k-points
        assert!(size > 8000, "Size {} too small", size);
        assert!(size < 15000, "Size {} too large", size);
    }

    #[test]
    fn test_maxwell_accessors() {
        let result = make_test_maxwell_result(0);
        assert!(result.is_maxwell());
        assert!(!result.is_ea());
        assert!(result.k_path().is_some());
        assert!(result.distances().is_some());
        assert!(result.bands().is_some());
        assert!(result.eigenvalues().is_none());
        assert_eq!(result.num_k_points(), 100);
        assert_eq!(result.num_bands(), 10);
    }

    #[test]
    fn test_ea_accessors() {
        let result = make_test_ea_result(0);
        assert!(!result.is_maxwell());
        assert!(result.is_ea());
        assert!(result.k_path().is_none());
        assert!(result.distances().is_none());
        assert!(result.bands().is_none());
        assert!(result.eigenvalues().is_some());
        assert_eq!(result.num_k_points(), 1);
        assert_eq!(result.num_bands(), 5);
    }
}
