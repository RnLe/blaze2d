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

    /// The result type (Maxwell band structure, or operator-data extraction)
    pub result_type: CompactResultType,
}

/// Type of compact result.
#[derive(Debug, Clone)]
pub enum CompactResultType {
    /// Maxwell result with full band structure
    Maxwell(MaxwellResult),
    /// Operator-data extraction result (matrix elements for multi-band theories)
    OperatorData(OperatorDataResult),
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

/// EA Hamiltonian extraction result: all matrix elements needed for moiré theory.
#[derive(Debug, Clone)]
pub struct OperatorDataResult {
    /// Carrier momentum k₀.
    pub k0: [f64; 2],
    /// Registry point (fractional coordinates).
    pub registry: [f64; 2],
    /// Number of retained bands.
    pub n_retained: usize,
    /// Number of remote bands.
    pub n_remote: usize,
    /// Grid dimensions [nx, ny].
    pub grid_dims: [usize; 2],

    /// Eigenvalues for all n_total bands: (2π/a)² units.
    pub eigenvalues: Vec<f64>,

    /// Velocity matrices v^(i)_{mn}, per direction. Shape: [n_retained × n_total].
    pub velocity_matrices: [Vec<ComplexPair>; 2],

    /// Raw second-derivative matrices w^(ij)_{mn}. Shape: [n_retained × n_retained].
    pub w_matrices: [[Vec<ComplexPair>; 2]; 2],

    /// Löwdin-corrected inverse mass tensor M⁻¹_{ij,mn}. Shape: [n_retained × n_retained].
    pub mass_tensor_inv: [[Vec<ComplexPair>; 2]; 2],

    /// Registry-derivative matrices ⟨uₘ|∂L₀/∂Rⱼ|uₙ⟩. Shape: [n_retained × n_total].
    pub r_derivative_matrices: Option<[Vec<ComplexPair>; 2]>,

    /// Metric-derivative matrices ⟨uₘ|∂B/∂Rⱼ|uₙ⟩. Shape: [n_retained × n_total].
    pub metric_derivative_matrices: Option<[Vec<ComplexPair>; 2]>,

    /// Berry-connection matrices A_j in the retained space. Shape: [n_retained × n_retained].
    pub berry_connection_matrices: Option<[Vec<ComplexPair>; 2]>,

    /// Born–Huang potential Φ_{mn}(R). Shape: [n_retained × n_retained].
    pub born_huang: Option<Vec<ComplexPair>>,

    /// Overlap matrix S_{mn}. Shape: [n_retained × n_retained].
    pub overlap_matrix: Option<Vec<ComplexPair>>,

    /// Number of eigensolver iterations.
    pub n_iterations: usize,
    /// Whether the eigensolver converged.
    pub converged: bool,
    /// Polarization used.
    pub polarization: String,
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
            CompactResultType::OperatorData(h) => {
                let eig_size = h.eigenvalues.len() * 8;
                let vel_size: usize = h.velocity_matrices.iter().map(|v| v.len() * 16).sum();
                let w_size: usize = h.w_matrices.iter().flat_map(|row| row.iter()).map(|v| v.len() * 16).sum();
                let mass_size: usize = h.mass_tensor_inv.iter().flat_map(|row| row.iter()).map(|v| v.len() * 16).sum();
                let r_size: usize = h.r_derivative_matrices.as_ref().map(|m| m.iter().map(|v| v.len() * 16).sum()).unwrap_or(0);
                let metric_size: usize = h.metric_derivative_matrices.as_ref().map(|m| m.iter().map(|v| v.len() * 16).sum()).unwrap_or(0);
                let berry_size: usize = h.berry_connection_matrices.as_ref().map(|m| m.iter().map(|v| v.len() * 16).sum()).unwrap_or(0);
                let bh_size = h.born_huang.as_ref().map(|v| v.len() * 16).unwrap_or(0);
                let ov_size = h.overlap_matrix.as_ref().map(|v| v.len() * 16).unwrap_or(0);
                eig_size + vel_size + w_size + mass_size + r_size + metric_size + berry_size + bh_size + ov_size
            }
        };

        base + params_size + result_size
    }

    /// Number of k-points in this result (Maxwell only).
    pub fn num_k_points(&self) -> usize {
        match &self.result_type {
            CompactResultType::Maxwell(m) => m.k_path.len(),
            CompactResultType::OperatorData(_) => 1,
        }
    }

    /// Number of bands computed.
    pub fn num_bands(&self) -> usize {
        match &self.result_type {
            CompactResultType::Maxwell(m) => m.bands.first().map(|b| b.len()).unwrap_or(0),
            CompactResultType::OperatorData(h) => h.eigenvalues.len(),
        }
    }

    /// Get k_path if this is a Maxwell result.
    pub fn k_path(&self) -> Option<&Vec<[f64; 2]>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.k_path),
            _ => None,
        }
    }

    /// Get distances if this is a Maxwell result.
    pub fn distances(&self) -> Option<&Vec<f64>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.distances),
            _ => None,
        }
    }

    /// Get bands if this is a Maxwell result.
    pub fn bands(&self) -> Option<&Vec<Vec<f64>>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.bands),
            _ => None,
        }
    }

    /// Get eigenvalues if this is an EA-Hamiltonian result.
    pub fn eigenvalues(&self) -> Option<&Vec<f64>> {
        match &self.result_type {
            CompactResultType::Maxwell(_) => None,
            CompactResultType::OperatorData(h) => Some(&h.eigenvalues),
        }
    }

    /// Check if this is a Maxwell result.
    pub fn is_maxwell(&self) -> bool {
        matches!(self.result_type, CompactResultType::Maxwell(_))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expansion::AtomParams;
    use blaze2d_core::polarization::Polarization;

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
        assert!(result.k_path().is_some());
        assert!(result.distances().is_some());
        assert!(result.bands().is_some());
        assert!(result.eigenvalues().is_none());
        assert_eq!(result.num_k_points(), 100);
        assert_eq!(result.num_bands(), 10);
    }
}
