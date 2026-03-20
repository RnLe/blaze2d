//! Band tracking across k-points using polar decomposition + Hungarian algorithm.
//!
//! When computing band structures, eigenvalues at consecutive k-points may
//! swap order due to band crossings or near-degeneracies. This module provides
//! tools to track bands by computing the optimal rotation between eigenspaces,
//! then using the Hungarian algorithm to extract the globally optimal permutation.
//!
//! # Algorithm
//!
//! At each k-point transition from k to k+1:
//!
//! 1. Compute the complex B-weighted overlap matrix:
//!    ```text
//!    O = X_prev† B X_curr    (m × m matrix)
//!    ```
//!
//! 2. Extract the unitary rotation via polar decomposition:
//!    ```text
//!    O = W Σ V†  →  U = W V†
//!    ```
//!
//! 3. Use Hungarian algorithm on |U| to find the globally optimal permutation
//!    that maximizes total overlap: π* = argmax_π Σ_i |U[i, π(i)]|
//!
//! 4. Reorder eigenvalues and eigenvectors at k+1 according to the permutation.
//!
//! # Why Polar Decomposition + Hungarian?
//!
//! - **Polar decomposition** solves the Procrustes problem and preserves phase,
//!   giving a unitary U that captures the subspace rotation optimally.
//! - **Hungarian algorithm** finds the globally optimal assignment from U,
//!   avoiding the pitfalls of greedy column-wise selection which can fail
//!   when multiple entries have similar magnitudes.
//!
//! The smallest singular value σ_min indicates subspace alignment quality:
//! - σ_min ≈ 1: bands are well-separated and tracking is reliable
//! - σ_min < 0.1: near-degeneracy or crossing, tracking may be ambiguous

use crate::eigensolver::subspace_prediction::{
    compute_complex_overlap_matrix, polar_decomposition, polar_decomposition_with_singular_values,
};
use crate::field::Field2D;
use log::debug;

// ============================================================================
// Band Tracking Result
// ============================================================================

/// Result of band tracking between consecutive k-points.
#[derive(Debug, Clone)]
pub struct BandTrackingResult {
    /// The permutation: permutation[i] is the index in curr that maps to band i.
    /// To reorder: new_omegas[i] = old_omegas[permutation[i]]
    pub permutation: Vec<usize>,
    /// Whether any bands were swapped (permutation differs from identity).
    pub had_swaps: bool,
    /// Smallest singular value from polar decomposition (quality indicator).
    /// Values close to 1.0 indicate reliable tracking; values < 0.1 indicate
    /// near-degeneracy or band crossing where tracking may be ambiguous.
    pub sigma_min: f64,
    /// Degenerate block info: list of (start_idx, block_size) for detected blocks.
    /// Empty if no degeneracies were detected.
    pub degenerate_blocks: Vec<(usize, usize)>,
}

// ============================================================================
// Hungarian Algorithm for Optimal Assignment
// ============================================================================

/// Find the optimal assignment that maximizes total weight using the Hungarian algorithm.
///
/// Given an n×n weight matrix W, finds a permutation π such that Σ_i W[i, π(i)] is maximized.
/// This is the assignment problem, solved optimally in O(n³) time.
///
/// # Algorithm
///
/// Uses the Kuhn-Munkres (Hungarian) algorithm:
/// 1. Convert maximization to minimization by negating weights
/// 2. Reduce rows and columns
/// 3. Find augmenting paths to build optimal matching
///
/// For small matrices (typical band counts ≤ 20), this is very fast.
///
/// # Arguments
/// * `weights` - n×n matrix where weights[i][j] is the weight for assigning row i to column j
///
/// # Returns
/// A permutation vector where permutation[i] = j means row i is assigned to column j
fn hungarian_assignment(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }

    // Handle trivial cases
    if n == 1 {
        return vec![0];
    }

    // Convert to minimization problem: cost = max_weight - weight
    let max_weight = weights
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let cost: Vec<Vec<f64>> = weights
        .iter()
        .map(|row| row.iter().map(|&w| max_weight - w).collect())
        .collect();

    // Hungarian algorithm state
    let mut u = vec![0.0; n + 1]; // Row potentials
    let mut v = vec![0.0; n + 1]; // Column potentials
    let mut p = vec![0usize; n + 1]; // p[j] = row assigned to column j (1-indexed internally)
    let mut way = vec![0usize; n + 1]; // way[j] = previous column in augmenting path

    for i in 1..=n {
        // Start augmenting path from row i
        p[0] = i;
        let mut j0 = 0usize; // Current column (0 = virtual column)
        let mut minv = vec![f64::INFINITY; n + 1]; // Minimum reduced cost to reach column
        let mut used = vec![false; n + 1]; // Columns visited in this iteration

        // Find augmenting path
        loop {
            used[j0] = true;
            let i0 = p[j0]; // Row assigned to current column
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize; // Next column to visit

            for j in 1..=n {
                if !used[j] {
                    // Reduced cost for assigning row i0 to column j
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Update potentials
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;

            if p[j0] == 0 {
                break; // Found unassigned column, path complete
            }
        }

        // Trace back and update assignment
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Extract permutation: build row->column mapping from column assignments
    let mut permutation = vec![0usize; n];
    for j in 1..=n {
        let row = p[j] - 1; // row assigned to column j (convert to 0-indexed)
        permutation[row] = j - 1; // row -> column
    }

    permutation
}

/// Extract the optimal permutation from the rotation matrix U using Hungarian algorithm.
///
/// Computes |U[i,j]| for all entries and finds the assignment that maximizes
/// the total magnitude, ensuring globally optimal band matching.
///
/// # Arguments
/// * `rotation` - The m×m unitary rotation matrix U from polar decomposition
///
/// # Returns
/// A permutation vector where permutation[i] = j means curr band at index j
/// should be reordered into position i (matching prev band i).
fn extract_permutation_from_rotation(rotation: &faer::Mat<faer::c64>) -> Vec<usize> {
    let m = rotation.ncols();
    if m == 0 {
        return Vec::new();
    }

    // Build weight matrix: |U[i,j]|
    let weights: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    let c = rotation.get(i, j);
                    (c.re * c.re + c.im * c.im).sqrt() // |U[i,j]|
                })
                .collect()
        })
        .collect();

    // Use Hungarian algorithm for globally optimal assignment
    hungarian_assignment(&weights)
}

// ============================================================================
// Main Band Tracking Function
// ============================================================================

/// Track bands between consecutive k-points using polar decomposition.
///
/// Computes the optimal rotation between eigenspaces and extracts the
/// permutation needed to maintain consistent band labeling.
///
/// # Arguments
/// * `prev_vecs` - Eigenvectors from previous k-point (reference)
/// * `curr_vecs` - Eigenvectors from current k-point (to be reordered)
/// * `eps` - Dielectric function for B-weighting (Some for TM, None for TE)
///
/// # Returns
/// A `BandTrackingResult` containing the permutation to apply to the current
/// k-point's eigenvalues and eigenvectors.
///
/// # Algorithm
///
/// 1. Compute complex overlap: O = X_prev† B X_curr
/// 2. Polar decomposition: O = W Σ V† → U = W V†
/// 3. Extract permutation from U by finding dominant entries per column
pub fn track_bands(
    prev_vecs: &[Field2D],
    curr_vecs: &[Field2D],
    eps: Option<&[f64]>,
) -> BandTrackingResult {
    let n = curr_vecs.len();

    if n == 0 || prev_vecs.is_empty() {
        return BandTrackingResult {
            permutation: (0..n).collect(),
            had_swaps: false,
            sigma_min: 1.0,
            degenerate_blocks: Vec::new(),
        };
    }

    // Step 1: Compute complex overlap matrix using the shared implementation
    let overlap = compute_complex_overlap_matrix(prev_vecs, curr_vecs, eps);

    // Step 2: Polar decomposition to get unitary rotation U and quality metric σ_min
    let (rotation, sigma_min) = polar_decomposition(&overlap);

    // Step 3: Extract permutation from rotation matrix
    let permutation = extract_permutation_from_rotation(&rotation);

    // Check if any swaps occurred
    let had_swaps = permutation.iter().enumerate().any(|(i, &p)| i != p);

    BandTrackingResult {
        permutation,
        had_swaps,
        sigma_min,
        degenerate_blocks: Vec::new(), // Basic version doesn't detect blocks
    }
}

/// Track bands with frequency-based disambiguation for degenerate blocks.
///
/// This enhanced version uses the singular value spectrum to detect degenerate
/// blocks (bands that are nearly indistinguishable by eigenvector overlap).
/// Within degenerate blocks, it uses frequency continuity as a tiebreaker:
/// the permutation is adjusted to minimize |ω_curr[perm[i]] - ω_prev[i]|.
///
/// # Arguments
/// * `prev_vecs` - Eigenvectors from previous k-point (reference)
/// * `curr_vecs` - Eigenvectors from current k-point (to be reordered)
/// * `prev_omegas` - Frequencies from previous k-point
/// * `curr_omegas` - Frequencies from current k-point (before reordering)
/// * `eps` - Dielectric function for B-weighting (Some for TM, None for TE)
///
/// # Returns
/// A `BandTrackingResult` with the permutation potentially refined for degenerate blocks.
pub fn track_bands_with_frequencies(
    prev_vecs: &[Field2D],
    curr_vecs: &[Field2D],
    prev_omegas: &[f64],
    curr_omegas: &[f64],
    eps: Option<&[f64]>,
) -> BandTrackingResult {
    let n = curr_vecs.len();

    if n == 0 || prev_vecs.is_empty() {
        return BandTrackingResult {
            permutation: (0..n).collect(),
            had_swaps: false,
            sigma_min: 1.0,
            degenerate_blocks: Vec::new(),
        };
    }

    // Step 1: Compute complex overlap matrix
    let overlap = compute_complex_overlap_matrix(prev_vecs, curr_vecs, eps);

    // Step 2: Polar decomposition with full singular value spectrum
    let (rotation, _sigma_all, sigma_min) = polar_decomposition_with_singular_values(&overlap);

    // Step 3: Detect degenerate bands from the rotation matrix |U|
    let rotation_blocks = detect_degenerate_bands_from_rotation(&rotation);
    let degenerate_blocks = if rotation_blocks.is_empty() {
        Vec::new()
    } else {
        let refined = split_blocks_by_frequency(&rotation_blocks, prev_omegas);
        if refined.is_empty() {
            debug!(
                "[band_tracking] Rotation detected blocks {:?} but frequency gaps pruned them",
                rotation_blocks
            );
            Vec::new()
        } else {
            refined
        }
    };

    // Step 4: Build weight matrix and extract initial permutation
    let weights = build_weight_matrix(&rotation);

    // If we have degenerate blocks, we need a combined approach:
    // - Use overlap weights for well-separated bands
    // - Use frequency continuity within degenerate blocks
    let mut permutation = if degenerate_blocks.is_empty() {
        // Simple case: just use Hungarian on |U|
        hungarian_assignment(&weights)
    } else {
        // Complex case: hybrid approach
        debug!(
            "[band_tracking] Detected {} degenerate block(s): {:?}",
            degenerate_blocks.len(),
            degenerate_blocks
        );
        compute_hybrid_permutation(&weights, &degenerate_blocks, prev_omegas, curr_omegas)
    };

    // Always look for large frequency mismatches after the initial assignment.
    // This catches cases where Hungarian swaps bands even though the previous
    // frequencies imply a different continuity pairing.
    let mismatch_blocks = detect_frequency_mismatch_blocks(&permutation, prev_omegas, curr_omegas);
    if !mismatch_blocks.is_empty() {
        debug!(
            "[band_tracking] σ_min={:.3e}: refining mismatch blocks {:?}",
            sigma_min, mismatch_blocks
        );
        for (block_start, block_size) in mismatch_blocks {
            refine_block_by_frequency(
                &mut permutation,
                block_start,
                block_size,
                prev_omegas,
                curr_omegas,
            );
        }
    }

    // Check if any swaps occurred
    let had_swaps = permutation.iter().enumerate().any(|(i, &p)| i != p);

    BandTrackingResult {
        permutation,
        had_swaps,
        sigma_min,
        degenerate_blocks,
    }
}

/// Build weight matrix |U[i,j]| from the rotation matrix.
fn build_weight_matrix(rotation: &faer::Mat<faer::c64>) -> Vec<Vec<f64>> {
    let m = rotation.ncols();
    (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    let c = rotation.get(i, j);
                    (c.re * c.re + c.im * c.im).sqrt()
                })
                .collect()
        })
        .collect()
}

/// Detect degenerate bands by analyzing the rotation matrix |U|.
///
/// A band is considered degenerate if:
/// 1. Its maximum overlap |U[i,j_max]| is below the threshold, OR
/// 2. There are multiple columns with similar overlap values
///
/// Returns blocks of contiguous band indices that form degenerate subspaces.
fn detect_degenerate_bands_from_rotation(rotation: &faer::Mat<faer::c64>) -> Vec<(usize, usize)> {
    let m = rotation.ncols();
    if m < 2 {
        return Vec::new();
    }

    // For each row, check if the assignment is ambiguous
    let mut is_ambiguous = vec![false; m];

    for i in 0..m {
        // Get all |U[i,j]| values
        let mut row_mags: Vec<f64> = (0..m)
            .map(|j| {
                let c = rotation.get(i, j);
                (c.re * c.re + c.im * c.im).sqrt()
            })
            .collect();

        // Sort descending to find top two
        row_mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let max_val = row_mags[0];
        let second_val = row_mags[1];

        // Ambiguous if:
        // 1. Max value is too low (< 0.7 means poor match)
        // 2. Second value is close to max (ratio > 0.5)
        let ratio = if max_val > 1e-10 {
            second_val / max_val
        } else {
            0.0
        };
        is_ambiguous[i] = max_val < 0.7 || ratio > 0.5;
    }

    // Group contiguous ambiguous bands into blocks
    let mut blocks = Vec::new();
    let mut block_start = None;

    for i in 0..m {
        if is_ambiguous[i] {
            if block_start.is_none() {
                block_start = Some(i);
            }
        } else {
            if let Some(start) = block_start {
                let size = i - start;
                if size > 1 {
                    blocks.push((start, size));
                }
                block_start = None;
            }
        }
    }

    // Handle trailing block
    if let Some(start) = block_start {
        let size = m - start;
        if size > 1 {
            blocks.push((start, size));
        }
    }

    blocks
}

/// Split rotation-detected blocks using frequency gaps so that only truly
/// degenerate neighbors (very small |Δω|) stay coupled. Prevents huge blocks
/// from triggering frequency-based reordering for bands that are well separated.
fn split_blocks_by_frequency(
    blocks: &[(usize, usize)],
    prev_omegas: &[f64],
) -> Vec<(usize, usize)> {
    const ABS_TOL: f64 = 5e-3;
    const REL_TOL: f64 = 0.015; // 1.5% relative gap tolerance

    fn nearly_equal(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
        let gap = (a - b).abs();
        if gap <= abs_tol {
            return true;
        }
        let scale = 0.5 * (a.abs() + b.abs()).max(1e-6);
        gap / scale <= rel_tol
    }

    let mut refined = Vec::new();

    for &(start, size) in blocks {
        if size < 2 {
            continue;
        }
        if start >= prev_omegas.len() {
            continue;
        }
        let block_end = prev_omegas.len().min(start.saturating_add(size));
        if block_end - start < 2 {
            continue;
        }

        let mut curr_start = start;
        let mut curr_len = 1;

        for idx in (start + 1)..block_end {
            let prev = prev_omegas[idx - 1];
            let curr = prev_omegas[idx];
            if nearly_equal(prev, curr, ABS_TOL, REL_TOL) {
                curr_len += 1;
            } else {
                if curr_len > 1 {
                    refined.push((curr_start, curr_len));
                }
                curr_start = idx;
                curr_len = 1;
            }
        }

        if curr_len > 1 {
            refined.push((curr_start, curr_len));
        }
    }

    refined
}

/// Detect contiguous ranges where the current assignment creates a large
/// frequency mismatch. When multiple adjacent bands jump much more than their
/// neighbors, we treat them as an implicit degenerate block and reapply the
/// frequency continuity refinement pass.
fn detect_frequency_mismatch_blocks(
    permutation: &[usize],
    prev_omegas: &[f64],
    curr_omegas: &[f64],
) -> Vec<(usize, usize)> {
    const ABS_TOL: f64 = 5e-3;
    const REL_TOL: f64 = 0.02;

    let mut blocks = Vec::new();
    let mut block_start: Option<usize> = None;

    for (i, &curr_idx) in permutation.iter().enumerate() {
        if i >= prev_omegas.len() || curr_idx >= curr_omegas.len() {
            if let Some(start) = block_start.take() {
                let len = i - start;
                if len > 1 {
                    blocks.push((start, len));
                }
            }
            continue;
        }

        let mismatch = (prev_omegas[i] - curr_omegas[curr_idx]).abs();
        let scale = prev_omegas[i]
            .abs()
            .max(curr_omegas[curr_idx].abs())
            .max(1e-3);
        let is_bad = mismatch > ABS_TOL && (mismatch / scale) > REL_TOL;

        if is_bad {
            if block_start.is_none() {
                block_start = Some(i);
            }
        } else if let Some(start) = block_start.take() {
            let len = i - start;
            if len > 1 {
                blocks.push((start, len));
            }
        }
    }

    if let Some(start) = block_start {
        let len = permutation.len() - start;
        if len > 1 {
            blocks.push((start, len));
        }
    }

    blocks
}

/// Compute permutation using hybrid approach:
/// - Use overlap weights for well-separated bands
/// - Use frequency continuity within degenerate blocks
fn compute_hybrid_permutation(
    weights: &[Vec<f64>],
    degenerate_blocks: &[(usize, usize)],
    prev_omegas: &[f64],
    curr_omegas: &[f64],
) -> Vec<usize> {
    let m = weights.len();
    if m == 0 {
        return Vec::new();
    }

    // Start with the standard Hungarian assignment
    let mut permutation = hungarian_assignment(weights);

    // Refine within each degenerate block
    for &(block_start, block_size) in degenerate_blocks {
        refine_block_by_frequency(
            &mut permutation,
            block_start,
            block_size,
            prev_omegas,
            curr_omegas,
        );
    }

    permutation
}

/// Refine permutation within a single degenerate block using frequency continuity.
fn refine_block_by_frequency(
    permutation: &mut [usize],
    block_start: usize,
    block_size: usize,
    prev_omegas: &[f64],
    curr_omegas: &[f64],
) {
    let block_end = block_start + block_size;
    if block_end > permutation.len() || block_end > prev_omegas.len() {
        return;
    }

    // Get the current indices that are assigned to this block
    let block_indices: Vec<usize> = (block_start..block_end).collect();
    let curr_indices: Vec<usize> = block_indices.iter().map(|&i| permutation[i]).collect();

    // Check if all curr_indices are valid
    if curr_indices.iter().any(|&ci| ci >= curr_omegas.len()) {
        return;
    }

    // Build cost matrix: minimize |ω_curr[curr_idx] - ω_prev[block_idx]|
    // Use negative for Hungarian (it maximizes)
    let freq_weights: Vec<Vec<f64>> = block_indices
        .iter()
        .map(|&bi| {
            curr_indices
                .iter()
                .map(|&ci| -(prev_omegas[bi] - curr_omegas[ci]).abs())
                .collect()
        })
        .collect();

    // Find optimal assignment within this block
    let local_perm = hungarian_assignment(&freq_weights);

    // Apply the local permutation to the global permutation
    let old_curr_indices = curr_indices.clone();
    for (local_i, &local_j) in local_perm.iter().enumerate() {
        if local_i < block_size && local_j < block_size {
            let global_i = block_start + local_i;
            permutation[global_i] = old_curr_indices[local_j];
        }
    }

    debug!(
        "[band_tracking] Block [{}, {}): refined perm using freq continuity",
        block_start, block_end
    );
}

// ============================================================================
// Apply Permutation
// ============================================================================

/// Apply a permutation to reorder eigenvalues and eigenvectors.
///
/// After this call:
/// - `omegas[i]` corresponds to band i (tracked from previous k-point)
/// - `eigenvectors[i]` corresponds to band i
///
/// # Arguments
/// * `permutation` - permutation[i] = j means output position i gets input index j
/// * `omegas` - Eigenfrequencies to reorder in-place
/// * `eigenvectors` - Eigenvectors to reorder in-place
pub fn apply_permutation(
    permutation: &[usize],
    omegas: &mut Vec<f64>,
    eigenvectors: &mut Vec<Field2D>,
) {
    let n = permutation.len().min(omegas.len()).min(eigenvectors.len());

    // Create temporary copies
    let omegas_orig: Vec<f64> = omegas.clone();
    let eigenvectors_orig: Vec<Field2D> = eigenvectors.clone();

    // Apply permutation
    for (i, &src) in permutation.iter().enumerate().take(n) {
        if src < omegas_orig.len() {
            omegas[i] = omegas_orig[src];
        }
        if src < eigenvectors_orig.len() {
            eigenvectors[i] = eigenvectors_orig[src].clone();
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use num_complex::Complex64;

    fn make_test_field(grid: Grid2D, values: Vec<Complex64>) -> Field2D {
        Field2D::from_vec(grid, values)
    }

    // ------------------------------------------------------------------------
    // Hungarian Algorithm Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_hungarian_identity() {
        // Diagonal matrix -> identity permutation
        let weights = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let perm = hungarian_assignment(&weights);
        assert_eq!(perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_hungarian_swap() {
        // Off-diagonal peaks -> swap permutation
        let weights = vec![
            vec![0.0, 1.0, 0.0], // row 0 wants col 1
            vec![1.0, 0.0, 0.0], // row 1 wants col 0
            vec![0.0, 0.0, 1.0], // row 2 wants col 2
        ];
        let perm = hungarian_assignment(&weights);
        // perm[j] = row assigned to column j
        // col 0 -> row 1, col 1 -> row 0, col 2 -> row 2
        assert_eq!(perm, vec![1, 0, 2]);
    }

    #[test]
    fn test_hungarian_cycle() {
        // Cyclic permutation
        let weights = vec![
            vec![0.0, 1.0, 0.0], // row 0 wants col 1
            vec![0.0, 0.0, 1.0], // row 1 wants col 2
            vec![1.0, 0.0, 0.0], // row 2 wants col 0
        ];
        let perm = hungarian_assignment(&weights);
        // row 0 -> col 1, row 1 -> col 2, row 2 -> col 0
        assert_eq!(perm, vec![1, 2, 0]);
    }

    #[test]
    fn test_hungarian_ambiguous() {
        // Ambiguous case: multiple good choices, should still find optimal
        let weights = vec![
            vec![0.9, 0.8, 0.1],
            vec![0.8, 0.9, 0.1],
            vec![0.1, 0.1, 0.9],
        ];
        let perm = hungarian_assignment(&weights);
        // Optimal: col 0 -> row 0, col 1 -> row 1, col 2 -> row 2
        // Total = 0.9 + 0.9 + 0.9 = 2.7
        // Alternative: col 0 -> row 1, col 1 -> row 0, col 2 -> row 2 = 0.8 + 0.8 + 0.9 = 2.5
        assert_eq!(perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_hungarian_single() {
        let weights = vec![vec![0.5]];
        let perm = hungarian_assignment(&weights);
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_hungarian_empty() {
        let weights: Vec<Vec<f64>> = vec![];
        let perm = hungarian_assignment(&weights);
        assert!(perm.is_empty());
    }

    #[test]
    fn test_hungarian_2x2_conflict() {
        // Both rows prefer the same column, but different assignment is needed
        let weights = vec![
            vec![0.9, 0.1], // row 0 strongly prefers col 0
            vec![0.8, 0.2], // row 1 also prefers col 0, but less strongly
        ];
        let perm = hungarian_assignment(&weights);
        // Optimal: col 0 -> row 0 (0.9), col 1 -> row 1 (0.2), total = 1.1
        // Alternative: col 0 -> row 1 (0.8), col 1 -> row 0 (0.1), total = 0.9
        assert_eq!(perm, vec![0, 1]);
    }

    // ------------------------------------------------------------------------
    // Band Tracking Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_identity_tracking() {
        // When prev == curr (orthonormal), should get identity permutation
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        let v1 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
        let v2 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
        );

        let prev = vec![v1.clone(), v2.clone()];
        let curr = vec![v1, v2];

        let result = track_bands(&prev, &curr, None);

        assert_eq!(result.permutation, vec![0, 1]);
        assert!(!result.had_swaps);
        assert!(result.sigma_min > 0.9);
    }

    #[test]
    fn test_swap_tracking() {
        // When bands are swapped, should detect and return swap permutation
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        let v1 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
        let v2 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
        );

        let prev = vec![v1.clone(), v2.clone()];
        // Swap order in curr
        let curr = vec![v2, v1];

        let result = track_bands(&prev, &curr, None);

        // Should detect swap: position 0 should get curr[1], position 1 should get curr[0]
        assert_eq!(result.permutation, vec![1, 0]);
        assert!(result.had_swaps);
        assert!(result.sigma_min > 0.9);
    }

    #[test]
    fn test_apply_permutation() {
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        let v1 = make_test_field(grid, vec![Complex64::new(1.0, 0.0); 4]);
        let v2 = make_test_field(grid, vec![Complex64::new(2.0, 0.0); 4]);
        let v3 = make_test_field(grid, vec![Complex64::new(3.0, 0.0); 4]);

        let mut omegas = vec![0.1, 0.2, 0.3];
        let mut eigenvectors = vec![v1, v2, v3];

        // Permutation: [2, 0, 1] means output[0]=input[2], output[1]=input[0], output[2]=input[1]
        let permutation = vec![2, 0, 1];

        apply_permutation(&permutation, &mut omegas, &mut eigenvectors);

        assert!((omegas[0] - 0.3).abs() < 1e-10);
        assert!((omegas[1] - 0.1).abs() < 1e-10);
        assert!((omegas[2] - 0.2).abs() < 1e-10);

        // Check eigenvector values were swapped correctly
        assert!((eigenvectors[0].as_slice()[0].re - 3.0).abs() < 1e-10);
        assert!((eigenvectors[1].as_slice()[0].re - 1.0).abs() < 1e-10);
        assert!((eigenvectors[2].as_slice()[0].re - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_tracking() {
        let result = track_bands(&[], &[], None);
        assert!(result.permutation.is_empty());
        assert!(!result.had_swaps);
    }
}
