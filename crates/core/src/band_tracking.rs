//! Band tracking across k-points using eigenvector overlap.
//!
//! When computing band structures, eigenvalues at consecutive k-points may
//! swap order due to band crossings or near-degeneracies. This module provides
//! tools to track bands by their eigenvector overlap, ensuring smooth band
//! curves for visualization and analysis.
//!
//! # Algorithm
//!
//! At each k-point transition from k to k+1:
//!
//! 1. Compute the B-weighted overlap matrix:
//!    ```text
//!    O[i,j] = |⟨x_i^{(k)}, x_j^{(k+1)}⟩_B|
//!    ```
//!
//! 2. Find the optimal assignment that maximizes total overlap:
//!    ```text
//!    π* = argmax_π Σ_i O[i, π(i)]
//!    ```
//!
//! 3. Reorder eigenvalues and eigenvectors at k+1 according to π.
//!
//! # Physical Motivation
//!
//! In photonic crystals, bands represent continuous dispersion relations ω(k).
//! Near band crossings or avoided crossings, eigenvalues at adjacent k-points
//! may swap indices. By tracking which eigenvector at k+1 has maximum overlap
//! with each eigenvector at k, we can maintain consistent band labeling.

use num_complex::Complex64;

use crate::backend::SpectralBackend;
use crate::field::Field2D;

// ============================================================================
// Overlap Matrix Computation
// ============================================================================

/// Compute the B-weighted overlap matrix between two sets of eigenvectors.
///
/// Returns O where O[i,j] = |⟨prev[i], curr[j]⟩_B| (absolute value of B-inner product).
///
/// # Arguments
/// * `backend` - The spectral backend for vector operations
/// * `prev_vecs` - Eigenvectors from the previous k-point
/// * `curr_vecs` - Eigenvectors from the current k-point
/// * `eps` - The dielectric function ε(r) for B-weighting (TM mode) or None (TE mode)
///
/// For TE mode (B = I), pass `eps = None`.
/// For TM mode (B = ε), pass `eps = Some(&dielectric_values)`.
pub fn compute_overlap_matrix<B: SpectralBackend>(
    _backend: &B,
    prev_vecs: &[Field2D],
    curr_vecs: &[Field2D],
    eps: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    let n_prev = prev_vecs.len();
    let n_curr = curr_vecs.len();

    if n_prev == 0 || n_curr == 0 {
        return vec![vec![0.0; n_curr]; n_prev];
    }

    // Allocate overlap matrix
    let mut overlap = vec![vec![0.0; n_curr]; n_prev];

    // Compute overlaps
    for (i, prev) in prev_vecs.iter().enumerate() {
        for (j, curr) in curr_vecs.iter().enumerate() {
            let inner = if let Some(epsilon) = eps {
                // B-weighted inner product: ⟨prev, curr⟩_B = Σ prev^* · ε · curr
                compute_b_inner_product(prev.as_slice(), curr.as_slice(), epsilon)
            } else {
                // Standard inner product: ⟨prev, curr⟩ = Σ prev^* · curr
                compute_inner_product(prev.as_slice(), curr.as_slice())
            };

            // Store absolute value
            overlap[i][j] = inner.norm();
        }
    }

    overlap
}

/// Compute standard inner product: ⟨x, y⟩ = Σ x^* · y
fn compute_inner_product(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| xi.conj() * yi).sum()
}

/// Compute B-weighted inner product: ⟨x, y⟩_B = Σ x^* · ε · y
fn compute_b_inner_product(x: &[Complex64], y: &[Complex64], eps: &[f64]) -> Complex64 {
    x.iter()
        .zip(y.iter())
        .zip(eps.iter())
        .map(|((xi, yi), &e)| xi.conj() * yi * e)
        .sum()
}

// ============================================================================
// Optimal Assignment (Hungarian Algorithm)
// ============================================================================

/// Find the optimal band assignment that maximizes total overlap.
///
/// Given an n×m overlap matrix O, find a permutation π such that:
/// - π[i] = j means band i at prev maps to band j at curr
/// - The total overlap Σ_i O[i, π(i)] is maximized
///
/// Uses a greedy algorithm for efficiency. For small matrices (typical band
/// structure calculations have ≤20 bands), this gives near-optimal results.
///
/// # Returns
/// A vector `assignment` where `assignment[i]` is the index in `curr` that
/// best matches band `i` from `prev`. Returns `None` for unmatched bands.
pub fn find_optimal_assignment(overlap: &[Vec<f64>]) -> Vec<Option<usize>> {
    let n_prev = overlap.len();
    if n_prev == 0 {
        return Vec::new();
    }
    let n_curr = overlap[0].len();
    if n_curr == 0 {
        return vec![None; n_prev];
    }

    // Use Hungarian algorithm for optimal assignment
    // For now, use a greedy approach that works well for typical band structures
    // where overlaps are usually very high (>0.9) for correct matches

    let mut assignment = vec![None; n_prev];
    let mut used_curr: Vec<bool> = vec![false; n_curr];

    // Collect all (overlap, prev_idx, curr_idx) tuples
    let mut candidates: Vec<(f64, usize, usize)> = Vec::with_capacity(n_prev * n_curr);
    for (i, row) in overlap.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            candidates.push((val, i, j));
        }
    }

    // Sort by overlap descending (highest overlap first)
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy assignment: pick highest overlaps first
    for (_, prev_idx, curr_idx) in candidates {
        if assignment[prev_idx].is_none() && !used_curr[curr_idx] {
            assignment[prev_idx] = Some(curr_idx);
            used_curr[curr_idx] = true;
        }
    }

    assignment
}

/// Result of band tracking between consecutive k-points.
#[derive(Debug, Clone)]
pub struct BandTrackingResult {
    /// The permutation: tracked_order[i] is the original index of band i.
    pub permutation: Vec<usize>,
    /// The overlap matrix (for diagnostics).
    pub overlap_matrix: Vec<Vec<f64>>,
    /// Whether any bands were swapped.
    pub had_swaps: bool,
    /// Minimum overlap in the assignment (lower values indicate potential issues).
    pub min_overlap: f64,
}

/// Track bands between consecutive k-points and return the optimal reordering.
///
/// # Arguments
/// * `backend` - The spectral backend
/// * `prev_vecs` - Eigenvectors from previous k-point
/// * `curr_vecs` - Eigenvectors from current k-point
/// * `eps` - Dielectric function for B-weighting (None for TM mode)
///
/// # Returns
/// A `BandTrackingResult` containing the permutation to apply to the current
/// k-point's eigenvalues and eigenvectors.
pub fn track_bands<B: SpectralBackend>(
    backend: &B,
    prev_vecs: &[Field2D],
    curr_vecs: &[Field2D],
    eps: Option<&[f64]>,
) -> BandTrackingResult {
    let overlap_matrix = compute_overlap_matrix(backend, prev_vecs, curr_vecs, eps);
    let assignment = find_optimal_assignment(&overlap_matrix);

    let n = curr_vecs.len();

    // Build permutation: permutation[i] = j means position i gets the vector originally at j
    // The assignment gives us: assignment[prev_i] = Some(curr_j) means prev band i maps to curr band j
    // We want the inverse: for each output position, which input index?

    // First, build forward mapping: new_pos[curr_j] = prev_i (where prev_i had best match with curr_j)
    let mut inverse_assignment: Vec<Option<usize>> = vec![None; n];
    for (prev_i, &maybe_curr_j) in assignment.iter().enumerate() {
        if let Some(curr_j) = maybe_curr_j {
            if curr_j < n {
                inverse_assignment[curr_j] = Some(prev_i);
            }
        }
    }

    // Build permutation: reorder curr vectors so that curr[permutation[i]] corresponds to prev[i]
    // If assignment[i] = Some(j), then curr[j] should go to position i
    let mut permutation: Vec<usize> = Vec::with_capacity(n);
    let mut used: Vec<bool> = vec![false; n];

    // First pass: place matched bands
    for (prev_i, &maybe_curr_j) in assignment.iter().enumerate() {
        if prev_i < n {
            if let Some(curr_j) = maybe_curr_j {
                if curr_j < n {
                    permutation.push(curr_j);
                    used[curr_j] = true;
                    continue;
                }
            }
            // No match for this prev band, will fill later
            permutation.push(usize::MAX); // placeholder
        }
    }

    // Extend if curr has more bands than prev
    while permutation.len() < n {
        permutation.push(usize::MAX);
    }

    // Second pass: fill unmatched positions with remaining curr indices
    let mut next_unused = 0;
    for i in 0..n {
        if permutation[i] == usize::MAX {
            // Find next unused curr index
            while next_unused < n && used[next_unused] {
                next_unused += 1;
            }
            if next_unused < n {
                permutation[i] = next_unused;
                used[next_unused] = true;
            }
        }
    }

    // Check for swaps
    let had_swaps = permutation.iter().enumerate().any(|(i, &p)| i != p);

    // Compute minimum overlap in the assignment
    let mut min_overlap = f64::INFINITY;
    for (prev_i, &maybe_curr_j) in assignment.iter().enumerate() {
        if let Some(curr_j) = maybe_curr_j {
            if prev_i < overlap_matrix.len() && curr_j < overlap_matrix[prev_i].len() {
                min_overlap = min_overlap.min(overlap_matrix[prev_i][curr_j]);
            }
        }
    }
    if min_overlap == f64::INFINITY {
        min_overlap = 0.0;
    }

    BandTrackingResult {
        permutation,
        overlap_matrix,
        had_swaps,
        min_overlap,
    }
}

/// Apply a permutation to reorder eigenvalues and eigenvectors.
///
/// After this call:
/// - `omegas[i]` corresponds to band i (tracked from previous k-point)
/// - `eigenvectors[i]` corresponds to band i
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_assignment_identity() {
        // Perfect diagonal overlap matrix -> identity permutation
        let overlap = vec![
            vec![1.0, 0.1, 0.1],
            vec![0.1, 1.0, 0.1],
            vec![0.1, 0.1, 1.0],
        ];

        let assignment = find_optimal_assignment(&overlap);
        assert_eq!(assignment, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_greedy_assignment_swap() {
        // Bands 0 and 1 are swapped
        let overlap = vec![
            vec![0.1, 0.9, 0.1], // prev[0] matches best with curr[1]
            vec![0.9, 0.1, 0.1], // prev[1] matches best with curr[0]
            vec![0.1, 0.1, 0.9], // prev[2] matches best with curr[2]
        ];

        let assignment = find_optimal_assignment(&overlap);
        assert_eq!(assignment, vec![Some(1), Some(0), Some(2)]);
    }

    #[test]
    fn test_greedy_assignment_rectangular() {
        // More curr bands than prev bands
        let overlap = vec![vec![0.9, 0.1, 0.1, 0.1], vec![0.1, 0.9, 0.1, 0.1]];

        let assignment = find_optimal_assignment(&overlap);
        assert_eq!(assignment, vec![Some(0), Some(1)]);
    }
}
