//! Tests for the eigensolver normalization module.

#![cfg(test)]

use super::normalization::{
    SvqbConfig, b_inner_product, b_norm, normalize_to_unit_b_norm,
    normalize_to_unit_b_norm_with_tol, orthogonalize_against_basis, orthonormalize_against_basis,
    project_out, svqb_orthonormalize, zero_buffer,
};
use crate::backend::SpectralBackend;
use crate::field::Field2D;
use crate::grid::Grid2D;
use num_complex::Complex64;

// ============================================================================
// Test Backend
// ============================================================================

/// A simple test backend using Field2D with identity mass matrix.
#[derive(Clone, Copy, Default)]
struct TestBackend;

impl SpectralBackend for TestBackend {
    type Buffer = Field2D;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in buffer.as_mut_slice() {
            *value *= alpha;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *dst += alpha * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn make_field(grid: Grid2D, values: &[Complex64]) -> Field2D {
    let mut field = Field2D::zeros(grid);
    for (i, &v) in values.iter().enumerate() {
        if i < field.as_slice().len() {
            field.as_mut_slice()[i] = v;
        }
    }
    field
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// ============================================================================
// B-Norm Tests
// ============================================================================

#[test]
fn test_b_norm_unit_vector() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let vector = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mass = vector.clone(); // Identity B

    let norm = b_norm(&backend, &vector, &mass);
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "Expected norm 1.0, got {}",
        norm
    );
}

#[test]
fn test_b_norm_scaled_vector() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // [3, 4, 0, 0] has norm 5
    let vector = make_field(grid, &[c(3.0, 0.0), c(4.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mass = vector.clone();

    let norm = b_norm(&backend, &vector, &mass);
    assert!(
        (norm - 5.0).abs() < 1e-10,
        "Expected norm 5.0, got {}",
        norm
    );
}

#[test]
fn test_b_norm_complex_vector() {
    let backend = TestBackend;
    let grid = Grid2D::new(2, 1, 1.0, 1.0);

    // [1+i, 1-i] has squared norm |1+i|^2 + |1-i|^2 = 2 + 2 = 4
    let vector = make_field(grid, &[c(1.0, 1.0), c(1.0, -1.0)]);
    let mass = vector.clone();

    let norm = b_norm(&backend, &vector, &mass);
    assert!(
        (norm - 2.0).abs() < 1e-10,
        "Expected norm 2.0, got {}",
        norm
    );
}

// ============================================================================
// Normalization Tests
// ============================================================================

#[test]
fn test_normalize_to_unit_b_norm() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let mut vector = make_field(grid, &[c(3.0, 0.0), c(4.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();

    let original_norm = normalize_to_unit_b_norm(&backend, &mut vector, &mut mass);

    assert!(
        (original_norm - 5.0).abs() < 1e-10,
        "Original norm should be 5.0"
    );

    let new_norm = b_norm(&backend, &vector, &mass);
    assert!(
        (new_norm - 1.0).abs() < 1e-10,
        "New norm should be 1.0, got {}",
        new_norm
    );
}

#[test]
fn test_normalize_zero_vector() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let mut vector = Field2D::zeros(grid);
    let mut mass = Field2D::zeros(grid);

    let original_norm = normalize_to_unit_b_norm(&backend, &mut vector, &mut mass);

    assert!(
        original_norm.abs() < 1e-10,
        "Zero vector should have zero norm"
    );

    // Vector should still be zero
    for val in vector.as_slice() {
        assert!(val.norm() < 1e-10);
    }
}

#[test]
fn test_normalize_with_custom_tolerance() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Very small vector
    let mut vector = make_field(
        grid,
        &[c(1e-10, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    );
    let mut mass = vector.clone();

    // With default tolerance (1e-15), should normalize
    let norm1 = normalize_to_unit_b_norm(&backend, &mut vector, &mut mass);
    assert!(norm1 > 0.0);

    // Reset
    vector = make_field(
        grid,
        &[c(1e-10, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    );
    mass = vector.clone();

    // With large tolerance (1e-8), should NOT normalize
    let norm2 = normalize_to_unit_b_norm_with_tol(&backend, &mut vector, &mut mass, 1e-8);
    assert!(norm2.abs() < 1e-10, "Should return 0 when below tolerance");
}

// ============================================================================
// B-Inner Product Tests
// ============================================================================

#[test]
fn test_b_inner_product_orthogonal() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let x = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let y = make_field(grid, &[c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let by = y.clone(); // Identity B

    let inner = b_inner_product(&backend, &x, &by);
    assert!(
        inner.norm() < 1e-10,
        "Orthogonal vectors should have zero inner product"
    );
}

#[test]
fn test_b_inner_product_parallel() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let x = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let bx = x.clone();

    let inner = b_inner_product(&backend, &x, &bx);
    assert!(
        (inner.re - 1.0).abs() < 1e-10,
        "Self inner product should be 1.0"
    );
    assert!(inner.im.abs() < 1e-10, "Self inner product should be real");
}

// ============================================================================
// Projection Tests
// ============================================================================

#[test]
fn test_project_out_orthogonal() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Already orthogonal vectors
    let mut vector = make_field(grid, &[c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();
    let basis = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let basis_mass = basis.clone();

    project_out(&backend, &mut vector, &mut mass, &basis, &basis_mass);

    // Should remain unchanged
    assert!((vector.as_slice()[0].norm()).abs() < 1e-10);
    assert!((vector.as_slice()[1].re - 1.0).abs() < 1e-10);
}

#[test]
fn test_project_out_parallel() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Parallel vectors
    let mut vector = make_field(grid, &[c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();
    let basis = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let basis_mass = basis.clone();

    project_out(&backend, &mut vector, &mut mass, &basis, &basis_mass);

    // Should become zero
    for val in vector.as_slice() {
        assert!(
            val.norm() < 1e-10,
            "Projected parallel vector should be zero"
        );
    }
}

#[test]
fn test_project_out_partial() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Vector with parallel and orthogonal components
    let mut vector = make_field(grid, &[c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();
    let basis = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let basis_mass = basis.clone();

    project_out(&backend, &mut vector, &mut mass, &basis, &basis_mass);

    // First component should be projected out, second should remain
    assert!(vector.as_slice()[0].norm() < 1e-10);
    assert!((vector.as_slice()[1].re - 1.0).abs() < 1e-10);
}

// ============================================================================
// Orthogonalization Against Basis Tests
// ============================================================================

#[test]
fn test_orthogonalize_against_empty_basis() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let mut vector = make_field(grid, &[c(3.0, 0.0), c(4.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();

    let norm = orthogonalize_against_basis(&backend, &mut vector, &mut mass, &[], &[]);

    assert!((norm - 5.0).abs() < 1e-10, "Norm should be unchanged");
}

#[test]
fn test_orthonormalize_success() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    let mut vector = make_field(grid, &[c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();
    let basis = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let basis_mass = basis.clone();

    let success = orthonormalize_against_basis(
        &backend,
        &mut vector,
        &mut mass,
        &[basis],
        &[basis_mass],
        1e-12,
    );

    assert!(success, "Should succeed");

    // Should be normalized
    let norm = b_norm(&backend, &vector, &mass);
    assert!((norm - 1.0).abs() < 1e-10);

    // Should be orthogonal to basis
    let original_basis = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let inner = b_inner_product(&backend, &vector, &original_basis);
    assert!(inner.norm() < 1e-10);
}

#[test]
fn test_orthonormalize_failure_linearly_dependent() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Vector that is in span of basis
    let mut vector = make_field(grid, &[c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut mass = vector.clone();
    let basis = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let basis_mass = basis.clone();

    let success = orthonormalize_against_basis(
        &backend,
        &mut vector,
        &mut mass,
        &[basis],
        &[basis_mass],
        1e-12,
    );

    assert!(!success, "Should fail for linearly dependent vector");
}

// ============================================================================
// Zero Buffer Tests
// ============================================================================

#[test]
fn test_zero_buffer() {
    let mut buf = vec![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)];
    zero_buffer(&mut buf);

    for val in &buf {
        assert_eq!(*val, Complex64::ZERO);
    }
}

// ============================================================================
// SVQB Tests
// ============================================================================

#[test]
fn test_svqb_empty() {
    let backend = TestBackend;
    let config = SvqbConfig::default();

    let mut vectors: Vec<Field2D> = vec![];
    let mut mass_vectors: Vec<Field2D> = vec![];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 0);
    assert_eq!(result.input_count, 0);
}

#[test]
fn test_svqb_single_vector() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    let v = make_field(grid, &[c(3.0, 0.0), c(4.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let mut vectors = vec![v.clone()];
    let mut mass_vectors = vec![v];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 1);
    assert_eq!(result.dropped_count, 0);

    // Should be normalized
    let norm = b_norm(&backend, &vectors[0], &mass_vectors[0]);
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "Should be unit norm, got {}",
        norm
    );
}

#[test]
fn test_svqb_orthogonal_vectors() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(0.0, 0.0), c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 2);
    assert_eq!(result.dropped_count, 0);

    // Both should be normalized
    for i in 0..2 {
        let norm = b_norm(&backend, &vectors[i], &mass_vectors[i]);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Vector {} should have unit norm",
            i
        );
    }

    // Should be orthogonal
    let inner = b_inner_product(&backend, &vectors[0], &mass_vectors[1]);
    assert!(inner.norm() < 1e-10, "Vectors should be orthogonal");
}

#[test]
fn test_svqb_nearly_parallel_vectors() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Two nearly parallel vectors
    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(
        grid,
        &[c(1.0, 0.0), c(1e-14, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    );

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    // Should detect rank deficiency
    assert_eq!(result.output_rank, 1, "Should keep only 1 vector");
    assert_eq!(result.dropped_count, 1, "Should drop 1 vector");
    assert!(result.had_rank_deficiency());

    // First vector should be normalized
    let norm = b_norm(&backend, &vectors[0], &mass_vectors[0]);
    assert!((norm - 1.0).abs() < 1e-10);

    // Second slot should be zeroed
    for val in vectors[1].as_slice() {
        assert!(val.norm() < 1e-10);
    }
}

#[test]
fn test_svqb_three_vectors_one_dependent() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Three vectors where third is sum of first two
    // This creates a rank-2 Gram matrix (the third column is linearly dependent)
    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v3 = make_field(grid, &[c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone(), v3.clone()];
    let mut mass_vectors = vec![v1, v2, v3];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 2, "Should keep only 2 vectors");
    assert_eq!(result.dropped_count, 1, "Should drop 1 vector");
}

#[test]
fn test_svqb_condition_number() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Well-conditioned orthogonal vectors with different norms
    let v1 = make_field(grid, &[c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    // Condition number should be reasonable (ratio of original norms squared = 4)
    let cond = result.condition_number();
    assert!(
        cond >= 1.0 && cond < 10.0,
        "Condition number should be ~4, got {}",
        cond
    );
}

// ============================================================================
// SVQB Critical Edge Cases for LOBPCG
// ============================================================================

#[test]
fn test_svqb_complex_eigenvectors() {
    // In photonic crystals at general k-points, eigenvectors are complex
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Two complex orthogonal vectors: [1, i, 0, 0] and [i, 1, 0, 0]
    // Inner product: conj([1,i]) · [i,1] = 1*i + (-i)*1 = i - i = 0 ✓
    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 1.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(0.0, 1.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 2, "Should keep both complex vectors");

    // Verify B-orthonormality
    for i in 0..2 {
        let norm = b_norm(&backend, &vectors[i], &mass_vectors[i]);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Vector {} should have unit norm, got {}",
            i,
            norm
        );
    }

    let inner = b_inner_product(&backend, &vectors[0], &mass_vectors[1]);
    assert!(
        inner.norm() < 1e-10,
        "Complex vectors should be orthogonal, got {}",
        inner.norm()
    );
}

#[test]
fn test_svqb_twofold_degeneracy() {
    // Simulates 2-fold degeneracy at X-point: two vectors spanning the same eigenspace
    // After Rayleigh-Ritz, they might be nearly parallel
    let backend = TestBackend;
    let grid = Grid2D::new(8, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Two vectors that span a 2D subspace at an angle
    // The Gram matrix eigenvalue ratio should be above 1e-12 to keep both
    // Using angle θ = 0.01 rad gives sin²(θ) ≈ 1e-4, well above threshold
    //
    // v1 = [1, 0, 0, 0, 0, 0, 0, 0]
    // v2 = [cos(θ), sin(θ), 0, 0, 0, 0, 0, 0] with θ = 0.01
    let theta = 0.01_f64; // ~0.57 degrees
    let v1 = make_field(
        grid,
        &[
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );
    let v2 = make_field(
        grid,
        &[
            c(theta.cos(), 0.0),
            c(theta.sin(), 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    // Should keep both since smallest Gram eigenvalue ≈ sin²(θ/2) ≈ 2.5e-5 >> threshold
    assert_eq!(
        result.output_rank, 2,
        "Should keep both vectors for 2-fold degeneracy"
    );

    // Critically: output must be orthonormal even for nearly parallel input
    let inner = b_inner_product(&backend, &vectors[0], &mass_vectors[1]);
    assert!(
        inner.norm() < 1e-10,
        "Output must be orthogonal, got inner product {}",
        inner.norm()
    );
}

#[test]
fn test_svqb_threefold_degeneracy() {
    // Simulates 3-fold degeneracy at Γ-point with symmetry
    let backend = TestBackend;
    let grid = Grid2D::new(6, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Three vectors spanning a 2D subspace (one is dependent)
    // v1 = [1, 0, 0, 0, 0, 0]
    // v2 = [0, 1, 0, 0, 0, 0]
    // v3 = [0.6, 0.8, 0, 0, 0, 0] = 0.6*v1 + 0.8*v2
    let v1 = make_field(
        grid,
        &[
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );
    let v2 = make_field(
        grid,
        &[
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );
    let v3 = make_field(
        grid,
        &[
            c(0.6, 0.0),
            c(0.8, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );

    let mut vectors = vec![v1.clone(), v2.clone(), v3.clone()];
    let mut mass_vectors = vec![v1, v2, v3];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(
        result.output_rank, 2,
        "Should detect rank-2 for 3 vectors in 2D subspace"
    );

    // All kept vectors must be orthonormal
    for i in 0..result.output_rank {
        let norm = b_norm(&backend, &vectors[i], &mass_vectors[i]);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Vector {} not normalized: {}",
            i,
            norm
        );

        for j in (i + 1)..result.output_rank {
            let inner = b_inner_product(&backend, &vectors[i], &mass_vectors[j]);
            assert!(
                inner.norm() < 1e-10,
                "Vectors {} and {} not orthogonal: {}",
                i,
                j,
                inner.norm()
            );
        }
    }
}

#[test]
fn test_svqb_mixed_scales() {
    // Some vectors have large norm, others small - tests numerical stability
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // v1 has norm 1000, v2 has norm 0.001
    let v1 = make_field(
        grid,
        &[c(1000.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    );
    let v2 = make_field(
        grid,
        &[c(0.0, 0.0), c(0.001, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    );

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    // Both should be kept (they're orthogonal)
    assert_eq!(
        result.output_rank, 2,
        "Should keep both vectors despite scale difference"
    );

    // Both must be normalized to 1
    for i in 0..2 {
        let norm = b_norm(&backend, &vectors[i], &mass_vectors[i]);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Vector {} should be normalized, got {}",
            i,
            norm
        );
    }
}

#[test]
fn test_svqb_ill_conditioned_gram() {
    // Gram matrix with very high condition number
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Two nearly parallel vectors with small angle
    // This creates an ill-conditioned Gram matrix
    let angle: f64 = 1e-6; // radians
    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(
        grid,
        &[
            c(angle.cos(), 0.0),
            c(angle.sin(), 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    // The angle is ~1e-6, so sin²(θ) ≈ 1e-12, which is at the threshold
    // Either 1 or 2 vectors may be kept depending on exact numerics
    assert!(result.output_rank >= 1 && result.output_rank <= 2);

    // What matters: output is well-conditioned
    if result.output_rank == 2 {
        let inner = b_inner_product(&backend, &vectors[0], &mass_vectors[1]);
        assert!(inner.norm() < 1e-8, "Output must be orthogonal");
    }
}

#[test]
fn test_svqb_preserves_subspace() {
    // Critical: SVQB must preserve the span of the input vectors
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Input spans a specific 2D subspace
    let v1 = make_field(grid, &[c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(1.0, 0.0), c(-1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 2);

    // Output vectors should span the same subspace: only have support in first 2 components
    for i in 0..2 {
        let data = vectors[i].as_slice();
        assert!(
            data[2].norm() < 1e-10,
            "Vector {} leaked into component 2",
            i
        );
        assert!(
            data[3].norm() < 1e-10,
            "Vector {} leaked into component 3",
            i
        );
    }
}

#[test]
fn test_svqb_hermitian_with_phases() {
    // Complex vectors with various phases - tests the Jacobi algorithm's phase handling
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Three vectors with complex phases
    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(0.0, 0.0), c(0.0, 1.0), c(0.0, 0.0), c(0.0, 0.0)]); // pure imaginary
    let v3 = make_field(
        grid,
        &[c(0.0, 0.0), c(0.0, 0.0), c(0.707, 0.707), c(0.0, 0.0)],
    ); // 45° phase

    let mut vectors = vec![v1.clone(), v2.clone(), v3.clone()];
    let mut mass_vectors = vec![v1, v2, v3];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(
        result.output_rank, 3,
        "All three orthogonal vectors should be kept"
    );

    // Verify full orthonormality
    for i in 0..3 {
        let norm = b_norm(&backend, &vectors[i], &mass_vectors[i]);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Vector {} not normalized: {}",
            i,
            norm
        );

        for j in (i + 1)..3 {
            let inner = b_inner_product(&backend, &vectors[i], &mass_vectors[j]);
            assert!(
                inner.norm() < 1e-10,
                "Vectors {} and {} not orthogonal: {}",
                i,
                j,
                inner.norm()
            );
        }
    }
}

#[test]
fn test_svqb_all_zero_vectors() {
    // Edge case: all input vectors are zero
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    let v1 = Field2D::zeros(grid);
    let v2 = Field2D::zeros(grid);

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 0, "Zero vectors should all be dropped");
    assert_eq!(result.dropped_count, 2);
}

#[test]
fn test_svqb_one_zero_one_nonzero() {
    // Mix of zero and non-zero vectors
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    let v1 = make_field(grid, &[c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = Field2D::zeros(grid);
    let v3 = make_field(grid, &[c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone(), v3.clone()];
    let mut mass_vectors = vec![v1, v2, v3];

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 2, "Should keep 2 non-zero vectors");
    assert_eq!(result.dropped_count, 1, "Should drop 1 zero vector");
}

#[test]
fn test_svqb_large_block() {
    // Test with larger block size typical for LOBPCG (8-10 vectors)
    let backend = TestBackend;
    let grid = Grid2D::new(16, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    // Create 8 orthogonal unit vectors
    let mut vectors = Vec::new();
    let mut mass_vectors = Vec::new();

    for i in 0..8 {
        let mut data = vec![c(0.0, 0.0); 16];
        data[i] = c(1.0, 0.0);
        let v = make_field(grid, &data);
        vectors.push(v.clone());
        mass_vectors.push(v);
    }

    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(
        result.output_rank, 8,
        "All 8 orthogonal vectors should be kept"
    );

    // Full orthonormality check
    for i in 0..8 {
        let norm = b_norm(&backend, &vectors[i], &mass_vectors[i]);
        assert!((norm - 1.0).abs() < 1e-10, "Vector {} not normalized", i);

        for j in (i + 1)..8 {
            let inner = b_inner_product(&backend, &vectors[i], &mass_vectors[j]);
            assert!(
                inner.norm() < 1e-10,
                "Vectors {} and {} not orthogonal",
                i,
                j
            );
        }
    }
}

#[test]
fn test_svqb_numerical_stability_repeated() {
    // Apply SVQB twice - should be idempotent
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let config = SvqbConfig::default();

    let v1 = make_field(grid, &[c(3.0, 1.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
    let v2 = make_field(grid, &[c(0.0, 0.0), c(2.0, -1.0), c(0.0, 0.0), c(0.0, 0.0)]);

    let mut vectors = vec![v1.clone(), v2.clone()];
    let mut mass_vectors = vec![v1, v2];

    // First application
    svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    // Second application (should be idempotent)
    let result = svqb_orthonormalize(&backend, &mut vectors, &mut mass_vectors, &config);

    assert_eq!(result.output_rank, 2, "Should still have rank 2");

    // Vectors should be essentially unchanged (idempotent within tolerance)
    // Note: phase may differ, so check that they span the same subspace
    let norm1 = b_norm(&backend, &vectors[0], &mass_vectors[0]);
    let norm2 = b_norm(&backend, &vectors[1], &mass_vectors[1]);
    assert!((norm1 - 1.0).abs() < 1e-10);
    assert!((norm2 - 1.0).abs() < 1e-10);
}
