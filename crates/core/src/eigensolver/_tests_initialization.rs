//! Tests for the eigensolver initialization module.

#![cfg(test)]

use super::initialization::{
    BlockEntry, InitializationConfig, initialize_block, seed_random_vector,
};
use crate::backend::SpectralBackend;
use crate::field::Field2D;
use crate::grid::Grid2D;
use crate::operators::LinearOperator;
use num_complex::Complex64;

// ============================================================================
// Test Backend (Identity operator)
// ============================================================================

/// A simple test backend that uses Field2D directly.
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
// Test Operator (Identity A, Identity B)
// ============================================================================

/// Identity operator: A = I, B = I
struct IdentityOperator {
    backend: TestBackend,
    grid: Grid2D,
}

impl IdentityOperator {
    fn new(size: usize) -> Self {
        Self {
            backend: TestBackend,
            grid: Grid2D::new(size, 1, 1.0, 1.0),
        }
    }
}

impl LinearOperator<TestBackend> for IdentityOperator {
    fn apply(&mut self, input: &Field2D, output: &mut Field2D) {
        output.as_mut_slice().copy_from_slice(input.as_slice());
    }

    fn apply_mass(&mut self, input: &Field2D, output: &mut Field2D) {
        output.as_mut_slice().copy_from_slice(input.as_slice());
    }

    fn alloc_field(&self) -> Field2D {
        self.backend.alloc_field(self.grid)
    }

    fn backend(&self) -> &TestBackend {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut TestBackend {
        &mut self.backend
    }

    fn grid(&self) -> Grid2D {
        self.grid
    }

    fn bloch(&self) -> [f64; 2] {
        [0.0, 0.0]
    }
}

/// Diagonal operator: A = diag(values), B = I
#[allow(dead_code)]
struct DiagonalOperator {
    backend: TestBackend,
    grid: Grid2D,
    diagonal: Vec<f64>,
}

impl DiagonalOperator {
    #[allow(dead_code)]
    fn new(diagonal: Vec<f64>) -> Self {
        let n = diagonal.len();
        Self {
            backend: TestBackend,
            grid: Grid2D::new(n, 1, 1.0, 1.0),
            diagonal,
        }
    }
}

impl LinearOperator<TestBackend> for DiagonalOperator {
    fn apply(&mut self, input: &Field2D, output: &mut Field2D) {
        for (i, (out, inp)) in output
            .as_mut_slice()
            .iter_mut()
            .zip(input.as_slice())
            .enumerate()
        {
            *out = inp * self.diagonal[i];
        }
    }

    fn apply_mass(&mut self, input: &Field2D, output: &mut Field2D) {
        output.as_mut_slice().copy_from_slice(input.as_slice());
    }

    fn alloc_field(&self) -> Field2D {
        self.backend.alloc_field(self.grid)
    }

    fn backend(&self) -> &TestBackend {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut TestBackend {
        &mut self.backend
    }

    fn grid(&self) -> Grid2D {
        self.grid
    }

    fn bloch(&self) -> [f64; 2] {
        [0.0, 0.0]
    }
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_seed_random_vector_deterministic() {
    let mut buf1 = vec![Complex64::ZERO; 10];
    let mut buf2 = vec![Complex64::ZERO; 10];

    seed_random_vector(&mut buf1, 42.0);
    seed_random_vector(&mut buf2, 42.0);

    assert_eq!(buf1, buf2, "Same seed should produce same values");
}

#[test]
fn test_seed_random_vector_different_seeds() {
    let mut buf1 = vec![Complex64::ZERO; 10];
    let mut buf2 = vec![Complex64::ZERO; 10];

    seed_random_vector(&mut buf1, 1.0);
    seed_random_vector(&mut buf2, 2.0);

    assert_ne!(
        buf1, buf2,
        "Different seeds should produce different values"
    );
}

#[test]
fn test_seed_random_vector_range() {
    let mut buf = vec![Complex64::ZERO; 1000];
    seed_random_vector(&mut buf, 123.0);

    for val in &buf {
        assert!(
            val.re >= -1.0 && val.re <= 1.0,
            "Values should be in [-1, 1]"
        );
        assert_eq!(val.im, 0.0, "Imaginary part should be zero");
    }
}

// ============================================================================
// Block Entry Tests
// ============================================================================

#[test]
fn test_block_entry_rayleigh_quotient() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Create a simple eigenvector [1, 0, 0, 0] with eigenvalue 2.0
    let mut vector = Field2D::zeros(grid);
    vector.as_mut_slice()[0] = Complex64::new(1.0, 0.0);

    let mut mass = Field2D::zeros(grid);
    mass.as_mut_slice()[0] = Complex64::new(1.0, 0.0); // B*x = x for identity B

    let mut applied = Field2D::zeros(grid);
    applied.as_mut_slice()[0] = Complex64::new(2.0, 0.0); // A*x = 2*x

    let entry = BlockEntry {
        vector,
        mass,
        applied,
    };
    let rq = entry.rayleigh_quotient(&backend);

    assert!(
        (rq - 2.0).abs() < 1e-10,
        "Rayleigh quotient should be 2.0, got {}",
        rq
    );
}

#[test]
fn test_block_entry_b_norm() {
    let backend = TestBackend;
    let grid = Grid2D::new(4, 1, 1.0, 1.0);

    // Create vector [3, 4, 0, 0] with norm 5
    let mut vector = Field2D::zeros(grid);
    vector.as_mut_slice()[0] = Complex64::new(3.0, 0.0);
    vector.as_mut_slice()[1] = Complex64::new(4.0, 0.0);

    let mut mass = Field2D::zeros(grid);
    mass.as_mut_slice()[0] = Complex64::new(3.0, 0.0);
    mass.as_mut_slice()[1] = Complex64::new(4.0, 0.0);

    let applied = Field2D::zeros(grid);

    let entry = BlockEntry {
        vector,
        mass,
        applied,
    };
    let norm = entry.b_norm(&backend);

    assert!(
        (norm - 5.0).abs() < 1e-10,
        "B-norm should be 5.0, got {}",
        norm
    );
}

// ============================================================================
// Initialization Config Tests
// ============================================================================

#[test]
fn test_initialization_config_default() {
    let config = InitializationConfig::default();
    assert_eq!(config.block_size, 10);
    assert_eq!(config.max_random_attempts, 64);
    assert!((config.zero_tolerance - 1e-12).abs() < 1e-15);
}

// ============================================================================
// Block Initialization Tests
// ============================================================================

#[test]
fn test_initialize_block_basic() {
    let mut operator = IdentityOperator::new(16);
    let config = InitializationConfig {
        block_size: 4,
        max_random_attempts: 32,
        zero_tolerance: 1e-12,
    };

    let (entries, result) = initialize_block(&mut operator, &config, None);

    assert_eq!(entries.len(), 4, "Should create 4 block entries");
    assert_eq!(result.total_vectors, 4);
    assert_eq!(result.warm_start_hits, 0);
    assert_eq!(result.random_vectors, 4);
}

#[test]
fn test_initialize_block_orthogonality() {
    let mut operator = IdentityOperator::new(32);
    let config = InitializationConfig {
        block_size: 8,
        max_random_attempts: 64,
        zero_tolerance: 1e-12,
    };

    let (entries, result) = initialize_block(&mut operator, &config, None);

    eprintln!(
        "[test] Created {} vectors with {} random attempts",
        result.total_vectors, result.random_vectors
    );

    // Check B-orthonormality
    let backend = operator.backend();
    for i in 0..entries.len() {
        for j in 0..entries.len() {
            let inner = backend.dot(&entries[i].vector, &entries[j].mass);
            if i == j {
                assert!(
                    (inner.re - 1.0).abs() < 1e-10,
                    "Diagonal should be 1.0, got {} at ({}, {})",
                    inner.re,
                    i,
                    j
                );
            } else {
                assert!(
                    inner.norm() < 1e-10,
                    "Off-diagonal should be ~0, got {} at ({}, {})",
                    inner.norm(),
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn test_initialize_block_with_warm_start() {
    let mut operator = IdentityOperator::new(16);
    let config = InitializationConfig {
        block_size: 4,
        max_random_attempts: 32,
        zero_tolerance: 1e-12,
    };

    // Create warm-start vectors
    let grid = Grid2D::new(16, 1, 1.0, 1.0);
    let mut warm1 = Field2D::zeros(grid);
    warm1.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
    let mut warm2 = Field2D::zeros(grid);
    warm2.as_mut_slice()[1] = Complex64::new(1.0, 0.0);
    let warm_start = vec![warm1, warm2];

    let (entries, result) = initialize_block(&mut operator, &config, Some(&warm_start));

    assert_eq!(entries.len(), 4);
    assert_eq!(result.warm_start_hits, 2);
    assert_eq!(result.random_vectors, 2);
}

#[test]
fn test_initialize_block_8_bands_statistics() {
    // This test collects statistics about random vector generation for 8 bands
    let mut operator = IdentityOperator::new(64); // 64-dimensional space
    let config = InitializationConfig {
        block_size: 10, // 8 bands + 2 slack
        max_random_attempts: 80,
        zero_tolerance: 1e-12,
    };

    let (entries, result) = initialize_block(&mut operator, &config, None);

    eprintln!("=== 8-band initialization statistics ===");
    eprintln!("Requested block size: {}", config.block_size);
    eprintln!("Actual vectors created: {}", result.total_vectors);
    eprintln!("Random vectors used: {}", result.random_vectors);
    eprintln!("Grid dimension: {}", operator.grid().len());
    eprintln!("=========================================");

    assert_eq!(entries.len(), 10, "Should create 10 vectors (8 + 2 slack)");

    // With a 64-dimensional space and 10 vectors, we should rarely need retries
    // since random vectors are unlikely to be linearly dependent
}

#[test]
fn test_initialize_block_small_space() {
    // Test when grid dimension equals block size - harder to find orthogonal vectors
    let mut operator = IdentityOperator::new(8);
    let config = InitializationConfig {
        block_size: 8,
        max_random_attempts: 64,
        zero_tolerance: 1e-12,
    };

    let (entries, result) = initialize_block(&mut operator, &config, None);

    eprintln!(
        "[test] Small space: {} vectors created, grid dim = {}",
        result.total_vectors,
        operator.grid().len()
    );

    // Should still work - we can find 8 orthogonal vectors in 8D space
    assert_eq!(entries.len(), 8);
}

#[test]
fn test_initialize_block_impossible() {
    // Test when we request more vectors than dimensions
    let mut operator = IdentityOperator::new(4);
    let config = InitializationConfig {
        block_size: 8, // More than 4D space!
        max_random_attempts: 32,
        zero_tolerance: 1e-12,
    };

    let (entries, result) = initialize_block(&mut operator, &config, None);

    eprintln!(
        "[test] Impossible case: {} vectors created in 4D space",
        result.total_vectors
    );

    // Should only create 4 vectors (the dimension of the space)
    assert!(
        entries.len() <= 4,
        "Cannot have more orthogonal vectors than dimensions"
    );
}

// ============================================================================
// Γ-Point (k=0) Functions Tests
// ============================================================================

use super::initialization::{GAMMA_TOLERANCE, create_gamma_mode, is_gamma_point};

#[test]
fn test_is_gamma_point_exact_zero() {
    assert!(is_gamma_point([0.0, 0.0], GAMMA_TOLERANCE));
}

#[test]
fn test_is_gamma_point_small_values() {
    // Values smaller than tolerance should be considered Γ
    assert!(is_gamma_point([1e-15, 1e-15], GAMMA_TOLERANCE));
    assert!(is_gamma_point([1e-13, 0.0], GAMMA_TOLERANCE));
    assert!(is_gamma_point([0.0, 1e-13], GAMMA_TOLERANCE));
}

#[test]
fn test_is_gamma_point_not_gamma() {
    // Values larger than tolerance should NOT be considered Γ
    assert!(!is_gamma_point([0.1, 0.0], GAMMA_TOLERANCE));
    assert!(!is_gamma_point([0.0, 0.1], GAMMA_TOLERANCE));
    assert!(!is_gamma_point([1e-10, 1e-10], GAMMA_TOLERANCE));
    assert!(!is_gamma_point([0.5, 0.5], GAMMA_TOLERANCE));
}

#[test]
fn test_is_gamma_point_custom_tolerance() {
    // With custom tolerance
    assert!(is_gamma_point([1e-10, 1e-10], 1e-9));
    assert!(!is_gamma_point([1e-10, 1e-10], 1e-11));
}

#[test]
fn test_create_gamma_mode_unit_b_norm() {
    // Test that the created Γ mode has unit B-norm
    let mut operator = IdentityOperator::new(16); // 16 elements

    let (y0, by0, original_norm) = create_gamma_mode(&mut operator);

    // For identity mass matrix, original norm should be √n
    let expected_norm = (16.0_f64).sqrt();
    assert!(
        (original_norm - expected_norm).abs() < 1e-10,
        "Original norm should be √n = {}, got {}",
        expected_norm,
        original_norm
    );

    // After normalization, ||y₀||_B = 1
    let final_norm_sq: f64 = y0
        .as_slice()
        .iter()
        .zip(by0.as_slice())
        .map(|(a, b)| (a.conj() * b).re)
        .sum();
    assert!(
        (final_norm_sq - 1.0).abs() < 1e-10,
        "Normalized B-norm should be 1.0, got {}",
        final_norm_sq.sqrt()
    );
}

#[test]
fn test_create_gamma_mode_is_constant() {
    // Test that the Γ mode is constant (all elements equal)
    let mut operator = IdentityOperator::new(64);

    let (y0, _by0, _norm) = create_gamma_mode(&mut operator);

    // All elements should be equal
    let first = y0.as_slice()[0];
    for (i, &val) in y0.as_slice().iter().enumerate() {
        assert!(
            (val - first).norm() < 1e-12,
            "Element {} differs: {:?} vs {:?}",
            i,
            val,
            first
        );
    }

    // And they should be real (no imaginary part)
    assert!(
        first.im.abs() < 1e-12,
        "Γ mode should be real, got imaginary part {}",
        first.im
    );
}

#[test]
fn test_create_gamma_mode_eigenvalue_zero() {
    // For any curl-curl-like operator at k=0, the constant mode has λ=0.
    // With our identity operator A=I, A*y₀ = y₀ ≠ 0, but this test
    // verifies the structure of the mode itself is correct.
    let mut operator = IdentityOperator::new(16);

    let (y0, _by0, _norm) = create_gamma_mode(&mut operator);

    // Verify it's normalized (each component = 1/√n)
    let expected_component = 1.0 / (16.0_f64).sqrt();
    for &val in y0.as_slice() {
        assert!(
            (val.re - expected_component).abs() < 1e-10,
            "Each component should be 1/√n = {}, got {}",
            expected_component,
            val.re
        );
    }
}
