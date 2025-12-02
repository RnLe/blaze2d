//! Tests for the eigensolver module.

#![cfg(test)]

use super::backend::SpectralBackend;
use super::eigensolver::{BandState, ConvergenceInfo, Eigensolver, EigensolverConfig};
use super::field::Field2D;
use super::grid::Grid2D;
use super::operators::LinearOperator;
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
// Test Operator (Diagonal matrix)
// ============================================================================

/// A simple diagonal operator for testing.
/// A = diag(eigenvalues), B = I (identity mass matrix)
struct DiagonalOperator {
    backend: TestBackend,
    grid: Grid2D,
    diagonal: Vec<f64>,
}

impl DiagonalOperator {
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
        // Identity mass matrix
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
        // Non-zero to avoid Γ-point deflation in tests
        // (we want to test general LOBPCG behavior, not Γ-specific handling)
        [0.1, 0.1]
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_config_default() {
    let config = EigensolverConfig::default();
    assert_eq!(config.n_bands, 8);
    assert_eq!(config.max_iter, 200);
    assert!((config.tol - 1e-6).abs() < 1e-10);
    assert_eq!(config.effective_block_size(), 10); // 8 + 2 slack
}

#[test]
fn test_config_effective_block_size() {
    let mut config = EigensolverConfig::default();
    config.n_bands = 5;
    assert_eq!(config.effective_block_size(), 7); // 5 + 2 slack

    config.block_size = 10;
    assert_eq!(config.effective_block_size(), 10); // explicit override
}

#[test]
fn test_config_effective_block_size_minimum() {
    let mut config = EigensolverConfig::default();
    config.n_bands = 10;
    config.block_size = 5; // Less than n_bands
    assert_eq!(config.effective_block_size(), 10); // Should be at least n_bands
}

// ============================================================================
// Eigensolver Initialization Tests
// ============================================================================

#[test]
fn test_eigensolver_creation() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0]);
    let config = EigensolverConfig {
        n_bands: 2,
        max_iter: 100,
        tol: 1e-8,
        block_size: 0,
        ..Default::default()
    };

    let solver = Eigensolver::new(&mut operator, config, None, None);

    assert!(!solver.is_initialized());
    assert_eq!(solver.iteration(), 0);
    assert_eq!(solver.block_size(), 0); // Not initialized yet
}

#[test]
fn test_eigensolver_initialize() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let config = EigensolverConfig {
        n_bands: 4,
        max_iter: 100,
        tol: 1e-8,
        block_size: 0, // Auto: 4 + 2 = 6
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    solver.initialize();

    assert!(solver.is_initialized());
    assert_eq!(solver.block_size(), 6); // 4 bands + 2 slack
    assert_eq!(solver.eigenvalues().len(), 6);
}

#[test]
fn test_eigensolver_double_initialize() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0]);
    let config = EigensolverConfig {
        n_bands: 2,
        max_iter: 100,
        tol: 1e-8,
        block_size: 3,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    solver.initialize();
    let block_size_1 = solver.block_size();

    // Second initialize should be a no-op
    solver.initialize();
    let block_size_2 = solver.block_size();

    assert_eq!(block_size_1, block_size_2);
}

#[test]
fn test_eigensolver_has_preconditioner() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0]);
    let config = EigensolverConfig::default();

    let solver = Eigensolver::new(&mut operator, config, None, None);
    assert!(!solver.has_preconditioner());
}

// ============================================================================
// Convergence Tracking Tests
// ============================================================================

#[test]
fn test_convergence_info_new() {
    let info = ConvergenceInfo::new(5);
    assert_eq!(info.band_states.len(), 5);
    assert_eq!(info.relative_residuals.len(), 5);
    assert_eq!(info.n_converged, 0);
    assert!(!info.all_converged);
    assert!(info.max_residual.is_infinite());

    for state in &info.band_states {
        assert_eq!(*state, BandState::Active);
    }
}

#[test]
fn test_convergence_info_update_none_converged() {
    let mut info = ConvergenceInfo::new(3);
    let residuals = vec![1e-3, 1e-4, 1e-5];
    let tol = 1e-6;

    info.update(&residuals, tol);

    assert_eq!(info.n_converged, 0);
    assert!(!info.all_converged);
    assert_eq!(
        info.band_states,
        vec![BandState::Active, BandState::Active, BandState::Active]
    );
    assert!((info.max_residual - 1e-3).abs() < 1e-10);
}

#[test]
fn test_convergence_info_update_some_converged() {
    let mut info = ConvergenceInfo::new(4);
    let residuals = vec![1e-8, 1e-3, 1e-9, 1e-4];
    let tol = 1e-6;

    info.update(&residuals, tol);

    assert_eq!(info.n_converged, 2);
    assert!(!info.all_converged);
    assert_eq!(info.band_states[0], BandState::Converged);
    assert_eq!(info.band_states[1], BandState::Active);
    assert_eq!(info.band_states[2], BandState::Converged);
    assert_eq!(info.band_states[3], BandState::Active);
    assert!((info.max_residual - 1e-3).abs() < 1e-10);
}

#[test]
fn test_convergence_info_update_all_converged() {
    let mut info = ConvergenceInfo::new(3);
    let residuals = vec![1e-8, 1e-9, 1e-7];
    let tol = 1e-6;

    info.update(&residuals, tol);

    assert_eq!(info.n_converged, 3);
    assert!(info.all_converged);
    assert_eq!(
        info.band_states,
        vec![
            BandState::Converged,
            BandState::Converged,
            BandState::Converged
        ]
    );
    // max_residual should be 0 since no bands are active
    assert_eq!(info.max_residual, 0.0);
}

#[test]
fn test_convergence_info_locked_bands_preserved() {
    let mut info = ConvergenceInfo::new(3);

    // First update converges band 0
    info.update(&[1e-8, 1e-3, 1e-4], 1e-6);
    assert_eq!(info.band_states[0], BandState::Converged);

    // Lock band 0
    info.band_states[0] = BandState::Locked;

    // Second update - locked band stays locked even with high residual
    info.update(&[1.0, 1e-3, 1e-4], 1e-6);
    assert_eq!(info.band_states[0], BandState::Locked);
    // Locked bands count as converged
    assert_eq!(info.n_converged, 1);
}

// ============================================================================
// Residual Computation Tests
// ============================================================================

#[test]
fn test_solve_computes_residuals() {
    // Create an operator where we know the exact eigenvalues
    // A = diag(1, 2, 3, 4), B = I
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let config = EigensolverConfig {
        n_bands: 4,
        max_iter: 1,
        tol: 1e-8,
        block_size: 4,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let result = solver.solve();

    // After solve, we should have eigenvalue estimates
    assert_eq!(result.eigenvalues.len(), 4);
    assert_eq!(result.iterations, 1);

    // With random initial vectors and only 1 iteration, we won't converge
    assert!(!result.converged);
}

#[test]
fn test_solve_with_exact_eigenvector() {
    // If we start with exact eigenvectors, residual should be zero
    // This tests that our residual formula R = A*x - λ*B*x is correct
    // Note: With eigenvalue-based convergence, we need at least 2 iterations:
    // - Iteration 1: establishes eigenvalue estimates
    // - Iteration 2: confirms eigenvalues haven't changed (Δλ ≈ 0)

    let diagonal = vec![1.0, 4.0, 9.0, 16.0];
    let mut operator = DiagonalOperator::new(diagonal.clone());
    let config = EigensolverConfig {
        n_bands: 2,
        max_iter: 100,
        tol: 1e-10,
        block_size: 2,
        ..Default::default()
    };

    // Create warm-start vectors that are exact eigenvectors
    // e_0 = [1, 0, 0, 0] with eigenvalue 1
    // e_1 = [0, 1, 0, 0] with eigenvalue 4
    let grid = Grid2D::new(4, 1, 1.0, 1.0);
    let mut e0 = Field2D::zeros(grid);
    let mut e1 = Field2D::zeros(grid);
    e0.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
    e1.as_mut_slice()[1] = Complex64::new(1.0, 0.0);
    let warm_start = vec![e0, e1];

    let mut solver = Eigensolver::new(&mut operator, config, None, Some(&warm_start));
    let result = solver.solve();

    // With exact eigenvectors, should converge very quickly
    // (typically 2-3 iterations: to establish eigenvalues and confirm convergence)
    assert!(result.converged, "Should converge with exact eigenvectors");
    assert!(
        result.iterations <= 3,
        "Should converge in 3 iterations or less, got {}",
        result.iterations
    );

    // Eigenvalues should be exact
    assert!(
        (result.eigenvalues[0] - 1.0).abs() < 1e-10,
        "λ_0 should be 1.0, got {}",
        result.eigenvalues[0]
    );
    assert!(
        (result.eigenvalues[1] - 4.0).abs() < 1e-10,
        "λ_1 should be 4.0, got {}",
        result.eigenvalues[1]
    );
}

#[test]
fn test_eigenvectors_extraction() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0]);
    let config = EigensolverConfig {
        n_bands: 2,
        max_iter: 1,
        tol: 1e-8,
        block_size: 3,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    solver.initialize();

    let eigenvectors = solver.eigenvectors();
    assert_eq!(eigenvectors.len(), 2); // n_bands, not block_size

    // Each eigenvector should have the right grid
    for v in &eigenvectors {
        assert_eq!(v.len(), 4);
    }
}

#[test]
fn test_eigensolverresult_structure() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let config = EigensolverConfig {
        n_bands: 4,
        max_iter: 1,
        tol: 1e-8,
        block_size: 0,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let result = solver.solve();

    // Check result structure
    assert_eq!(result.eigenvalues.len(), 4);
    assert_eq!(result.convergence.band_states.len(), 4);
    assert_eq!(result.convergence.relative_residuals.len(), 4);
    assert!(result.iterations >= 1);
}

// ============================================================================
// Preconditioning Tests
// ============================================================================

/// A simple scaling preconditioner for testing.
/// Multiplies each component by a fixed scale factor.
struct ScalingPreconditioner {
    scale: f64,
}

impl crate::preconditioners::OperatorPreconditioner<TestBackend> for ScalingPreconditioner {
    fn apply(&mut self, _backend: &TestBackend, buffer: &mut Field2D) {
        for value in buffer.as_mut_slice() {
            *value *= self.scale;
        }
    }
}

#[test]
fn test_solve_without_preconditioner() {
    // Without preconditioner, P_k = R_k (identity preconditioning)
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0]);
    let config = EigensolverConfig {
        n_bands: 2,
        max_iter: 1,
        tol: 1e-8,
        block_size: 2,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let result = solver.solve();

    // Should complete without error
    assert_eq!(result.iterations, 1);
}

#[test]
fn test_solve_with_preconditioner() {
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0]);
    let config = EigensolverConfig {
        n_bands: 2,
        max_iter: 1,
        tol: 1e-8,
        block_size: 2,
        ..Default::default()
    };

    let mut precond = ScalingPreconditioner { scale: 0.5 };
    let mut solver = Eigensolver::new(&mut operator, config, Some(&mut precond), None);

    assert!(solver.has_preconditioner());

    let result = solver.solve();

    // Should complete without error
    assert_eq!(result.iterations, 1);
}

// ============================================================================
// Subspace Construction Tests
// ============================================================================

#[test]
fn test_subspace_dimension_first_iteration() {
    // On first iteration, Z_k = [X_k, P_k] has dimension 2m
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let config = EigensolverConfig {
        n_bands: 4,
        max_iter: 1,
        tol: 1e-8,
        block_size: 4,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    solver.initialize();

    // First iteration: no W_k yet
    assert_eq!(solver.subspace_dimension(), 2 * 4); // 2m = 8
}

#[test]
fn test_subspace_dimension_after_first_iteration() {
    // After first iteration, W_k is populated, so Z_k = [X_k, P_k, W_k] has dimension 3m
    let mut operator = DiagonalOperator::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let config = EigensolverConfig {
        n_bands: 4,
        max_iter: 2, // Allow 2 iterations
        tol: 1e-15,  // Won't converge
        block_size: 4,
        ..Default::default()
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let _result = solver.solve();

    // After one iteration, W_k should be populated
    // (Currently we break after first iteration, so this tests the W_k storage)
    // Once we implement the full loop, this will test 3m dimension
}

// ============================================================================
// SPD Toy Problem Tests - Eigensolver Convergence Verification
// ============================================================================

/// Test the eigensolver on a simple diagonal SPD operator.
///
/// This is a critical test: for a diagonal operator with eigenvalues 1, 2, 3, ..., N,
/// the eigensolver MUST converge to machine precision. The "eigenvectors" are just
/// the standard basis vectors e_i.
///
/// If residuals plateau before machine precision, the eigensolver has a bug.
#[test]
fn test_eigensolver_spd_diagonal_convergence() {
    // Create a diagonal operator with eigenvalues 1, 2, 3, ..., 16
    let eigenvalues: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let mut operator = DiagonalOperator::new(eigenvalues.clone());

    let n_bands = 8;
    let exact: Vec<f64> = eigenvalues[..n_bands].to_vec();

    // Configure eigensolver with tight tolerance
    let config = EigensolverConfig {
        n_bands,
        max_iter: 200,
        tol: 1e-10,
        block_size: 0, // Auto
        record_diagnostics: false,
        k_index: None,
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let result = solver.solve();

    // Print diagnostics regardless of outcome
    eprintln!("\n[SPD Diagonal Test]");
    eprintln!("  Iterations: {}", result.iterations);
    eprintln!("  Converged: {}", result.converged);
    eprintln!("  Max residual: {:.2e}", result.convergence.max_residual);
    eprintln!("  Computed eigenvalues: {:?}", result.eigenvalues);
    eprintln!("  Exact eigenvalues:    {:?}", exact);

    // The solver MUST converge on this SPD problem
    assert!(
        result.converged,
        "SPD diagonal problem MUST converge! iterations={}, max_residual={:.2e}\n\
        This indicates a bug in the eigensolver, not the physics operators.",
        result.iterations, result.convergence.max_residual
    );

    // Check that eigenvalues match the exact values
    for (i, (&computed, &exact_val)) in result.eigenvalues.iter().zip(exact.iter()).enumerate() {
        let rel_error: f64 = (computed - exact_val).abs() / exact_val.abs().max(1e-15);
        assert!(
            rel_error < 1e-6,
            "Eigenvalue {} mismatch: computed={:.10e}, exact={:.10e}, rel_error={:.2e}",
            i,
            computed,
            exact_val,
            rel_error
        );
    }
}

/// Test SPD convergence with diagnostics to see the convergence trajectory.
#[test]
fn test_eigensolver_spd_diagonal_convergence_with_diagnostics() {
    // Create a diagonal operator with eigenvalues 1, 2, 3, ..., 16
    let eigenvalues: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let mut operator = DiagonalOperator::new(eigenvalues.clone());

    let n_bands = 4;
    let exact: Vec<f64> = eigenvalues[..n_bands].to_vec();

    let config = EigensolverConfig {
        n_bands,
        max_iter: 100,
        tol: 1e-12, // Very tight
        block_size: 0,
        record_diagnostics: true,
        k_index: None,
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let diag_result = solver.solve_with_diagnostics("spd_diagonal_test");

    let result = &diag_result.result;
    let diagnostics = &diag_result.diagnostics;

    // Print detailed convergence history
    eprintln!("\n[SPD Diagonal Diagnostics]");
    eprintln!("  Iterations: {}", result.iterations);
    eprintln!("  Converged: {}", result.converged);
    eprintln!(
        "  Final max residual: {:.2e}",
        result.convergence.max_residual
    );
    eprintln!("  Computed eigenvalues: {:?}", result.eigenvalues);
    eprintln!("  Exact eigenvalues:    {:?}", exact);

    // Show per-iteration residual decline
    if !result.converged {
        eprintln!("\n  Convergence history (max relative residual):");
        for snapshot in &diagnostics.snapshots {
            if let Some(max_rel) = snapshot.relative_residuals.iter().cloned().reduce(f64::max) {
                eprintln!(
                    "    iter {:>3}: max_rel = {:.2e}",
                    snapshot.iteration + 1,
                    max_rel
                );
            }
        }

        eprintln!("\n  Subspace / SVQB diagnostics:");
        for snapshot in &diagnostics.snapshots {
            eprintln!(
                "    iter {:>3}: rank={:>2} dim_in={:>2} dropped={:>2} w_size={:>2} locked={:>2} active={:>2}",
                snapshot.iteration + 1,
                snapshot.subspace_rank_output,
                snapshot.subspace_dim_input,
                snapshot.svqb_dropped,
                snapshot.w_size,
                snapshot.n_locked,
                snapshot.n_active,
            );
        }
    }

    // Must converge
    assert!(
        result.converged,
        "SPD diagonal problem (diagnostics) MUST converge! iterations={}, max_residual={:.2e}",
        result.iterations, result.convergence.max_residual
    );
}

/// Test with a larger condition number to see if that affects convergence.
#[test]
fn test_eigensolver_spd_high_condition_number() {
    // Eigenvalues: 1, 10, 100, 1000, ... (condition number ~10^7)
    let eigenvalues: Vec<f64> = (0..16).map(|i| 10.0_f64.powi(i as i32 / 2)).collect();
    let mut operator = DiagonalOperator::new(eigenvalues.clone());

    let n_bands = 4;
    let exact: Vec<f64> = eigenvalues[..n_bands].to_vec();

    let config = EigensolverConfig {
        n_bands,
        max_iter: 500, // May need more iterations
        tol: 1e-8,     // Slightly relaxed due to conditioning
        block_size: 0,
        record_diagnostics: false,
        k_index: None,
    };

    let mut solver = Eigensolver::new(&mut operator, config, None, None);
    let result = solver.solve();

    let condition_number = eigenvalues.last().unwrap() / eigenvalues.first().unwrap();

    eprintln!("\n[SPD High Condition Number Test]");
    eprintln!("  Condition number: {:.2e}", condition_number);
    eprintln!("  Iterations: {}", result.iterations);
    eprintln!("  Converged: {}", result.converged);
    eprintln!("  Max residual: {:.2e}", result.convergence.max_residual);
    eprintln!("  Computed eigenvalues: {:?}", result.eigenvalues);
    eprintln!("  Exact eigenvalues:    {:?}", exact);

    // Even with high condition number, SPD should converge (maybe slower)
    assert!(
        result.converged,
        "SPD high-κ problem should converge! κ={:.2e}, iterations={}, max_residual={:.2e}",
        condition_number, result.iterations, result.convergence.max_residual
    );
}
