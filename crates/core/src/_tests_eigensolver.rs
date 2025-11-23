#![cfg(test)]

use super::backend::SpectralBackend;
use super::eigensolver::{
    EigenOptions, GammaContext, PowerIterationOptions, PreconditionerKind,
    build_deflation_workspace, power_iteration, solve_lowest_eigenpairs,
};
use super::field::Field2D;
use super::grid::Grid2D;
use super::operator::LinearOperator;
use super::symmetry::{Parity, ReflectionAxis, ReflectionConstraint};
use num_complex::Complex64;

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

struct DenseHermitianOp {
    backend: TestBackend,
    grid: Grid2D,
    matrix: Vec<f64>,
}

impl DenseHermitianOp {
    fn from_diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut matrix = vec![0.0; n * n];
        for (idx, &value) in diag.iter().enumerate() {
            matrix[idx * n + idx] = value;
        }
        Self {
            backend: TestBackend,
            grid: Grid2D::new(n, 1, 1.0, 1.0),
            matrix,
        }
    }

    fn with_entries(size: usize, entries: &[(usize, usize, f64)]) -> Self {
        let mut matrix = vec![0.0; size * size];
        for &(row, col, value) in entries {
            matrix[row * size + col] = value;
            matrix[col * size + row] = value;
        }
        Self {
            backend: TestBackend,
            grid: Grid2D::new(size, 1, 1.0, 1.0),
            matrix,
        }
    }

    fn size(&self) -> usize {
        self.grid.len()
    }
}

impl LinearOperator<TestBackend> for DenseHermitianOp {
    fn apply(&mut self, input: &Field2D, output: &mut Field2D) {
        let n = self.size();
        let in_data = input.as_slice();
        let out_data = output.as_mut_slice();
        for value in out_data.iter_mut() {
            *value = Complex64::default();
        }
        for row in 0..n {
            let mut accum = 0.0;
            for col in 0..n {
                let coeff = self.matrix[row * n + col];
                let contrib = coeff * in_data[col].re;
                accum += contrib;
            }
            out_data[row] = Complex64::new(accum, 0.0);
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
}

fn approx_eq(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "band {idx} mismatch: got {got}, expected {want}, diff {diff}"
        );
    }
}

#[test]
fn diagonal_operator_recovers_sorted_bands() {
    let diag = [0.25, 1.0, 4.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let opts = EigenOptions {
        n_bands: 3,
        max_iter: 32,
        tol: 1e-12,
        ..Default::default()
    };
    let result = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    approx_eq(&result.omegas, &[0.5, 1.0, 2.0], 1e-9);
    assert!(result.iterations <= opts.max_iter);
}

#[test]
fn degenerate_spectrum_preserves_duplicates() {
    let diag = [1.0, 1.0, 9.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let opts = EigenOptions {
        n_bands: 3,
        max_iter: 24,
        tol: 1e-12,
        ..Default::default()
    };
    let result = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    assert_eq!(
        result.omegas.len(),
        2,
        "degenerate modes collapse to unique bands"
    );
    approx_eq(&result.omegas, &[1.0, 3.0], 1e-9);
}

#[test]
fn negative_modes_are_filtered_out() {
    let diag = [-4.0, 1.0, 4.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let opts = EigenOptions {
        n_bands: 3,
        max_iter: 24,
        tol: 1e-10,
        ..Default::default()
    };
    let result = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    approx_eq(&result.omegas, &[1.0, 2.0], 1e-9);
}

#[test]
fn krylov_limit_caps_iterations_and_band_count() {
    let diag = [0.25, 1.0, 4.0, 9.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let opts = EigenOptions {
        n_bands: 10,
        max_iter: 5,
        tol: 1e-9,
        ..Default::default()
    };
    let result = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    assert!(result.iterations <= opts.max_iter);
    assert_eq!(result.omegas.len(), 4);
}

#[test]
fn off_diagonal_coupling_matches_expected_modes() {
    // 2x2 matrix with entries [[2,1],[1,3]] has eigenvalues 1.381966 and 3.618034.
    let mut op = DenseHermitianOp::with_entries(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
    let opts = EigenOptions {
        n_bands: 2,
        max_iter: 32,
        tol: 1e-12,
        ..Default::default()
    };
    let result = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    approx_eq(&result.omegas, &[1.17628, 1.90211], 1e-3);
}

#[test]
fn power_iteration_converges_to_dominant_eigenvalue() {
    let diag = [0.25, 9.0, 4.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let mut vector = op.alloc_field();
    {
        let data = vector.as_mut_slice();
        for (idx, value) in data.iter_mut().enumerate() {
            *value = Complex64::new((idx as f64 + 0.5).cos(), 0.0);
        }
    }
    let eig = power_iteration(
        &mut op,
        &mut vector,
        &PowerIterationOptions {
            max_iter: 256,
            tol: 1e-12,
        },
    );
    assert!(
        (eig - 9.0).abs() < 1e-6,
        "expected dominant eigenvalue near 9, got {eig}"
    );
}

#[test]
fn deflation_workspace_ignores_zero_norm_modes() {
    let mut op = DenseHermitianOp::from_diagonal(&[1.0]);
    let zero = Field2D::zeros(op.grid());
    let workspace = build_deflation_workspace::<_, _>(&mut op, [&zero]);
    assert_eq!(workspace.len(), 0, "zero vector should not enter workspace");
}

#[test]
fn deflation_prevents_refinding_lowest_mode() {
    let diag = [0.25, 1.0, 4.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let opts = EigenOptions {
        n_bands: 1,
        max_iter: 64,
        tol: 1e-12,
        ..Default::default()
    };
    let initial = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    assert_eq!(initial.omegas.len(), 1);
    let refs: Vec<&Field2D> = initial.modes.iter().collect();
    let workspace = build_deflation_workspace(&mut op, refs);
    assert!(
        workspace.len() >= 1,
        "deflation workspace should capture mode"
    );

    let mut second_opts = EigenOptions {
        n_bands: 2,
        max_iter: 64,
        tol: 1e-12,
        ..Default::default()
    };
    second_opts.deflation.enabled = true;
    let second = solve_lowest_eigenpairs(
        &mut op,
        &second_opts,
        None,
        GammaContext::default(),
        None,
        Some(&workspace),
        None,
    );
    assert_eq!(second.omegas.len(), 2);
    approx_eq(&second.omegas, &[1.0, 2.0], 1e-9);
}

#[test]
fn symmetry_constraints_enforce_odd_parity() {
    let diag = [0.25, 1.0, 4.0, 9.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let mut opts = EigenOptions {
        n_bands: 2,
        max_iter: 64,
        tol: 1e-12,
        ..Default::default()
    };
    opts.deflation.enabled = true;
    opts.symmetry.reflections = vec![ReflectionConstraint {
        axis: ReflectionAxis::X,
        parity: Parity::Odd,
    }];
    let result = solve_lowest_eigenpairs(
        &mut op,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    assert!(
        result.modes.len() >= 1,
        "captured modes should be available"
    );
    let mode = &result.modes[0];
    let grid = mode.grid();
    for ix in 0..grid.nx {
        let idx = grid.idx(ix, 0);
        let mirror_idx = grid.idx((grid.nx - ix) % grid.nx, 0);
        let value = mode.as_slice()[idx] + mode.as_slice()[mirror_idx];
        assert!(
            value.norm() < 1e-9,
            "odd parity should enforce antisymmetry for column {ix}"
        );
    }
}

#[test]
fn preconditioner_defaults_to_structured() {
    assert_eq!(
        PreconditionerKind::default(),
        PreconditionerKind::StructuredDiagonal,
        "runs should use the structured preconditioner by default"
    );
}

#[test]
fn warm_start_reuses_previous_modes() {
    let diag = [0.25, 1.0, 4.0];
    let mut op = DenseHermitianOp::from_diagonal(&diag);
    let base_opts = EigenOptions {
        n_bands: 3,
        max_iter: 32,
        tol: 1e-10,
        ..Default::default()
    };
    let initial = solve_lowest_eigenpairs(
        &mut op,
        &base_opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    assert_eq!(initial.omegas.len(), 3);
    let seeds = initial.modes.clone();

    let mut reuse_op = DenseHermitianOp::from_diagonal(&diag);
    let mut reuse_opts = base_opts.clone();
    reuse_opts.max_iter = 0;
    let reuse = solve_lowest_eigenpairs(
        &mut reuse_op,
        &reuse_opts,
        None,
        GammaContext::default(),
        Some(seeds.as_slice()),
        None,
        None,
    );
    approx_eq(&reuse.omegas, &[0.5, 1.0, 2.0], 1e-9);
    assert_eq!(reuse.iterations, 0, "warm starts should bypass iterations");
}
