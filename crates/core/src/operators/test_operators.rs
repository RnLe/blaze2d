//! Test operators with known spectra for eigensolver validation.
//!
//! These operators are designed for testing and benchmarking the LOBPCG
//! eigensolver without the complexity of the full Maxwell operator.

use std::f64::consts::PI;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::grid::Grid2D;
use crate::operators::LinearOperator;

// ============================================================================
// ToyLaplacian
// ============================================================================

/// A simple periodic Laplacian operator for testing.
///
/// This operator implements -∇² on a periodic 2D domain using FFT.
/// The eigenvalues are |k|² = k_x² + k_y² for k on the reciprocal lattice.
///
/// # Usage
///
/// ```ignore
/// let lap = ToyLaplacian::new(backend, grid);
/// let mut output = lap.alloc_field();
/// lap.apply(&input, &mut output);
/// ```
pub struct ToyLaplacian<B: SpectralBackend> {
    backend: B,
    grid: Grid2D,
    kx: Vec<f64>,
    ky: Vec<f64>,
    scratch: B::Buffer,
}

impl<B: SpectralBackend> ToyLaplacian<B> {
    /// Create a new ToyLaplacian operator.
    pub fn new(backend: B, grid: Grid2D) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid must have non-zero dimensions"
        );
        assert!(
            grid.lx > 0.0 && grid.ly > 0.0,
            "grid lengths must be positive"
        );
        let kx = build_k_vector(grid.nx, grid.lx);
        let ky = build_k_vector(grid.ny, grid.ly);
        let scratch = backend.alloc_field(grid);
        Self {
            backend,
            grid,
            kx,
            ky,
            scratch,
        }
    }

    /// Get the computational grid.
    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    /// Allocate a new field buffer.
    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    /// Get a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a mutable reference to the backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ToyLaplacian<B> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let data = self.scratch.as_mut_slice();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iy * nx + ix;
                let k2 = self.kx[ix] * self.kx[ix] + self.ky[iy] * self.ky[iy];
                data[idx] *= k2;
            }
        }
        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(output, input);
    }

    fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    fn backend(&self) -> &B {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut B {
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
// ToyDiagonalSPD
// ============================================================================

/// A simple diagonal SPD operator with known eigenvalues for testing.
///
/// This operator has eigenvalues λ_k = 1 + k for k = 0, 1, 2, ... (N-1) in Fourier space,
/// where each Fourier mode is an exact eigenvector. This allows verification that
/// the eigensolver converges to machine precision on a well-conditioned problem.
///
/// The operator is:
/// - **Self-adjoint**: A = A^* (diagonal in Fourier space with real entries)
/// - **Positive definite**: All eigenvalues λ_k ≥ 1 > 0
/// - **Well-conditioned**: κ = λ_max / λ_min = N / 1 = N (linear in grid size)
///
/// The mass operator is the identity (B = I), so this is a standard eigenproblem.
///
/// # Expected eigenvalues
///
/// For an N×N grid, the smallest eigenvalues are:
/// - λ_0 = 1 (constant mode at index 0)
/// - λ_1 = 2, λ_2 = 3, ... (subsequent modes)
///
/// The eigenvectors are the standard Fourier basis functions.
pub struct ToyDiagonalSPD<B: SpectralBackend> {
    backend: B,
    grid: Grid2D,
    /// Eigenvalues λ_k = 1 + k for each Fourier mode (sorted by magnitude)
    eigenvalues: Vec<f64>,
    scratch: B::Buffer,
}

impl<B: SpectralBackend> ToyDiagonalSPD<B> {
    /// Create a new ToyDiagonalSPD operator.
    ///
    /// The eigenvalues are λ_k = 1 + k for k = 0, 1, 2, ..., N-1.
    pub fn new(backend: B, grid: Grid2D) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid must have non-zero dimensions"
        );

        let n = grid.len();
        let eigenvalues: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
        let scratch = backend.alloc_field(grid);

        Self {
            backend,
            grid,
            eigenvalues,
            scratch,
        }
    }

    /// Get the exact eigenvalues (for verification).
    pub fn exact_eigenvalues(&self, n_bands: usize) -> Vec<f64> {
        self.eigenvalues[..n_bands.min(self.eigenvalues.len())].to_vec()
    }

    /// Get the condition number κ = λ_max / λ_min.
    pub fn condition_number(&self) -> f64 {
        let n = self.eigenvalues.len();
        if n == 0 {
            return 1.0;
        }
        self.eigenvalues[n - 1] / self.eigenvalues[0]
    }

    /// Get the computational grid.
    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    /// Allocate a new field buffer.
    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    /// Get a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a mutable reference to the backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ToyDiagonalSPD<B> {
    /// Apply A: multiply each Fourier mode by its eigenvalue λ_k = 1 + k.
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        let data = self.scratch.as_mut_slice();
        for (k, value) in data.iter_mut().enumerate() {
            *value *= self.eigenvalues[k];
        }

        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    /// Mass operator is identity: B = I.
    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(output, input);
    }

    fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    fn backend(&self) -> &B {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut B {
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
// Helper Functions
// ============================================================================

fn build_k_vector(n: usize, length: f64) -> Vec<f64> {
    let two_pi = 2.0 * PI;
    (0..n)
        .map(|i| {
            let centered = if i <= n / 2 {
                i as isize
            } else {
                i as isize - n as isize
            };
            two_pi * centered as f64 / length
        })
        .collect()
}

fn copy_buffer<T: SpectralBuffer>(dst: &mut T, src: &T) {
    dst.as_mut_slice().copy_from_slice(src.as_slice());
}
