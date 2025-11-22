//! Operator implementations (toy Laplacian + physical Θ operator).

use num_complex::Complex64;

use std::f64::consts::PI;

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    grid::Grid2D,
    polarization::Polarization,
    preconditioner::{FOURIER_DIAGONAL_SHIFT, FourierDiagonalPreconditioner},
};

pub trait LinearOperator<B: SpectralBackend> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer);
    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer);
    fn alloc_field(&self) -> B::Buffer;
    fn backend(&self) -> &B;
    fn backend_mut(&mut self) -> &mut B;
    fn grid(&self) -> Grid2D;
}

pub struct ThetaOperator<B: SpectralBackend> {
    backend: B,
    dielectric: Dielectric2D,
    polarization: Polarization,
    grid: Grid2D,
    kx_shifted: Vec<f64>,
    ky_shifted: Vec<f64>,
    k_plus_g_sq: Vec<f64>,
    scratch: B::Buffer,
    grad_x: B::Buffer,
    grad_y: B::Buffer,
}

impl<B: SpectralBackend> ThetaOperator<B> {
    pub fn new(
        backend: B,
        dielectric: Dielectric2D,
        polarization: Polarization,
        bloch_k: [f64; 2],
    ) -> Self {
        let grid = dielectric.grid;
        let kx = build_k_vector(grid.nx, grid.lx);
        let ky = build_k_vector(grid.ny, grid.ly);
        let kx_shifted = shift_k_vector(&kx, bloch_k[0]);
        let ky_shifted = shift_k_vector(&ky, bloch_k[1]);
        let k_plus_g_sq = build_k_plus_g_squares(&kx_shifted, &ky_shifted, grid);
        let scratch = backend.alloc_field(grid);
        let grad_x = backend.alloc_field(grid);
        let grad_y = backend.alloc_field(grid);
        Self {
            backend,
            dielectric,
            polarization,
            grid,
            kx_shifted,
            ky_shifted,
            k_plus_g_sq,
            scratch,
            grad_x,
            grad_y,
        }
    }

    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn build_fourier_diagonal_preconditioner(&self) -> FourierDiagonalPreconditioner {
        let shift = FOURIER_DIAGONAL_SHIFT;
        let inverse_diagonal = match self.polarization {
            Polarization::TE => self
                .k_plus_g_sq
                .iter()
                .copied()
                .map(|k| 1.0 / (k + shift))
                .collect(),
            Polarization::TM => {
                let eps_eff = self.effective_tm_epsilon();
                let denom = eps_eff.max(1e-12);
                self.k_plus_g_sq
                    .iter()
                    .copied()
                    .map(|k| 1.0 / (k / denom + shift))
                    .collect()
            }
        };
        FourierDiagonalPreconditioner::new(inverse_diagonal)
    }

    fn apply_tm(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);

        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.kx_shifted,
            &self.ky_shifted,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            self.dielectric.inv_eps(),
        );

        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            self.scratch.as_mut_slice(),
            &self.kx_shifted,
            &self.ky_shifted,
        );

        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    fn apply_te(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let data = self.scratch.as_mut_slice();
        for (value, &k_sq) in data.iter_mut().zip(self.k_plus_g_sq.iter()) {
            *value *= k_sq;
        }
        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }
}

impl<B: SpectralBackend> ThetaOperator<B> {
    fn effective_tm_epsilon(&self) -> f64 {
        let inv_eps = self.dielectric.inv_eps();
        if inv_eps.is_empty() {
            return 1.0;
        }
        let avg_inv = inv_eps.iter().copied().sum::<f64>() / inv_eps.len() as f64;
        if avg_inv <= 0.0 { 1.0 } else { 1.0 / avg_inv }
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ThetaOperator<B> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        match self.polarization {
            Polarization::TM => self.apply_tm(input, output),
            Polarization::TE => self.apply_te(input, output),
        }
    }

    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        match self.polarization {
            Polarization::TM => copy_buffer(output, input),
            Polarization::TE => {
                copy_buffer(output, input);
                apply_scalar_eps(output.as_mut_slice(), self.dielectric.eps());
            }
        }
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
}

pub struct ToyLaplacian<B: SpectralBackend> {
    backend: B,
    grid: Grid2D,
    kx: Vec<f64>,
    ky: Vec<f64>,
    scratch: B::Buffer,
}

impl<B: SpectralBackend> ToyLaplacian<B> {
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

    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

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
}

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

fn build_k_plus_g_squares(kx_shifted: &[f64], ky_shifted: &[f64], grid: Grid2D) -> Vec<f64> {
    let nx = grid.nx;
    let ny = grid.ny;
    let mut values = vec![0.0; grid.len()];
    for iy in 0..ny {
        let ky_shift = ky_shifted[iy];
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let kx_shift = kx_shifted[ix];
            values[idx] = kx_shift * kx_shift + ky_shift * ky_shift;
        }
    }
    values
}

fn copy_buffer<T: SpectralBuffer>(dst: &mut T, src: &T) {
    dst.as_mut_slice().copy_from_slice(src.as_slice());
}

fn apply_gradient_factors(
    grad_x: &mut [Complex64],
    grad_y: &mut [Complex64],
    kx_shifted: &[f64],
    ky_shifted: &[f64],
) {
    let nx = kx_shifted.len();
    let ny = ky_shifted.len();
    for iy in 0..ny {
        let ky_shift = ky_shifted[iy];
        let factor_y = Complex64::new(0.0, ky_shift);
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let kx_shift = kx_shifted[ix];
            let factor_x = Complex64::new(0.0, kx_shift);
            grad_x[idx] *= factor_x;
            grad_y[idx] *= factor_y;
        }
    }
}

fn apply_inv_eps(grad_x: &mut [Complex64], grad_y: &mut [Complex64], inv_eps: &[f64]) {
    for ((gx, gy), &inv) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(inv_eps.iter()) {
        *gx *= inv;
        *gy *= inv;
    }
}

fn apply_scalar_eps(field: &mut [Complex64], eps: &[f64]) {
    for (value, &eps_val) in field.iter_mut().zip(eps.iter()) {
        *value *= eps_val;
    }
}

fn assemble_divergence(
    grad_x: &[Complex64],
    grad_y: &[Complex64],
    output: &mut [Complex64],
    kx_shifted: &[f64],
    ky_shifted: &[f64],
) {
    let nx = kx_shifted.len();
    let ny = ky_shifted.len();
    for iy in 0..ny {
        let ky_shift = ky_shifted[iy];
        let factor_y = Complex64::new(0.0, ky_shift);
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let kx_shift = kx_shifted[ix];
            let factor_x = Complex64::new(0.0, kx_shift);
            let div = factor_x * grad_x[idx] + factor_y * grad_y[idx];
            output[idx] = -div;
        }
    }
}

fn shift_k_vector(base: &[f64], shift: f64) -> Vec<f64> {
    base.iter().map(|&k| k + shift).collect()
}
