//! Operator implementations (toy Laplacian + physical Θ operator).

use num_complex::Complex64;

use std::f64::consts::PI;

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    grid::Grid2D,
    polarization::Polarization,
    preconditioner::RealSpaceJacobi,
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
    bloch_k: [f64; 2],
    grid: Grid2D,
    kx: Vec<f64>,
    ky: Vec<f64>,
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
        let scratch = backend.alloc_field(grid);
        let grad_x = backend.alloc_field(grid);
        let grad_y = backend.alloc_field(grid);
        Self {
            backend,
            dielectric,
            polarization,
            bloch_k,
            grid,
            kx,
            ky,
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

    pub fn build_real_space_jacobi_preconditioner(&self) -> RealSpaceJacobi {
        RealSpaceJacobi::from_dielectric(&self.dielectric)
    }

    fn apply_tm(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);

        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.kx,
            &self.ky,
            self.bloch_k,
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
            &self.kx,
            &self.ky,
            self.bloch_k,
        );

        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    fn apply_te(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let data = self.scratch.as_mut_slice();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for iy in 0..ny {
            let ky_shift = self.ky[iy] + self.bloch_k[1];
            let ky_sq = ky_shift * ky_shift;
            for ix in 0..nx {
                let idx = iy * nx + ix;
                let kx_shift = self.kx[ix] + self.bloch_k[0];
                let k_sq = kx_shift * kx_shift + ky_sq;
                data[idx] *= k_sq;
            }
        }
        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
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

fn copy_buffer<T: SpectralBuffer>(dst: &mut T, src: &T) {
    dst.as_mut_slice().copy_from_slice(src.as_slice());
}

fn apply_gradient_factors(
    grad_x: &mut [Complex64],
    grad_y: &mut [Complex64],
    kx: &[f64],
    ky: &[f64],
    bloch_k: [f64; 2],
) {
    let nx = kx.len();
    let ny = ky.len();
    for iy in 0..ny {
        let ky_shift = ky[iy] + bloch_k[1];
        let factor_y = Complex64::new(0.0, ky_shift);
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let kx_shift = kx[ix] + bloch_k[0];
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
    kx: &[f64],
    ky: &[f64],
    bloch_k: [f64; 2],
) {
    let nx = kx.len();
    let ny = ky.len();
    for iy in 0..ny {
        let ky_shift = ky[iy] + bloch_k[1];
        let factor_y = Complex64::new(0.0, ky_shift);
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let kx_shift = kx[ix] + bloch_k[0];
            let factor_x = Complex64::new(0.0, kx_shift);
            let div = factor_x * grad_x[idx] + factor_y * grad_y[idx];
            output[idx] = -div;
        }
    }
}
