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

pub(crate) const K_PLUS_G_NEAR_ZERO_FLOOR: f64 = 1e-9;
pub(crate) const STRUCTURED_WEIGHT_MIN: f64 = 1e-3;
pub(crate) const STRUCTURED_WEIGHT_MAX: f64 = 1e3;

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
    bloch: [f64; 2],
    kx_shifted: Vec<f64>,
    ky_shifted: Vec<f64>,
    k_plus_g_x: Vec<f64>,
    k_plus_g_y: Vec<f64>,
    k_plus_g_sq: Vec<f64>,
    #[allow(dead_code)]
    k_plus_g_sq_min: f64,
    #[allow(dead_code)]
    k_plus_g_sq_min_raw: f64,
    #[allow(dead_code)]
    k_plus_g_floor_count: usize,
    scratch: B::Buffer,
    grad_x: B::Buffer,
    grad_y: B::Buffer,
    k_plus_g_was_clamped: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct OperatorSnapshotData {
    pub grid: Grid2D,
    pub field_spatial: Vec<Complex64>,
    pub field_fourier: Vec<Complex64>,
    pub theta_spatial: Vec<Complex64>,
    pub theta_fourier: Vec<Complex64>,
    pub grad_x: Option<Vec<Complex64>>,
    pub grad_y: Option<Vec<Complex64>>,
    pub eps_grad_x: Option<Vec<Complex64>>,
    pub eps_grad_y: Option<Vec<Complex64>>,
}

impl OperatorSnapshotData {
    pub fn len(&self) -> usize {
        self.grid.len()
    }
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
        let (
            k_plus_g_x,
            k_plus_g_y,
            k_plus_g_sq,
            k_plus_g_sq_min_raw,
            k_plus_g_sq_min,
            k_plus_g_floor_count,
            k_plus_g_was_clamped,
        ) = build_k_plus_g_tables(&kx_shifted, &ky_shifted, grid);
        let scratch = backend.alloc_field(grid);
        let grad_x = backend.alloc_field(grid);
        let grad_y = backend.alloc_field(grid);
        Self {
            backend,
            dielectric,
            polarization,
            grid,
            bloch: bloch_k,
            kx_shifted,
            ky_shifted,
            k_plus_g_x,
            k_plus_g_y,
            k_plus_g_sq,
            k_plus_g_sq_min,
            k_plus_g_sq_min_raw,
            k_plus_g_floor_count,
            scratch,
            grad_x,
            grad_y,
            k_plus_g_was_clamped,
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

    pub fn build_homogeneous_preconditioner(&self) -> FourierDiagonalPreconditioner {
        let shift = FOURIER_DIAGONAL_SHIFT;
        let inverse_diagonal = match self.polarization {
            Polarization::TE => {
                let eps_eff = self.effective_te_epsilon();
                build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff)
            }
            Polarization::TM => {
                let eps_eff = self.effective_tm_epsilon();
                build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff)
            }
        };
        FourierDiagonalPreconditioner::new(inverse_diagonal)
    }

    pub fn build_structured_preconditioner(&self) -> FourierDiagonalPreconditioner {
        let shift = FOURIER_DIAGONAL_SHIFT;
        match self.polarization {
            Polarization::TE => {
                let eps_eff = self.effective_te_epsilon();
                let inverse_diagonal = build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff);
                let weights = build_structured_weights_te(&self.dielectric, eps_eff);
                FourierDiagonalPreconditioner::with_weights(inverse_diagonal, weights)
            }
            Polarization::TM => {
                let eps_eff = self.effective_tm_epsilon();
                let inverse_diagonal = build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff);
                let weights = build_structured_weights_tm(&self.dielectric);
                FourierDiagonalPreconditioner::with_weights(inverse_diagonal, weights)
            }
        }
    }

    #[allow(dead_code)]
    pub fn build_fourier_diagonal_preconditioner(&self) -> FourierDiagonalPreconditioner {
        self.build_homogeneous_preconditioner()
    }

    pub(crate) fn bloch(&self) -> [f64; 2] {
        self.bloch
    }

    pub(crate) fn kx_shifted(&self) -> &[f64] {
        &self.kx_shifted
    }

    pub(crate) fn ky_shifted(&self) -> &[f64] {
        &self.ky_shifted
    }

    pub(crate) fn k_plus_g_squares(&self) -> &[f64] {
        &self.k_plus_g_sq
    }

    pub(crate) fn k_plus_g_components(&self) -> (&[f64], &[f64]) {
        (&self.k_plus_g_x, &self.k_plus_g_y)
    }

    pub(crate) fn k_plus_g_clamp_mask(&self) -> &[bool] {
        &self.k_plus_g_was_clamped
    }

    #[allow(dead_code)]
    pub(crate) fn k_plus_g_sq_min(&self) -> f64 {
        self.k_plus_g_sq_min
    }

    #[allow(dead_code)]
    pub(crate) fn k_plus_g_sq_min_raw(&self) -> f64 {
        self.k_plus_g_sq_min_raw
    }

    #[allow(dead_code)]
    pub(crate) fn k_plus_g_near_zero_count(&self) -> usize {
        self.k_plus_g_floor_count
    }

    pub(crate) fn capture_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        match self.polarization {
            Polarization::TM => self.capture_tm_snapshot(input),
            Polarization::TE => self.capture_te_snapshot(input),
        }
    }

    fn apply_tm(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);

        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.dielectric,
        );

        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            self.scratch.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
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

    fn capture_tm_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        let field_spatial = input.as_slice().to_vec();
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let field_fourier = self.scratch.as_slice().to_vec();

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);
        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);
        let grad_x = self.grad_x.as_slice().to_vec();
        let grad_y = self.grad_y.as_slice().to_vec();

        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.dielectric,
        );
        let eps_grad_x = self.grad_x.as_slice().to_vec();
        let eps_grad_y = self.grad_y.as_slice().to_vec();

        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            self.scratch.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );
        let theta_fourier = self.scratch.as_slice().to_vec();

        self.backend.inverse_fft_2d(&mut self.scratch);
        let theta_spatial = self.scratch.as_slice().to_vec();

        OperatorSnapshotData {
            grid: self.grid,
            field_spatial,
            field_fourier,
            theta_spatial,
            theta_fourier,
            grad_x: Some(grad_x),
            grad_y: Some(grad_y),
            eps_grad_x: Some(eps_grad_x),
            eps_grad_y: Some(eps_grad_y),
        }
    }

    fn capture_te_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        let field_spatial = input.as_slice().to_vec();
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let field_fourier = self.scratch.as_slice().to_vec();

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);
        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);
        let grad_x = self.grad_x.as_slice().to_vec();
        let grad_y = self.grad_y.as_slice().to_vec();

        for (value, &k_sq) in self
            .scratch
            .as_mut_slice()
            .iter_mut()
            .zip(self.k_plus_g_sq.iter())
        {
            *value *= k_sq;
        }
        let theta_fourier = self.scratch.as_slice().to_vec();

        self.backend.inverse_fft_2d(&mut self.scratch);
        let theta_spatial = self.scratch.as_slice().to_vec();

        OperatorSnapshotData {
            grid: self.grid,
            field_spatial,
            field_fourier,
            theta_spatial,
            theta_fourier,
            grad_x: Some(grad_x),
            grad_y: Some(grad_y),
            eps_grad_x: None,
            eps_grad_y: None,
        }
    }
}

impl<B: SpectralBackend> ThetaOperator<B> {
    fn effective_te_epsilon(&self) -> f64 {
        harmonic_mean(self.dielectric.inv_eps()).unwrap_or(1.0)
    }

    fn effective_tm_epsilon(&self) -> f64 {
        arithmetic_mean(self.dielectric.eps()).unwrap_or(1.0)
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

fn build_k_plus_g_tables(
    kx_shifted: &[f64],
    ky_shifted: &[f64],
    grid: Grid2D,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, usize, Vec<bool>) {
    let nx = grid.nx;
    let ny = grid.ny;
    let len = grid.len();
    let mut k_plus_g_x = vec![0.0; len];
    let mut k_plus_g_y = vec![0.0; len];
    let mut squares = vec![0.0; len];
    let mut clamp_mask = vec![false; len];
    let mut raw_min = f64::INFINITY;
    let mut clamped_min = f64::INFINITY;
    let mut floor_count = 0usize;
    for iy in 0..ny {
        let raw_ky = ky_shifted[iy];
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let raw_kx = kx_shifted[ix];
            let raw_sq = raw_kx * raw_kx + raw_ky * raw_ky;
            if raw_sq.is_finite() {
                raw_min = raw_min.min(raw_sq);
            }
            let (clamped_kx, clamped_ky) = clamp_gradient_components(raw_kx, raw_ky);
            let clamped_sq = clamped_kx * clamped_kx + clamped_ky * clamped_ky;
            clamped_min = clamped_min.min(clamped_sq);
            if raw_sq <= K_PLUS_G_NEAR_ZERO_FLOOR {
                floor_count += 1;
                clamp_mask[idx] = true;
            }
            k_plus_g_x[idx] = clamped_kx;
            k_plus_g_y[idx] = clamped_ky;
            squares[idx] = clamped_sq;
        }
    }
    if raw_min == f64::INFINITY {
        raw_min = 0.0;
    }
    if clamped_min == f64::INFINITY {
        clamped_min = 0.0;
    }
    (
        k_plus_g_x,
        k_plus_g_y,
        squares,
        raw_min,
        clamped_min,
        floor_count,
        clamp_mask,
    )
}

fn inverse_scale(k_sq: f64, shift: f64, eps_eff: f64) -> f64 {
    if !k_sq.is_finite() || !eps_eff.is_finite() || eps_eff <= 0.0 {
        return 0.0;
    }

    let safe_k_sq = k_sq.max(K_PLUS_G_NEAR_ZERO_FLOOR);
    let shift_scaled = shift * eps_eff.max(1e-12);
    eps_eff / (safe_k_sq + shift_scaled)
}

fn copy_buffer<T: SpectralBuffer>(dst: &mut T, src: &T) {
    dst.as_mut_slice().copy_from_slice(src.as_slice());
}

fn apply_gradient_factors(
    grad_x: &mut [Complex64],
    grad_y: &mut [Complex64],
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
) {
    for (((gx, gy), &kx), &ky) in grad_x
        .iter_mut()
        .zip(grad_y.iter_mut())
        .zip(k_plus_g_x.iter())
        .zip(k_plus_g_y.iter())
    {
        let factor_x = Complex64::new(0.0, kx);
        let factor_y = Complex64::new(0.0, ky);
        *gx *= factor_x;
        *gy *= factor_y;
    }
}

fn apply_inv_eps(grad_x: &mut [Complex64], grad_y: &mut [Complex64], dielectric: &Dielectric2D) {
    if let Some(tensors) = dielectric.inv_eps_tensors() {
        for ((gx, gy), tensor) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(tensors.iter()) {
            let orig_x = *gx;
            let orig_y = *gy;
            let out_x = orig_x * tensor[0] + orig_y * tensor[1];
            let out_y = orig_x * tensor[2] + orig_y * tensor[3];
            *gx = out_x;
            *gy = out_y;
        }
    } else {
        for ((gx, gy), &inv) in grad_x
            .iter_mut()
            .zip(grad_y.iter_mut())
            .zip(dielectric.inv_eps().iter())
        {
            *gx *= inv;
            *gy *= inv;
        }
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
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
) {
    for ((((out, &gx), &gy), &kx), &ky) in output
        .iter_mut()
        .zip(grad_x.iter())
        .zip(grad_y.iter())
        .zip(k_plus_g_x.iter())
        .zip(k_plus_g_y.iter())
    {
        let factor_x = Complex64::new(0.0, kx);
        let factor_y = Complex64::new(0.0, ky);
        let div = factor_x * gx + factor_y * gy;
        *out = -div;
    }
}

#[cfg(test)]
mod tests {
    use super::{K_PLUS_G_NEAR_ZERO_FLOOR, clamp_gradient_components, inverse_scale};

    #[test]
    fn clamp_gradient_handles_zero_and_nan() {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        let (x_zero, y_zero) = clamp_gradient_components(0.0, 0.0);
        assert_eq!(x_zero, magnitude);
        assert_eq!(y_zero, 0.0);

        let (x_nan, y_nan) = clamp_gradient_components(f64::NAN, f64::NAN);
        assert_eq!(x_nan, magnitude);
        assert_eq!(y_nan, 0.0);
    }

    #[test]
    fn inverse_scale_sanitizes_non_finite_and_underflow() {
        assert_eq!(inverse_scale(f64::NAN, 1e-3, 1.0), 0.0);
        assert_eq!(inverse_scale(1.0, 1e-3, f64::NAN), 0.0);

        let tiny = K_PLUS_G_NEAR_ZERO_FLOOR / 10.0;
        let expected = 1.0 / (K_PLUS_G_NEAR_ZERO_FLOOR + 1e-3);
        let actual = inverse_scale(tiny, 1e-3, 1.0);
        assert!((actual - expected).abs() < 1e-12);
    }
}

fn shift_k_vector(base: &[f64], shift: f64) -> Vec<f64> {
    base.iter().map(|&k| k + shift).collect()
}

fn clamp_gradient_components(kx: f64, ky: f64) -> (f64, f64) {
    if !kx.is_finite() || !ky.is_finite() {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        return (magnitude, 0.0);
    }

    let norm_sq = kx * kx + ky * ky;
    if norm_sq >= K_PLUS_G_NEAR_ZERO_FLOOR {
        (kx, ky)
    } else if norm_sq == 0.0 {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        (magnitude, 0.0)
    } else {
        let scale = (K_PLUS_G_NEAR_ZERO_FLOOR / norm_sq).sqrt();
        (kx * scale, ky * scale)
    }
}

fn build_inverse_diagonal(values: &[f64], shift: f64, eps_eff: f64) -> Vec<f64> {
    values
        .iter()
        .copied()
        .map(|k| inverse_scale(k, shift, eps_eff))
        .collect()
}

fn build_structured_weights_te(dielectric: &Dielectric2D, eps_eff: f64) -> Vec<f64> {
    let eps = dielectric.eps();
    if eps.is_empty() || eps_eff <= 0.0 {
        return vec![1.0; dielectric.grid.len()];
    }
    eps.iter().map(|&val| clamp_weight(val / eps_eff)).collect()
}

fn build_structured_weights_tm(dielectric: &Dielectric2D) -> Vec<f64> {
    let inv_eps = dielectric.inv_eps();
    if inv_eps.is_empty() {
        return vec![1.0; dielectric.grid.len()];
    }
    let avg_inv = arithmetic_mean(inv_eps).unwrap_or(0.0);
    if avg_inv <= 0.0 {
        return vec![1.0; dielectric.grid.len()];
    }
    inv_eps
        .iter()
        .map(|&val| clamp_weight(val / avg_inv))
        .collect()
}

fn clamp_weight(value: f64) -> f64 {
    value.max(STRUCTURED_WEIGHT_MIN).min(STRUCTURED_WEIGHT_MAX)
}

fn arithmetic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum: f64 = values.iter().copied().sum();
    Some(sum / values.len() as f64)
}

fn harmonic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum: f64 = values.iter().copied().sum();
    if sum <= 0.0 {
        return None;
    }
    Some(1.0 / (sum / values.len() as f64))
}
