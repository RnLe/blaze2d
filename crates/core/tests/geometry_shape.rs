use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::{SpectralBackend, SpectralBuffer},
    field::{Field2D, Real},
    geometry_warm_start::{
        shape::{self, Interpolation, Map, Options, Report},
        snapshot::FiberKey,
    },
    grid::Grid2D,
    lattice::Lattice2D,
};
use num_complex::{Complex, Complex64 as C};
use std::f64::consts::TAU;

fn key(grid: Grid2D, lattice: Lattice2D, k: [f64; 2]) -> FiberKey {
    let b = lattice.reciprocal();
    FiberKey {
        shape: [grid.nx, grid.ny],
        centered: grid.centered,
        storage_bits: 64,
        coordinate_bits: [
            grid.lx, grid.ly, b.b1[0], b.b1[1], b.b2[0], b.b2[1], k[0], k[1], 0.,
        ]
        .map(f64::to_bits),
    }
}
fn wave<B: SpectralBackend>(
    backend: B,
    grid: Grid2D,
    lattice: Lattice2D,
    target: &[f64],
    options: Options,
    tolerance: f64,
) -> f64 {
    wave_at(
        backend,
        grid,
        lattice,
        target,
        options,
        [2., -1.],
        tolerance,
    )
}
fn wave_at<B: SpectralBackend>(
    backend: B,
    grid: Grid2D,
    lattice: Lattice2D,
    target: &[f64],
    options: Options,
    frequency: [f64; 2],
    tolerance: f64,
) -> f64 {
    let k = lattice.reciprocal().fractional_to_cartesian([0.17, 0.13]);
    let map = Map::new(
        &key(grid, lattice, k),
        &[1., 0.49, 0.07, 0.16, 8.9],
        target,
        options,
    )
    .unwrap();
    let at = |f: [f64; 2]| C::from_polar(1., TAU * (frequency[0] * f[0] + frequency[1] * f[1]));
    let mut field = Field2D::<B::Real>::zeros(grid);
    for (i, v) in field.as_mut_slice().iter_mut().enumerate() {
        let p = grid.cartesian_coords(i % grid.nx, i / grid.nx);
        let z = at([p[0] / grid.lx, p[1] / grid.ly]);
        *v = Complex::new(B::Real::from_accum(z.re), B::Real::from_accum(z.im));
    }
    let mut report = Report::default();
    let output = shape::transport(
        &backend,
        grid,
        &[field],
        &map,
        options,
        1 << 24,
        &mut report,
    )
    .unwrap();
    let mut square = 0.;
    for (i, v) in output[0].as_slice().iter().enumerate() {
        let x = grid.cartesian_coords(i % grid.nx, i / grid.nx);
        let s = map.sample([x[0] / grid.lx, x[1] / grid.ly]);
        let expected = at(s.source)
            * C::from_polar(
                if options.jacobian {
                    s.determinant.sqrt()
                } else {
                    1.
                },
                k[0] * s.displacement[0] + k[1] * s.displacement[1],
            );
        square += (C::new(v.re.to_accum(), v.im.to_accum()) - expected).norm_sqr();
    }
    let error = (square / grid.len() as f64).sqrt();
    assert!(error < tolerance, "{error}: {report:?}");
    error
}
#[test]
fn rigid_pullbacks_include_bloch_phase_across_periodic_boundaries_in_both_precisions() {
    for lattice in [Lattice2D::square(1.3), Lattice2D::hexagonal(1.3)] {
        for centered in [false, true] {
            let mut grid = Grid2D::new(16, 18, 1.2, 0.9);
            grid.centered = centered;
            for shift in [1. / 16., 0.0237] {
                let target = [1., 0.49 + shift, 0.07, 0.16, 8.9];
                let nyquist = [grid.nx as f64 / 2., grid.ny as f64 / 2.];
                wave_at(
                    CpuBackend::<f64>::new(),
                    grid,
                    lattice,
                    &target,
                    Options::default(),
                    nyquist,
                    1e-12,
                );
                wave_at(
                    CpuBackend::<f32>::new(),
                    grid,
                    lattice,
                    &target,
                    Options::default(),
                    nyquist,
                    2e-6,
                );
                wave(
                    CpuBackend::<f64>::new(),
                    grid,
                    lattice,
                    &target,
                    Options::default(),
                    2e-14,
                );
                wave(
                    CpuBackend::<f32>::new(),
                    grid,
                    lattice,
                    &target,
                    Options::default(),
                    1e-6,
                );
            }
        }
    }
}
#[test]
fn local_inverse_map_matches_interfaces_and_has_a_periodic_positive_jacobian() {
    let lattice = Lattice2D::hexagonal(1.3);
    let grid = Grid2D::new(16, 16, 1., 1.);
    let source = [1., 0.49, 0.07, 0.16, 8.9];
    let target = [1., -0.508, 0.071, 0.161, 8.9];
    let map = Map::new(
        &key(grid, lattice, [0., 0.]),
        &source,
        &target,
        Options::default(),
    )
    .unwrap();
    for i in 0..24 {
        let angle = TAU * i as f64 / 24.;
        let radius = target[3] * lattice.characteristic_length();
        let delta = lattice.cartesian_to_fractional([radius * angle.cos(), radius * angle.sin()]);
        let x = [target[1] + delta[0], target[2] + delta[1]];
        let s = map.sample(x);
        let actual = lattice.fractional_to_cartesian([
            (s.source[0] - source[1] + 0.5).rem_euclid(1.) - 0.5,
            (s.source[1] - source[2] + 0.5).rem_euclid(1.) - 0.5,
        ]);
        assert!(
            (actual[0].hypot(actual[1]) - source[3] * lattice.characteristic_length()).abs()
                < 1e-13
        );
    }
    for i in 0..200 {
        let x = [i as f64 / 137., i as f64 / 211.];
        let s = map.sample(x);
        let periodic = map.sample([x[0] + 1., x[1] - 2.]);
        assert!((s.determinant - periodic.determinant).abs() < 1e-13);
        assert!(s.determinant >= (1. - map.derivative_bound).powi(2));
        assert!(s.determinant <= (1. + map.derivative_bound).powi(2));
        let h = 1e-6;
        let mut columns = [[0.; 2]; 2];
        for axis in 0..2 {
            let mut d = [0.; 2];
            d[axis] = h;
            let df = lattice.cartesian_to_fractional(d);
            let plus = map.sample([x[0] + df[0], x[1] + df[1]]);
            let minus = map.sample([x[0] - df[0], x[1] - df[1]]);
            for row in 0..2 {
                columns[axis][row] = f64::from(axis == row)
                    + (plus.displacement[row] - minus.displacement[row]) / (2. * h);
            }
        }
        let det = columns[0][0] * columns[1][1] - columns[1][0] * columns[0][1];
        assert!((det - s.determinant).abs() < 1e-8);
    }
    let bad = [1., 0.49, 0.07, 0.25, 8.9];
    assert!(
        Map::new(
            &key(grid, lattice, [0., 0.]),
            &source,
            &bad,
            Options::default()
        )
        .is_err()
    );
    let overlapping = [1., 0., 0., 0.2, 8.9, 0.1, 0., 0.15, 3.];
    let mut changed = overlapping;
    changed[3] += 0.001;
    assert!(
        Map::new(
            &key(grid, lattice, [0., 0.]),
            &overlapping,
            &changed,
            Options::default()
        )
        .is_err()
    );
}
#[test]
fn cubic_refines_toward_fourier_pullback_and_work_limits_precede_allocation() {
    for lattice in [Lattice2D::square(1.), Lattice2D::hexagonal(1.)] {
        let target = [1., 0.492, 0.071, 0.161, 8.9];
        let mut errors = Vec::new();
        for n in [16, 32, 64] {
            let grid = Grid2D::new_centered(n, n, 1., 1.);
            let options = Options {
                jacobian: true,
                ..Default::default()
            };
            errors.push(wave(
                CpuBackend::<f64>::new(),
                grid,
                lattice,
                &target,
                options,
                0.01,
            ));
            if n == 16 {
                wave(
                    CpuBackend::<f64>::new(),
                    grid,
                    lattice,
                    &target,
                    Options {
                        interpolation: Interpolation::Fourier,
                        ..options
                    },
                    2e-14,
                );
            }
        }
        eprintln!("cubic pullback refinement: {errors:?}");
        assert!(errors[2] < errors[1] / 3. && errors[1] < errors[0] / 3.);
        let grid = Grid2D::new(16, 16, 1., 1.);
        let options = Options {
            max_interpolation_terms: 10,
            ..Default::default()
        };
        let map = Map::new(
            &key(grid, lattice, [0., 0.]),
            &[1., 0.49, 0.07, 0.16, 8.9],
            &target,
            options,
        )
        .unwrap();
        let mut report = Report::default();
        assert!(
            shape::transport(
                &CpuBackend::<f64>::new(),
                grid,
                &[Field2D::zeros(grid)],
                &map,
                options,
                1 << 20,
                &mut report
            )
            .unwrap_err()
            .contains("work budget")
        );
        assert_eq!(report.interpolation_terms, 0);
        assert_eq!(report.forward_ffts, 0);
    }
}
