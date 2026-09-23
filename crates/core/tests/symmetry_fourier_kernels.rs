use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend,
    dielectric::Dielectric2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    preconditioners::OperatorPreconditioner,
};
use num_complex::Complex64 as C;

#[test]
fn direct_fourier_kernels_match_spatial_operators_and_preconditioners() {
    let backend = CpuBackend::<f64>::new();
    for n in [12, 13] {
        for lattice in [Lattice2D::square(1.0), Lattice2D::hexagonal(1.0)] {
            let grid = Grid2D::new(n, n - 2, 1.0, 1.0);
            let dielectric = Dielectric2D::from_geometry(
                &Geometry2D {
                    lattice, eps_bg: 1.0,
                    atoms: vec![BasisAtom { pos: [0.13, 0.07], radius: 0.2, eps_inside: 8.9 }],
                }, grid, &Default::default(),
            );
            assert!(dielectric.inv_eps_tensors().is_some());
            for pol in [Polarization::TM, Polarization::TE] {
                for k in [[0.0; 2], [0.13, 0.21], [0.5, 0.0]] {
                    let mut operator = ThetaOperator::new(backend.clone(), dielectric.clone(), pol,
                        lattice.reciprocal().fractional_to_cartesian(k));
                    let mut input = operator.alloc_field();
                    for (i, value) in input.as_mut_slice().iter_mut().enumerate() {
                        *value = C::new((i as f64 * 0.723).sin(), (i as f64 * 1.31).cos());
                    }
                    let mut reference = operator.alloc_field();
                    operator.apply(&input, &mut reference);
                    backend.forward_fft_2d(&mut reference);
                    let mut fourier_input = input.clone();
                    backend.forward_fft_2d(&mut fourier_input);
                    let mut direct = operator.alloc_field();
                    #[cfg(feature = "profiling")]
                    let before = fft_count();
                    operator.apply_fourier(&fourier_input, &mut direct);
                    #[cfg(feature = "profiling")]
                    assert_eq!(fft_count() - before, if pol == Polarization::TM { 0 } else { 4 });
                    assert_close(reference.as_slice(), direct.as_slice());

                    let mut preconditioners: Vec<Box<dyn OperatorPreconditioner<CpuBackend<f64>>>> = vec![
                        Box::new(operator.build_homogeneous_preconditioner_adaptive()),
                        Box::new(operator.build_transverse_projection_preconditioner_adaptive()),
                    ];
                    for (kind, preconditioner) in preconditioners.iter_mut().enumerate() {
                        let mut reference = input.clone();
                        preconditioner.apply(&backend, &mut reference);
                        backend.forward_fft_2d(&mut reference);
                        let mut direct = fourier_input.clone();
                        #[cfg(feature = "profiling")]
                        let before = fft_count();
                        assert!(preconditioner.apply_fourier(&backend, &mut direct));
                        #[cfg(feature = "profiling")]
                        assert_eq!(fft_count() - before, if kind == 0 { 0 } else { 4 });
                        #[cfg(not(feature = "profiling"))]
                        let _ = kind;
                        assert_close(reference.as_slice(), direct.as_slice());
                    }
                }
            }
        }
    }
}

fn assert_close(a: &[C], b: &[C]) {
    let error = a.iter().zip(b).map(|(x,y)| (x-y).norm_sqr()).sum::<f64>();
    let norm = a.iter().map(C::norm_sqr).sum::<f64>();
    assert!((error / norm.max(1e-300)).sqrt() < 2e-12);
}

#[cfg(feature = "profiling")]
fn fft_count() -> u64 {
    let c = blaze2d_core::profiler::get_counters();
    c.get("fft.forward.vectors").copied().unwrap_or(0)
        + c.get("fft.inverse.vectors").copied().unwrap_or(0)
}
