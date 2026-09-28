use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    eigensolver::{Eigensolver, EigensolverConfig, InitialAcceptance},
    field::{Field2D, Real},
    geometry::Geometry2D,
    geometry_warm_start::{
        cache::NativeBank,
        snapshot::{BasisGeneration, FiberKey, NativeSnapshot, Quality},
    },
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
};
use num_complex::{Complex, Complex64 as C};

fn op<B: SpectralBackend>(backend: B, epsilon: f64) -> ThetaOperator<B> {
    let grid = Grid2D::new(13, 13, 1., 1.);
    let material = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice: Lattice2D::square(1.),
            eps_bg: epsilon,
            atoms: vec![],
        },
        grid,
        &Default::default(),
    );
    ThetaOperator::new(backend, material, Polarization::TM, [0.; 2])
}
fn waves<B: SpectralBackend>(backend: &B, epsilon: f64) -> Vec<B::Buffer> {
    let grid = Grid2D::new(13, 13, 1., 1.);
    let mut result = Vec::new();
    for (gx, gy) in [
        (0., 0.),
        (1., 0.),
        (-1., 0.),
        (0., 1.),
        (0., -1.),
        (1., 1.),
        (-1., 1.),
        (1., -1.),
        (-1., -1.),
    ] {
        let mut field = backend.alloc_field(grid);
        for j in 0..13 {
            for i in 0..13 {
                let z = C::from_polar(
                    (epsilon * 169.).sqrt().recip(),
                    std::f64::consts::TAU * (gx * i as f64 + gy * j as f64) / 13.,
                );
                field.as_mut_slice()[grid.idx(i, j)] =
                    Complex::new(B::Real::from_accum(z.re), B::Real::from_accum(z.im));
            }
        }
        result.push(field);
    }
    result
}

fn roundtrip<R: Real, B: SpectralBackend<Real = R, Buffer = Field2D<R>> + Clone>(
    backend: B,
    quality: Quality,
) {
    let mut source = op(backend.clone(), 2.);
    let config = EigensolverConfig {
        n_bands: 9,
        tol: 1e-14,
        ..Default::default()
    };
    let mut solver = Eigensolver::new(&mut source, config.clone(), None, None)
        .with_native_warm_start(waves(&backend, 2.));
    solver.set_initial_acceptance(InitialAcceptance {
        certified_prefix: 9,
        relative_residual: quality.relative_residual,
        absolute_residual: quality.absolute_residual,
        b_orthogonality: quality.orthogonality,
    });
    assert_eq!(solver.solve().iterations, 0);
    let pairs = solver.into_native_eigenpairs();
    assert_eq!(pairs.len(), 9);
    let generation = BasisGeneration {
        job: 0,
        k_index: 0,
        accepted_generation: 1,
    };
    let snapshot = NativeSnapshot::capture(
        &mut source,
        pairs,
        5,
        generation,
        vec![],
        vec![2.],
        true,
        true,
    )
    .unwrap();
    assert!(snapshot.admitted(quality), "{:?}", snapshot.certificate());
    assert_eq!(snapshot.fields().len(), 8);
    assert_eq!(snapshot.positive_requested(), 4);
    assert_eq!(snapshot.guard_count(), 4);
    assert!(snapshot.kernel().is_some());
    assert_eq!(snapshot.applied().unwrap().len(), 8);
    let unchanged=blaze2d_core::geometry_warm_start::innovation::compress(&mut source,&snapshot,&snapshot,8,1e-3,1<<24).unwrap();
    assert!(unchanged.fields().is_empty(),"arithmetic noise became a normalized innovation");
    let mut other_fiber=snapshot.key().clone();
    other_fiber.coordinate_bits[6]=0.1_f64.to_bits();
    assert!(snapshot.applied_for(&other_fiber).is_none());
    let mut target = op(backend, 2.1);
    let mut solver = Eigensolver::new(&mut target, config, None, None)
        .with_native_warm_start(snapshot.fields().to_vec());
    solver.set_initial_acceptance(InitialAcceptance {
        certified_prefix: 9,
        relative_residual: quality.relative_residual,
        absolute_residual: quality.absolute_residual,
        b_orthogonality: quality.orthogonality,
    });
    let result = solver.solve();
    assert_eq!(result.iterations, 0);
    assert_eq!(solver.initialization_report().unwrap().warm_start_hits, 8);
    for (i, &lambda) in result.eigenvalues.iter().enumerate() {
        let expected = std::f64::consts::TAU.powi(2) / 2.1
            * if i == 0 {
                0.
            } else if i < 5 {
                1.
            } else {
                2.
            };
        assert!((lambda - expected).abs() < quality.absolute_residual * 10.);
    }
}

#[test]
fn native_roundtrip_keeps_certified_guards_and_storage_precision() {
    roundtrip(CpuBackend::<f64>::new(), Quality::default());
    roundtrip(
        CpuBackend::<f32>::new(),
        Quality {
            relative_residual: 2e-4,
            absolute_residual: 1e-3,
            orthogonality: 2e-5,
        },
    );
}

#[test]
fn rejected_and_oversized_targets_preserve_borrowed_gamma_donors() {
    let backend = CpuBackend::<f64>::new();
    let mut operator = op(backend.clone(), 2.);
    let mut bank = NativeBank::new(usize::MAX, 2);
    let capture =
        |operator: &mut ThetaOperator<CpuBackend<f64>>, generation, epsilon: f64, converged| {
            let mut pairs: Vec<_> = waves(&backend, epsilon)
                .into_iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        std::f64::consts::TAU.powi(2) / epsilon
                            * if i == 0 {
                                0.
                            } else if i < 5 {
                                1.
                            } else {
                                2.
                            },
                        x,
                    )
                })
                .collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            NativeSnapshot::capture(
                operator,
                pairs,
                5,
                generation,
                vec![],
                vec![epsilon],
                converged,
                true,
            )
            .unwrap()
        };
    let first = bank.next_generation(0, 0);
    let snapshot = capture(&mut operator, first, 2., true);
    let size = snapshot.payload_bytes();
    bank.resize(size, 2);
    assert!(bank.insert(snapshot, Quality::default()).stored);
    let key = FiberKey::from_operator(&operator);
    let leased = bank.donors(&key, 1);
    assert_eq!(leased[0].basis_generation(), first);
    let next = bank.next_generation(1, 0);
    assert!(
        !bank
            .insert(capture(&mut operator, next, 2., false), Quality::default())
            .stored
    );
    assert_eq!(bank.donors(&key, 1)[0].basis_generation(), first);
    // Changed epsilon and native actions remain attached to their own generation.
    let mut target = op(backend.clone(), 2.1);
    assert!(
        bank.insert(capture(&mut target, next, 2.1, true), Quality::default())
            .stored
    );
    assert_eq!(bank.payload_bytes(), size);
    assert_eq!(bank.len(), 1);
    assert_eq!(leased[0].basis_generation(), first);
    assert_eq!(bank.donors(&key, 1)[0].basis_generation(), next);
    bank.resize(size - 1, 2);
    assert!(bank.is_empty());
    let rejected = bank.insert(capture(&mut target, next, 2.1, true), Quality::default());
    assert!(!rejected.stored);
    assert!(rejected.reason.unwrap().contains("budget"));
}

#[test]
fn signed_secants_reject_turns_reversals_and_unbounded_steps() {
    use blaze2d_core::geometry_warm_start::predict::secant_ratio;
    assert!((secant_ratio(&[0.1,2.],&[0.2,3.],&[0.25,3.5]).unwrap()-0.5).abs()<1e-12);
    assert!(secant_ratio(&[0.1,2.],&[0.2,3.],&[0.1,2.]).is_none());
    assert!(secant_ratio(&[0.1,2.],&[0.2,3.],&[0.3,3.]).is_none());
    assert!(secant_ratio(&[0.1],&[0.2],&[0.6]).is_none());
    assert!(secant_ratio(&[0.1],&[0.1],&[0.2]).is_none());
}
