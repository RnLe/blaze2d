use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend,
    dielectric::{Dielectric2D, DielectricOptions, SmoothingMethod},
    field::{Field2D, Real, cscalar},
    geometry::{BasisAtom, Geometry2D},
    geometry_warm_start::{
        difference::DiscreteDifference,
        projected::ProjectedPencil,
        snapshot::{BasisGeneration, NativeSnapshot},
    },
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
};
use num_complex::Complex64 as C;

fn operator<B: SpectralBackend>(
    backend: B,
    n: usize,
    pol: Polarization,
    k: [f64; 2],
    changed: bool,
) -> ThetaOperator<B> {
    let geometry = Geometry2D {
        lattice: Lattice2D::hexagonal(1.),
        eps_bg: 1.,
        atoms: vec![BasisAtom {
            pos: if changed { [0.13, 0.08] } else { [0.07, 0.09] },
            radius: if changed { 0.237 } else { 0.213 },
            eps_inside: if changed { 12. } else { 8.9 },
        }],
    };
    let mut options = DielectricOptions::default();
    options.smoothing.method = SmoothingMethod::Voronoi;
    let d = Dielectric2D::from_geometry(&geometry, Grid2D::new(n, n, 1., 1.), &options);
    ThetaOperator::new(backend, d, pol, k)
}
fn norm<B: SpectralBackend>(b: &B, v: &B::Buffer) -> f64 {
    b.dot(v, v).re.sqrt()
}
fn qualify<R: Real, B: SpectralBackend<Real = R, Buffer = Field2D<R>> + Clone>(
    backend: B,
    tolerance: f64,
) {
    for pol in [Polarization::TM, Polarization::TE] {
        for n in [7, 11] {
            let mut a = operator(backend.clone(), n, pol, [0., 0.], false);
            let mut b = operator(backend.clone(), n, pol, [0., 0.], true);
            let change = DiscreteDifference::new(&a, &b).unwrap();
            assert!(
                change.report.changed_epsilon_cells > 0 && change.report.changed_tensor_cells > 0
            );
            let mut x = a.alloc_field();
            let mut ax = a.alloc_field();
            let mut bx = a.alloc_field();
            let mut dx = a.alloc_field();
            let mut db = a.alloc_field();
            let mut mass0 = a.alloc_field();
            let mut mass1 = a.alloc_field();
            // Every coordinate column checks the actual finite matrices, including
            // all tensor cross-components and the target/source residual identity.
            for column in 0..n * n {
                x.as_mut_slice().fill(cscalar(0., 0.));
                x.as_mut_slice()[column] = cscalar(1., 0.);
                a.apply(&x, &mut ax);
                b.apply(&x, &mut bx);
                a.apply_mass(&x, &mut mass0);
                b.apply_mass(&x, &mut mass1);
                change.apply(&mut b, &x, &mut dx, &mut db).unwrap();
                let scale = norm(&backend, &ax) + norm(&backend, &bx) + 1.;
                backend.axpy(-C::ONE, &ax, &mut bx);
                backend.axpy(-C::ONE, &dx, &mut bx);
                assert!(
                    norm(&backend, &bx) < tolerance * scale,
                    "delta A {pol:?} n={n} column={column}"
                );
                backend.axpy(-C::ONE, &mass0, &mut mass1);
                backend.axpy(-C::ONE, &db, &mut mass1);
                assert!(norm(&backend, &mass1) < tolerance * 16.);
                backend.axpy(C::new(-2.3, 0.), &mass1, &mut bx);
                assert!(norm(&backend, &bx) < tolerance * scale, "residual identity");
            }
            let mut pairs = Vec::new();
            let mass = if pol == Polarization::TM {
                a.dielectric().eps().iter().sum::<f64>()
            } else {
                (n * n) as f64
            };
            x.as_mut_slice().fill(cscalar(mass.sqrt().recip(), 0.));
            pairs.push((0., x.clone()));
            for j in 1..=3 {
                for (i, z) in x.as_mut_slice().iter_mut().enumerate() {
                    *z = cscalar(
                        ((i + 3 * j) as f64 * 0.7).sin(),
                        ((i + j) as f64 * 0.31).cos(),
                    );
                }
                pairs.push((j as f64, x.clone()));
            }
            let generation = BasisGeneration {
                job: 1,
                k_index: 0,
                accepted_generation: 1,
            };
            // A fixed-basis identity holds for arbitrary columns, not only for
            // converged eigenvectors. This deliberately unqualified block is never
            // admitted as a continuation donor.
            let snapshot =
                NativeSnapshot::capture(&mut a, pairs, 4, generation, vec![], vec![], false, true)
                    .unwrap();
            let products = ProjectedPencil::prepare(&mut a, &snapshot, 1 << 24).unwrap();
            assert!(ProjectedPencil::prepare(&mut a, &snapshot, 1).is_err());
            for k in [[0., 0.], [0.21, -0.13]] {
                let mut target = operator(backend.clone(), n, pol, k, true);
                let pencil = products.evaluate(&target, generation).unwrap();
                for j in 0..3 {
                    x.as_mut_slice()
                        .copy_from_slice(snapshot.fields()[j].as_slice());
                    target.apply(&x, &mut ax);
                    target.apply_mass(&x, &mut bx);
                    for i in 0..3 {
                        let h = backend.dot(&snapshot.fields()[i], &ax);
                        let m = backend.dot(&snapshot.fields()[i], &bx);
                        assert!(
                            (h - pencil.stiffness[i + j * 3]).norm() < tolerance * h.norm().max(1.),
                            "projected A {pol:?} {k:?}"
                        );
                        assert!(
                            (m - pencil.mass[i + j * 3]).norm() < tolerance * m.norm().max(1.),
                            "projected B"
                        );
                    }
                }
                let mut wrong = generation;
                wrong.accepted_generation += 1;
                assert!(products.evaluate(&target, wrong).is_err());
                if k != [0., 0.] {
                    assert!(DiscreteDifference::new(&a, &target).is_err());
                }
            }
        }
    }
}
#[test]
fn finite_tensor_differences_and_fixed_basis_updates_match_full_operators() {
    qualify(CpuBackend::<f64>::new(), 2e-11);
    qualify(CpuBackend::<f32>::new(), 3e-5);
}
