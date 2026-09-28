use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend,
    dielectric::{Dielectric2D, DielectricOptions, SmoothingMethod},
    eigensolver::dense::solve_hermitian_eigen,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    geometry_warm_start::{
        predict::rayleigh_ritz_trial,
        response::predict,
        snapshot::{BasisGeneration, NativeSnapshot, Quality},
    },
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::response::ResponseConfig,
};
use num_complex::Complex64 as C;
type Op = ThetaOperator<CpuBackend<f64>>;
#[test]
fn integer_cell_translation_preserves_the_material_pencil_and_shape_trials_obey_budgets() {
    use blaze2d_core::geometry_warm_start::shape::{self, Map, Options, Report};
    for lattice in [Lattice2D::square(1.), Lattice2D::hexagonal(1.)] {
        for pol in [Polarization::TM, Polarization::TE] {
            let n=9;let k=[0.17,0.13];let from=[1.,0.,0.,0.203,8.9];
            let to=[1.,1./n as f64,0.,0.203,8.9];
            let mut source=operator_at(n,pol,0.203,lattice,k,[0.,0.]);
            let pairs=dense(&mut source,7);
            let snapshot=NativeSnapshot::capture(&mut source,pairs,4,
                BasisGeneration{job:0,k_index:0,accepted_generation:1},vec![],from.to_vec(),true,true).unwrap();
            assert!(snapshot.admitted(Quality::default()));
            let mut target=operator_at(n,pol,0.203,lattice,k,[to[1],to[2]]);
            let map=Map::new(snapshot.key(),&from,&to,Options::default()).unwrap();
            let mut report=Report::default();
            let transported=shape::transport(target.backend(),target.grid(),snapshot.fields(),&map,Options::default(),1<<24,&mut report).unwrap();
            assert_eq!(report.forward_ffts,0);
            for (x,&lambda) in transported.iter().zip(snapshot.eigenvalues()) {
                let mut ax=target.alloc_field();let mut bx=target.alloc_field();
                target.apply(x,&mut ax);target.apply_mass(x,&mut bx);
                target.backend().axpy(C::new(-lambda,0.),&bx,&mut ax);
                assert!(target.backend().dot(&ax,&ax).re.sqrt()<1e-9);
            }
            let prediction=shape::prepare(&mut target,&snapshot,&to,Options::default(),None,7,14,1<<24).0.unwrap();
            for (x,a) in prediction.vectors.iter().zip(prediction.actions.as_ref().unwrap()) {
                let mut actual=target.alloc_field();target.apply(x,&mut actual);target.backend().axpy(-C::ONE,a,&mut actual);
                assert!(target.backend().dot(&actual,&actual).re.sqrt()<1e-9);
            }
            let (rejected,report)=shape::prepare(&mut target,&snapshot,&to,Options::default(),Some(&prediction.vectors),7,7,1<<24);
            assert!(rejected.is_none());assert!(report.reason.unwrap().contains("column budget"));
            let (rejected,report)=shape::prepare(&mut target,&snapshot,&to,Options::default(),None,7,14,1024);
            assert!(rejected.is_none());assert_eq!(report.interpolation_terms,0);
        }
    }
}
fn operator(n: usize, pol: Polarization, radius: f64) -> Op {
    operator_at(n, pol, radius, Lattice2D::square(1.), [0., 0.], [0., 0.])
}
fn operator_at(
    n: usize,
    pol: Polarization,
    radius: f64,
    lattice: Lattice2D,
    k: [f64; 2],
    position: [f64; 2],
) -> Op {
    let mut options = DielectricOptions::default();
    options.smoothing.method = SmoothingMethod::Voronoi;
    let d = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.,
            atoms: vec![BasisAtom {
                pos: position,
                radius,
                eps_inside: 8.9,
            }],
        },
        Grid2D::new(n, n, 1., 1.),
        &options,
    );
    ThetaOperator::new(
        CpuBackend::new(),
        d,
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}
fn dense(op: &mut Op, bands: usize) -> Vec<(f64, Field2D)> {
    let n = op.grid().len();
    let mut a = vec![C::ZERO; n * n];
    let mut x = op.alloc_field();
    let mut y = op.alloc_field();
    let inverse: Vec<_> = (0..n)
        .map(|i| {
            if op.polarization() == Polarization::TM {
                op.dielectric().eps()[i].sqrt().recip()
            } else {
                1.
            }
        })
        .collect();
    for j in 0..n {
        x.as_mut_slice().fill(C::ZERO);
        x.as_mut_slice()[j] = C::new(inverse[j], 0.);
        op.apply(&x, &mut y);
        for i in 0..n {
            a[i + j * n] = y.as_slice()[i] * inverse[i];
        }
    }
    for j in 0..n {
        for i in 0..=j {
            let v = 0.5 * (a[i + j * n] + a[j + i * n].conj());
            a[i + j * n] = v;
            a[j + i * n] = v.conj();
        }
    }
    let eig = solve_hermitian_eigen(&a, n);
    (0..bands)
        .map(|j| {
            let gamma = j == 0 && op.analytic_kernel_available();
            let values = if gamma {
                let norm = if op.polarization() == Polarization::TM {
                    op.dielectric().eps().iter().sum::<f64>().sqrt()
                } else {
                    (n as f64).sqrt()
                };
                vec![C::new(norm.recip(), 0.); n]
            } else {
                (0..n)
                    .map(|i| eig.eigenvectors[i + j * n] * inverse[i])
                    .collect()
            };
            (
                if gamma { 0. } else { eig.eigenvalues[j] },
                Field2D::from_vec(op.grid(), values),
            )
        })
        .collect()
}
fn span_error(op: &mut Op, actual: &[Field2D], reference: &[(f64, Field2D)]) -> f64 {
    let mut total = 0.;
    let mut mass = op.alloc_field();
    for field in actual {
        let mut residual = field.clone();
        for _ in 0..2 {
            op.apply_mass(&residual, &mut mass);
            for (_, q) in reference {
                let c = op.backend().dot(q, &mass);
                op.backend().axpy(-c, q, &mut residual);
            }
        }
        op.apply_mass(&residual, &mut mass);
        total += op.backend().dot(&residual, &mass).re;
    }
    total.max(0.).sqrt()
}
#[test]
fn native_gamma_response_has_second_order_low_span_and_valid_products() {
    for n in [7, 9] {
        for pol in [Polarization::TM, Polarization::TE] {
            let mut source = operator(n, pol, 0.203);
            let pairs = dense(&mut source, 7);
            // The requested window must contain the whole first doublet. A cut
            // inside it compares arbitrary bases, not a separated invariant span.
            assert!(pairs[4].0 - pairs[3].0 > 0.1);
            eprintln!(
                "spectrum {n} {pol:?}: {:?}",
                pairs.iter().map(|p| p.0).collect::<Vec<_>>()
            );
            let snapshot = NativeSnapshot::capture(
                &mut source,
                pairs,
                4,
                BasisGeneration {
                    job: 0,
                    k_index: 0,
                    accepted_generation: 1,
                },
                vec![],
                vec![0.203],
                true,
                true,
            )
            .unwrap();
            assert!(
                snapshot.admitted(Quality::default()),
                "{:?}",
                snapshot.certificate()
            );
            let mut copy_errors = Vec::new();
            let mut response_errors = Vec::new();
            for h in [0.001, 0.0005, 0.00025, 0.000125] {
                let mut target = operator(n, pol, 0.203 + h);
                let reference = dense(&mut target, 4);
                let response = predict(
                    &mut target,
                    &snapshot,
                    ResponseConfig {
                        max_inner_iterations: 256,
                        inner_relative_tolerance: 1e-11,
                    },
                    Quality::default(),
                    true,
                    1 << 24,
                )
                .unwrap();
                assert!(response.report.includes_gamma_kernel);
                assert_eq!(response.report.excluded_columns, 7);
                assert_eq!(response.report.response_columns, 3);
                assert_eq!(response.report.copied_guards, 3);
                assert!(
                    response
                        .report
                        .inner_columns
                        .iter()
                        .all(|c| c.inner_converged && c.excluded_space_defect < 1e-10)
                );
                if let Some(actions) = &response.actions {
                    let mut fresh = target.alloc_field();
                    for (x, ax) in response.vectors.iter().zip(actions) {
                        target.apply(x, &mut fresh);
                        target.backend().axpy(-C::ONE, ax, &mut fresh);
                        assert!(target.backend().dot(&fresh, &fresh).re.sqrt() < 1e-9);
                    }
                }
                let copy = rayleigh_ritz_trial(&mut target, snapshot.fields().to_vec(), 6, 1 << 24)
                    .unwrap();
                let response =
                    rayleigh_ritz_trial(&mut target, response.vectors, 6, 1 << 24).unwrap();
                copy_errors.push(span_error(&mut target, &copy.vectors[..3], &reference[1..]));
                response_errors.push(span_error(
                    &mut target,
                    &response.vectors[..3],
                    &reference[1..],
                ));
            }
            let copy_order = (copy_errors[1] / copy_errors[3]).log2() / 2.;
            let response_order = (response_errors[1] / response_errors[3]).log2() / 2.;
            eprintln!(
                "{n} {pol:?} copy={copy_errors:?} response={response_errors:?} orders=({copy_order},{response_order})"
            );
            assert!((0.9..1.1).contains(&copy_order));
            assert!((1.8..2.2).contains(&response_order));
            assert!(response_errors[3] < 0.05 * copy_errors[3]);
            let mut target = operator(n, pol, 0.204);
            for budget in [1, 2, 4] {
                let r = predict(
                    &mut target,
                    &snapshot,
                    ResponseConfig {
                        max_inner_iterations: budget,
                        inner_relative_tolerance: 1e-11,
                    },
                    Quality::default(),
                    false,
                    1 << 24,
                )
                .unwrap();
                assert!(r.actions.is_none());
                assert!(
                    r.report
                        .inner_columns
                        .iter()
                        .all(|c| c.iterations <= budget && c.excluded_space_defect < 1e-10)
                );
                assert!(r.report.inner_columns.iter().any(|c| !c.inner_converged));
            }
        }
    }
}

#[test]
fn nonzero_response_uses_its_own_pencil_through_radius_and_displacement_changes() {
    for (name, lattice) in [
        ("square", Lattice2D::square(1.)),
        ("triangular", Lattice2D::hexagonal(1.)),
    ] {
        for pol in [Polarization::TM, Polarization::TE] {
            for (kind, k) in [("radius", [0.5, 0.]), ("displacement", [0.17, 0.13])] {
                let mut source = operator_at(7, pol, 0.203, lattice, k, [0., 0.]);
                let pairs = dense(&mut source, 8);
                let requested = (2..=4)
                    .rev()
                    .find(|&p| pairs[p].0 - pairs[p - 1].0 > 0.05)
                    .unwrap();
                let snapshot = NativeSnapshot::capture(
                    &mut source,
                    pairs,
                    requested,
                    BasisGeneration {
                        job: 0,
                        k_index: 1,
                        accepted_generation: 1,
                    },
                    vec![],
                    vec![0.203],
                    true,
                    true,
                )
                .unwrap();
                assert!(snapshot.admitted(Quality::default()));
                assert!(snapshot.kernel().is_none());
                let mut errors = Vec::new();
                for h in [0.0004, 0.0002, 0.0001] {
                    let (radius, position) = if kind == "radius" {
                        (0.203 + h, [0., 0.])
                    } else {
                        (0.203, [h, 0.])
                    };
                    let mut target = operator_at(7, pol, radius, lattice, k, position);
                    let reference = dense(&mut target, requested);
                    let response = predict(
                        &mut target,
                        &snapshot,
                        ResponseConfig {
                            max_inner_iterations: 256,
                            inner_relative_tolerance: 1e-11,
                        },
                        Quality::default(),
                        true,
                        1 << 24,
                    )
                    .unwrap();
                    assert!(!response.report.includes_gamma_kernel);
                    assert_eq!(response.report.response_columns, requested);
                    assert!(
                        response
                            .report
                            .inner_columns
                            .iter()
                            .all(|c| c.inner_converged && c.excluded_space_defect < 1e-10)
                    );
                    let trial =
                        rayleigh_ritz_trial(&mut target, response.vectors, requested, 1 << 24)
                            .unwrap();
                    errors.push(span_error(&mut target, &trial.vectors, &reference));
                }
                let order = (errors[0] / errors[2]).log2() / 2.;
                eprintln!(
                    "{name} {pol:?} {kind} k={k:?} requested={requested} errors={errors:?} response_order={order}"
                );
                assert!((1.8..2.2).contains(&order));
            }
        }
    }
}

#[test]
fn compressed_history_preserves_the_full_trial_span_at_the_next_metric() {
    use blaze2d_core::geometry_warm_start::innovation::compress;
    for pol in [Polarization::TM,Polarization::TE] {
        let mut source=operator(7,pol,0.203);let pairs=dense(&mut source,7);
        let previous=NativeSnapshot::capture(&mut source,pairs,4,BasisGeneration{job:0,k_index:0,accepted_generation:1},vec![],vec![0.203],true,false).unwrap();
        let mut middle=operator(7,pol,0.205);let pairs=dense(&mut middle,7);
        let newest=NativeSnapshot::capture(&mut middle,pairs,4,BasisGeneration{job:1,k_index:0,accepted_generation:2},vec![],vec![0.205],true,false).unwrap();
        let compressed=compress(&mut middle,&newest,&previous,6,1e-7,1<<24).unwrap();
        assert!(compressed.matches(&newest));assert!(!compressed.matches(&previous));
        assert!(!compressed.fields().is_empty());assert!(compressed.fields().len()<=6);
        assert!(compressed.report.maximum_newest_overlap<1e-11);
        let mut target=operator(7,pol,0.207);
        let transport=|fields:&[Field2D],from:&Op,to:&Op|fields.iter().map(|f| {
            let values=f.as_slice().iter().enumerate().map(|(i,&z)|z*if pol==Polarization::TM {
                (from.dielectric().eps()[i]/to.dielectric().eps()[i]).sqrt()
            } else {1.}).collect();Field2D::from_vec(f.grid(),values)
        }).collect::<Vec<_>>();
        let mut full=transport(newest.fields(),&middle,&target);full.extend(transport(previous.fields(),&source,&target));
        let mut thin=transport(newest.fields(),&middle,&target);thin.extend(transport(compressed.fields(),&middle,&target));
        let full=rayleigh_ritz_trial(&mut target,full,6,1<<24).unwrap();
        let thin=rayleigh_ritz_trial(&mut target,thin,6,1<<24).unwrap();
        let reference:Vec<_>=full.vectors.into_iter().map(|x|(0.,x)).collect();
        let error=span_error(&mut target,&thin.vectors,&reference);
        eprintln!("{pol:?}: innovation rank {}, kernel component {}, next-metric span error {error:e}",compressed.fields().len(),compressed.report.maximum_kernel_component);
        assert!(error<1e-8);
        let field_bytes=7*7*std::mem::size_of::<C>();
        let prepared=compressed.fields().len();
        let cap=newest.payload_bytes()+compressed.payload_bytes()-(prepared/2)*field_bytes;
        let mut bank=blaze2d_core::geometry_warm_start::cache::NativeBank::new(cap,1);
        let key=newest.key().clone();
        let report=bank.insert_with_innovation(newest,Quality::default(),Some(compressed));
        assert!(report.stored);assert!(report.innovation_trimmed_columns>0);assert!(report.payload_bytes<=cap);
        let leased=bank.donors(&key,1);let frame=bank.innovations_for(&leased[0]).unwrap();
        assert_eq!(frame.fields().len()+report.innovation_trimmed_columns,prepared);
        bank.clear();assert_eq!(bank.payload_bytes(),0);
        assert!(frame.matches(&leased[0]));assert!(!frame.fields().is_empty());
    }
}

#[test]
fn reusable_tangent_and_complete_complement_recover_second_order_response() {
    use blaze2d_core::geometry_warm_start::model::{self,Attempt,Basis,Kind,Options};
    use std::sync::Arc;
    for pol in [Polarization::TM,Polarization::TE] {
        let mut source=operator(7,pol,0.203);let pairs=dense(&mut source,7);
        let snapshot=Arc::new(NativeSnapshot::capture(&mut source,pairs,4,
            BasisGeneration{job:0,k_index:0,accepted_generation:1},vec![],vec![0.203],true,false).unwrap());
        let options=Options{basis:Basis::Fourier,columns:49,training_iterations:256,
            training_tolerance:1e-11,..Default::default()};
        for kind in [Kind::Tangent,Kind::Reduced] {
            let mut errors=Vec::new();let mut copies=Vec::new();
            for h in [0.0005,0.00025,0.000125] {
                let mut training=operator(7,pol,0.203+h);let mut built=Attempt::default();
                let model=model::build(&mut training,snapshot.clone(),None,&[0.203+h],kind,
                    options,Quality::default(),1<<25,&mut built).unwrap();
                let mut target=operator(7,pol,0.203+2.*h);let reference=dense(&mut target,4);
                let mut evaluated=Attempt::default();
                let fields=model::evaluate(&mut target,&model,&[0.203+2.*h],1<<25,&mut evaluated).unwrap();
                if kind==Kind::Reduced {
                    assert_eq!(built.basis_columns,42);
                    assert!(evaluated.projected_relative_residuals.iter().all(|&r|r<1e-10));
                    assert!(evaluated.full_relative_residuals.iter().all(|&r|r<1e-8),"{evaluated:?}");
                } else {assert!(built.training.as_ref().unwrap().inner_columns.iter().all(|c|c.inner_converged));}
                assert!((evaluated.step_ratio.unwrap()-2.).abs()<1e-10);
                let trial=rayleigh_ritz_trial(&mut target,fields,6,1<<25).unwrap();
                let copy=rayleigh_ritz_trial(&mut target,snapshot.fields().to_vec(),6,1<<25).unwrap();
                errors.push(span_error(&mut target,&trial.vectors[..3],&reference[1..]));
                copies.push(span_error(&mut target,&copy.vectors[..3],&reference[1..]));
            }
            let order=(errors[0]/errors[2]).log2()/2.;
            eprintln!("M7 {pol:?} {kind:?} errors={errors:?}, copy={copies:?}, order={order}");
            assert!((1.8..2.2).contains(&order));assert!(errors[2]<0.05*copies[2]);
        }
    }
}

#[test]
fn small_response_spaces_report_the_omitted_equation_and_keep_valid_target_actions() {
    use blaze2d_core::geometry_warm_start::model::{self,Attempt,Basis,Kind,Options};
    use std::sync::Arc;
    for pol in [Polarization::TM,Polarization::TE] {
        let mut old=operator(7,pol,0.201);let pairs=dense(&mut old,7);
        let previous=NativeSnapshot::capture(&mut old,pairs,4,BasisGeneration{job:0,k_index:0,accepted_generation:1},vec![],vec![0.201],true,false).unwrap();
        let mut source=operator(7,pol,0.203);let pairs=dense(&mut source,7);
        let source=Arc::new(NativeSnapshot::capture(&mut source,pairs,4,BasisGeneration{job:1,k_index:0,accepted_generation:2},vec![],vec![0.203],true,false).unwrap());
        for basis in [Basis::Fourier,Basis::History,Basis::Residual] {
            let mut target=operator(7,pol,0.205);let mut built=Attempt::default();
            let options=Options{basis,columns:4,..Default::default()};
            let model=model::build(&mut target,source.clone(),Some(&previous),&[0.205],Kind::Reduced,
                options,Quality::default(),1<<25,&mut built).unwrap();
            let mut evaluated=Attempt::default();
            let fields=model::evaluate(&mut target,&model,&[0.205],1<<25,&mut evaluated).unwrap();
            assert_eq!(built.basis_columns,4);assert_eq!(built.work.basis_a_columns,4);
            assert!(evaluated.projected_relative_residuals.iter().all(|&r|r<1e-10));
            assert!(evaluated.full_relative_residuals.iter().any(|&r|r>1e-3));
            let trial=rayleigh_ritz_trial(&mut target,fields,6,1<<25).unwrap();
            let mut fresh=target.alloc_field();
            for (x,ax) in trial.vectors.iter().zip(trial.actions.as_ref().unwrap()) {
                target.apply(x,&mut fresh);target.backend().axpy(-C::ONE,ax,&mut fresh);
                assert!(target.backend().dot(&fresh,&fresh).re.sqrt()<1e-9);
            }
            eprintln!("M7 {pol:?} {basis:?} omitted equation residuals={:?}",evaluated.full_relative_residuals);
        }
    }
}

#[test]
fn tangent_refresh_and_cache_admission_preserve_immutable_anchors() {
    use blaze2d_core::geometry_warm_start::{model::{self,Attempt,Kind,Lease,Options},cache::NativeBank};
    use std::sync::Arc;
    let mut source=operator(7,Polarization::TM,0.203);let pairs=dense(&mut source,7);
    let source=Arc::new(NativeSnapshot::capture(&mut source,pairs,4,BasisGeneration{job:0,k_index:0,accepted_generation:1},vec![],vec![0.203],true,false).unwrap());
    let mut middle=operator(7,Polarization::TM,0.204);let mut built=Attempt::default();
    let options=Options{training_iterations:256,training_tolerance:1e-10,..Default::default()};
    let model=model::build(&mut middle,source.clone(),None,&[0.204],Kind::Tangent,options,Quality::default(),1<<25,&mut built).unwrap();
    let pairs=dense(&mut middle,7);
    let latest=Arc::new(NativeSnapshot::capture(&mut middle,pairs,4,BasisGeneration{job:1,k_index:0,accepted_generation:2},vec![],vec![0.204],true,false).unwrap());
    let cap=latest.payload_bytes()+model.payload_bytes()+16;
    let mut bank=NativeBank::new(cap,1);
    let stored=bank.insert_with_model(latest.clone(),Quality::default(),None,Some(Lease{model:model.clone(),uses:0}));
    assert!(stored.stored && stored.model_stored && stored.payload_bytes<=cap);
    let leased=bank.model_for(&latest).unwrap();bank.resize(0,1);assert_eq!(bank.payload_bytes(),0);
    let mut target=operator(7,Polarization::TM,0.205);
    let (prediction,retained,report)=model::prepare(&mut target,latest.clone(),None,Some(leased),&[0.205],Kind::Tangent,
        options,Quality::default(),6,1<<25);
    assert!(prediction.is_some() && report.accepted);assert_eq!(report.attempts.len(),1);
    assert!(report.attempts[0].reused);assert_eq!(retained.as_ref().unwrap().uses,1);
    let mut reversed=operator(7,Polarization::TM,0.202);
    let (_,refreshed,report)=model::prepare(&mut reversed,latest.clone(),None,retained,&[0.202],Kind::Tangent,
        options,Quality::default(),6,1<<25);
    assert!(report.attempts[0].rejection.as_ref().unwrap().contains("direction"));
    assert!(report.accepted);assert_eq!(refreshed.unwrap().model.anchor(),latest.basis_generation());
    let mut small=NativeBank::new(latest.payload_bytes()+16,1);
    let stored=small.insert_with_model(latest.clone(),Quality::default(),None,Some(Lease{model,uses:0}));
    assert!(stored.stored && !stored.model_stored && stored.model_rejection.is_some());
    assert_eq!(small.donors(latest.key(),1)[0].basis_generation(),latest.basis_generation());
    let mut failed=Attempt::default();
    assert!(model::build(&mut target,source,None,&[0.205],Kind::Tangent,Options {training_iterations:1,
        training_tolerance:1e-12,..options},Quality::default(),1<<25,&mut failed).is_err());
    assert!(failed.work.response.inner_a_columns>0);
}

#[test]
fn reusable_models_follow_nonzero_displacement_on_both_lattices() {
    use blaze2d_core::geometry_warm_start::model::{self,Attempt,Kind,Options};
    use std::sync::Arc;
    for lattice in [Lattice2D::square(1.),Lattice2D::hexagonal(1.)] {
      for pol in [Polarization::TM,Polarization::TE] {
        let k=[0.17,0.13];
        let mut source=operator_at(7,pol,0.203,lattice,k,[0.,0.]);let pairs=dense(&mut source,7);
        let requested=(2..=4).rev().find(|&p|pairs[p].0-pairs[p-1].0>0.05).unwrap();
        let snapshot=Arc::new(NativeSnapshot::capture(&mut source,pairs,requested,
            BasisGeneration{job:0,k_index:1,accepted_generation:1},vec![],vec![0.],true,false).unwrap());
        for kind in [Kind::Tangent,Kind::Reduced] {
            let options=Options{columns:49,training_iterations:256,training_tolerance:1e-11,..Default::default()};
            let mut errors=Vec::new();
            for h in [0.0002,0.0001,0.00005] {
                let mut train=operator_at(7,pol,0.203,lattice,k,[h,0.]);let mut built=Attempt::default();
                let model=model::build(&mut train,snapshot.clone(),None,&[h],kind,options,Quality::default(),1<<25,&mut built).unwrap();
                let mut target=operator_at(7,pol,0.203,lattice,k,[2.*h,0.]);let reference=dense(&mut target,requested);
                let mut report=Attempt::default();let fields=model::evaluate(&mut target,&model,&[2.*h],1<<25,&mut report).unwrap();
                let trial=rayleigh_ritz_trial(&mut target,fields,requested,1<<25).unwrap();
                errors.push(span_error(&mut target,&trial.vectors,&reference));
                let mut wrong=operator_at(7,pol,0.203,lattice,[0.18,0.13],[2.*h,0.]);
                assert!(model::evaluate(&mut wrong,&model,&[2.*h],1<<25,&mut Attempt::default()).is_err());
            }
            let order=(errors[0]/errors[2]).log2()/2.;
            eprintln!("M7 nonzero {:?} {pol:?} {kind:?}: errors={errors:?}, order={order}",lattice.a2);
            assert!((1.8..2.2).contains(&order));
        }
      }
    }
}
