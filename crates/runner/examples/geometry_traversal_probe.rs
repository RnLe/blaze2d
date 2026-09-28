//! Serial traversal experiment using the production native point solver.
//! This harness measures alternative schedules; it is not a public scheduler.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{band_tracking::{apply_permutation,track_bands_with_frequencies},
    dielectric::Dielectric2D,field::Field2D,geometry_warm_start::{path::PathHistory,point::{self,Context,DonorSelection,Method,Scope},snapshot::Quality},
    operators::ThetaOperator,preconditioners::OperatorPreconditioner,diagnostics::PreconditionerType};
use blaze2d_interface::{Config,Plan,Platform,Precision,Task};
use serde_json::json;
use std::{collections::{HashMap,VecDeque},time::Instant};

fn main() {
    let args:Vec<_>=std::env::args().collect();
    let config=Config::from_toml(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let strategy=args.get(2).map_or("geometry",String::as_str);
    let tile_p:usize=args.get(3).map_or(2,|s|s.parse().unwrap());
    let tile_k:usize=args.get(4).map_or(2,|s|s.parse().unwrap());
    assert!(tile_p>0 && tile_k>0 && matches!(strategy,"geometry"|"k"|"tile"));
    assert_eq!(config.task,Task::Bands);assert_eq!(config.eigensolver.precision,Precision::F64);
    let plan=Plan::new(config.clone(),Platform::Native).unwrap();
    let bands=config.bands.as_ref().unwrap();
    let quality=bands.accuracy.clone().unwrap_or_default();
    let options=point::Options {method:Method::Copy,scope:Scope::Path,anchors:vec![],donor_selection:DonorSelection::SameK,
        guards:2,max_guards:None,history:2,innovation_columns:0,innovation_relative_tolerance:1e-3,
        max_predictor_columns:0,max_preparation_bytes:64<<20,max_cache_bytes:64<<20,retain_tm_actions:false,
        quality:Quality {relative_residual:quality.max_relative_residual,absolute_residual:quality.max_absolute_residual,
            orthogonality:quality.max_orthogonality_defect},response_iterations:2,response_tolerance:0.25,
            model:Default::default(),transport:None};
    let p=plan.summary.jobs;let k=plan.job(0).unwrap().resolved.k_points_fractional.len();
    let mut nodes=Vec::new();
    match strategy {
        "geometry"=>for pi in 0..p {for ki in 0..k {nodes.push((pi,ki));}},
        "k"=>for ki in 0..k {for pi in 0..p {nodes.push((pi,ki));}},
        "tile"=>for p0 in (0..p).step_by(tile_p) {for k0 in (0..k).step_by(tile_k) {
            for pi in p0..(p0+tile_p).min(p) {for ki in k0..(k0+tile_k).min(k) {nodes.push((pi,ki));}}
        }},
        _=>unreachable!(),
    }
    let start=Instant::now();
    let backend=CpuBackend::<f64>::new();let mut context=Context::new(options.max_cache_bytes);
    let mut paths=HashMap::<usize,PathHistory<f64>>::new();
    let mut previous=HashMap::<usize,(Vec<f64>,Vec<Field2D>)>::new();
    let mut materials=VecDeque::<(usize,Dielectric2D)>::new();
    let mut records=Vec::new();let mut material_builds=0;let mut material_seconds=0.;let mut peak_payload=0;
    let mut roots=0;let mut rounds=0;let mut all_pass=true;
    for (pi,ki) in nodes {
        let job=plan.job(pi).unwrap();let core=job.bands_job();
        assert_eq!(core.k_path.len(),k);
        let material=if let Some(index)=materials.iter().position(|(id,_)|*id==pi) {
            let entry=materials.remove(index).unwrap();let d=entry.1.clone();materials.push_back(entry);d
        } else {
            let start=Instant::now();let d=Dielectric2D::from_geometry(&core.geom,core.grid,&core.dielectric);
            material_seconds+=start.elapsed().as_secs_f64();material_builds+=1;
            while materials.len()>=tile_p {materials.pop_front();}
            materials.push_back((pi,d.clone()));d
        };
        context.path=paths.remove(&pi).unwrap_or_else(||PathHistory::new(true,true));context.job_index=pi;
        let mut theta=ThetaOperator::new(backend.clone(),material.clone(),core.pol,job.resolved.k_points_cartesian[ki]);
        let mut preconditioner:Box<dyn OperatorPreconditioner<CpuBackend<f64>>>=match PreconditionerType::default().resolve_for_polarization(core.pol) {
            PreconditionerType::TransverseProjection=>Box::new(theta.build_transverse_projection_preconditioner_adaptive()),
            _=>Box::new(theta.build_homogeneous_preconditioner_adaptive()),
        };
        let parameters=std::iter::once(core.geom.eps_bg).chain(core.geom.atoms.iter().flat_map(|a|[a.pos[0],a.pos[1],a.radius,a.eps_inside])).collect();
        let distance=ki.checked_sub(1).map(|old|(core.k_path[ki][0]-core.k_path[old][0]).hypot(core.k_path[ki][1]-core.k_path[old][1]));
        let mut solved=point::solve(&mut theta,&core.eigensolver,Some(&mut *preconditioner),&mut context,
            &options,parameters,ki,distance,core.k_path[ki],true);
        let mut omega:Vec<_>=solved.values.iter().map(|l|l.max(0.).sqrt()).collect();
        let mut permutation:Vec<_>=(0..omega.len()).collect();
        if bands.tracking {
            if let Some((previous_omega,previous_fields))=previous.get(&pi) {
                let tracking=track_bands_with_frequencies(previous_fields,&solved.fields,previous_omega,&omega,
                    (core.pol==blaze2d_core::polarization::Polarization::TM).then_some(material.eps()));
                permutation=tracking.permutation;
                apply_permutation(&permutation,&mut omega,&mut solved.fields);
            }
        }
        let pass=solved.converged && solved.report.certificate.as_ref().is_some_and(|c|options.quality.accepts(c));
        all_pass &= pass;roots+=usize::from(solved.report.cold_root);rounds+=solved.report.iterations;
        if let Some(frame)=solved.native_frame {context.path.push(frame,core.k_path[ki],permutation.clone());}
        else {context.path.clear();}
        let material_payload:usize=materials.iter().map(|(_,d)|core.grid.len()*8*(2+4*usize::from(d.inv_eps_tensors().is_some())+usize::from(d.unsmoothed_eps().is_some())+usize::from(d.unsmoothed_inv_eps().is_some()))).sum();
        let tracking_payload:usize=previous.values().map(|(_,f)|f.iter().map(|x|x.len()*16).sum::<usize>()).sum();
        let resident=context.bank.payload_bytes()+context.path.payload_bytes()+paths.values().map(PathHistory::payload_bytes).sum::<usize>()+material_payload+tracking_payload+solved.report.leased_payload_bytes;
        peak_payload=peak_payload.max(resident);
        let fields=config.results.eigenvectors.then(||solved.fields.iter().flat_map(|x|x.as_slice().iter().flat_map(|z|[z.re,z.im])).collect::<Vec<_>>());
        records.push(json!({"job":pi,"k_index":ki,"k":core.k_path[ki],"frequencies":omega.iter().map(|w|w/std::f64::consts::TAU).collect::<Vec<_>>(),
            "permutation":permutation,"certificate_order":"sorted before the returned permutation","pass":pass,"warm_start":solved.report,"fields":fields}));
        if ki+1<k {
            paths.insert(pi,std::mem::replace(&mut context.path,PathHistory::new(true,true)));
            if bands.tracking {previous.insert(pi,(omega,solved.fields));}
        } else {context.path.clear();previous.remove(&pi);}
    }
    records.sort_by_key(|r|(r["job"].as_u64().unwrap(),r["k_index"].as_u64().unwrap()));
    println!("{}",json!({"scope":"serial schedule experiment; public-workflow measurements remain separate",
        "strategy":strategy,"tile_geometries":tile_p,"tile_points":tile_k,"material_cache_capacity":tile_p,
        "config":config,"pass":all_pass,"jobs":p,"points":k,"iterations":rounds,"roots":roots,
        "material_builds":material_builds,"material_seconds":material_seconds,"peak_conservative_payload_bytes":peak_payload,
        "payload_scope":"cache, active leases, native path frames, material residency and tracking; shared arrays charged repeatedly; excludes solver/library scratch and retained output",
        "elapsed_seconds":start.elapsed().as_secs_f64(),"records":records}));
    if !all_pass {std::process::exit(1);}
}
