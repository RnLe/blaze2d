//! Qualification of the actual sampled Maxwell pencil. No material averaging.
use serde::Serialize;
use crate::{backend::{SpectralBackend,SpectralBuffer},field::{Real,cscalar},lattice::Lattice2D,
    operators::{LinearOperator,ThetaOperator},polarization::Polarization};
use super::action::BlochActionPlan;

#[derive(Debug,Clone,PartialEq,Eq,Serialize)]
#[serde(rename_all="snake_case")]
pub enum RejectionReason { InvalidTolerance, FiberMismatch, MetricMismatch, MaterialMismatch, TensorCovarianceFailure, NonHermitianMaterial, OperatorCovarianceFailure, MassCovarianceFailure }

#[derive(Debug,Clone,Serialize)]
pub struct QualificationReport {
    pub authorized: bool,
    pub support_closed: bool,
    pub sample_permutation_exact: bool,
    pub metric_defect: f64,
    pub material_defect: f64,
    pub tensor_defect: f64,
    pub operator_defect: f64,
    pub mass_defect: f64,
    pub tolerance: f64,
    pub probe_count: usize,
    pub field_precision_bits: usize,
    pub evidence: &'static str,
    pub rejection_reasons: Vec<RejectionReason>,
}

/// An action authorized for the source/destination operators used to construct
/// it. Keep this together with those immutable material and fiber inputs; it is
/// not permission to reuse a report after changing geometry or discretization.
#[derive(Debug,Clone)]
pub struct QualifiedAction { plan: BlochActionPlan, report: QualificationReport }
impl QualifiedAction {
    pub fn plan(&self)->&BlochActionPlan {&self.plan}
    pub fn report(&self)->&QualificationReport {&self.report}
}

fn relative_defect<F:SpectralBuffer>(a:&F,b:&F)->f64 {
    let (mut diff,mut scale)=(0.0,0.0);
    for (x,y) in a.as_slice().iter().zip(b.as_slice()) {
        diff+=(x.re.to_accum()-y.re.to_accum()).powi(2)+(x.im.to_accum()-y.im.to_accum()).powi(2);
        scale+=x.re.to_accum().powi(2)+x.im.to_accum().powi(2)+y.re.to_accum().powi(2)+y.im.to_accum().powi(2);
    }
    if !diff.is_finite() || !scale.is_finite() {return f64::INFINITY;}
    (2.0*diff/scale.max(f64::MIN_POSITIVE)).sqrt()
}

/// Structural covariance of support, sample multiplication and physical
/// derivatives, checked against deterministic full-support complex probes.
/// Probes supplement the structural checks; they are not a formal norm bound.
pub fn qualify<B:SpectralBackend>(plan:BlochActionPlan,lattice:&Lattice2D,source:&mut ThetaOperator<B>,destination:&mut ThetaOperator<B>,tolerance:f64)
    -> Result<QualifiedAction,QualificationReport> {
    crate::profile_scope!("symmetry.qualify");
    let mut report=QualificationReport {authorized:false,support_closed:true,sample_permutation_exact:true,metric_defect:0.0,material_defect:0.0,tensor_defect:0.0,operator_defect:0.0,mass_defect:0.0,tolerance,probe_count:0,field_precision_bits:8*std::mem::size_of::<B::Real>(),evidence:"structural covariance with deterministic probes",rejection_reasons:vec![]};
    if !tolerance.is_finite() || tolerance<=0.0 || tolerance>=1e-3 {report.rejection_reasons.push(RejectionReason::InvalidTolerance);return Err(report);}
    let area=lattice.cell_area();
    if !lattice.a1.iter().chain(lattice.a2.iter()).all(|x|x.is_finite()) || !area.is_finite() || area<=f64::EPSILON {
        report.rejection_reasons.push(RejectionReason::MetricMismatch);return Err(report);
    }
    let reciprocal=lattice.reciprocal();
    for (theta,k) in [(&*source,plan.source()),(&*destination,plan.destination())] {
        let expected=reciprocal.fractional_to_cartesian(k);let actual=theta.bloch();
        if theta.grid().nx!=plan.grid().nx || theta.grid().ny!=plan.grid().ny || theta.grid().centered!=plan.grid().centered || theta.grid().lx!=plan.grid().lx || theta.grid().ly!=plan.grid().ly || theta.polarization()!=plan.polarization() ||
            (0..2).any(|i|(expected[i]-actual[i]).abs()>tolerance*expected[i].abs().max(1.0)) ||
            (0..2).any(|i|(theta.dielectric().reciprocal_b1()[i]-reciprocal.b1[i]).abs()>tolerance*reciprocal.b1[i].abs().max(1.0) || (theta.dielectric().reciprocal_b2()[i]-reciprocal.b2[i]).abs()>tolerance*reciprocal.b2[i].abs().max(1.0)) {
            report.rejection_reasons.push(RejectionReason::FiberMismatch);
        }
    }
    report.metric_defect=plan.operation().metric_defect(lattice);
    if !report.metric_defect.is_finite() || report.metric_defect>tolerance {report.rejection_reasons.push(RejectionReason::MetricMismatch);}
    if !report.rejection_reasons.is_empty() {return Err(report);}
    let w=plan.operation().matrix();let mut r=[[0.0;2];2];
    for j in 0..2 {let mut unit=[0.0;2];unit[j]=1.0;let f=lattice.cartesian_to_fractional(unit);let v=lattice.fractional_to_cartesian([w[0][0] as f64*f[0]+w[0][1] as f64*f[1],w[1][0] as f64*f[0]+w[1][1] as f64*f[1]]);for i in 0..2 {r[i][j]=v[i];}}
    let sd=source.dielectric();let dd=destination.dielectric();
    let tensor=|d:&crate::dielectric::Dielectric2D,j:usize|d.inv_eps_tensors().map_or([d.inv_eps()[j],0.0,0.0,d.inv_eps()[j]],|t|t[j]);
    for (j,&i) in plan.spatial_permutation().iter().enumerate() {
        let (a,b)=(sd.eps()[i],dd.eps()[j]);
        if !a.is_finite() || !b.is_finite() || a<=0.0 || b<=0.0 {report.rejection_reasons.push(RejectionReason::NonHermitianMaterial);break;}
        report.material_defect=report.material_defect.max((a-b).abs()/a.abs().max(b.abs()));
        if plan.polarization()==Polarization::TE {
            let a=tensor(sd,i);let b=tensor(dd,j);
            for t in [a,b] {if !t.iter().all(|x|x.is_finite()) || (t[1]-t[2]).abs()>tolerance*t[0].abs().max(t[3].abs()) || t[0]<=0.0 || t[3]<=0.0 || t[0]*t[3]-t[1]*t[2]<=0.0 {report.rejection_reasons.push(RejectionReason::NonHermitianMaterial);}}
            for row in 0..2 {for col in 0..2 {
                let transformed=(0..2).flat_map(|u|(0..2).map(move|v|r[row][u]*a[2*u+v]*r[col][v])).sum::<f64>();
                let scale=a.iter().chain(b.iter()).map(|x|x.abs()).fold(0.0,f64::max).max(f64::MIN_POSITIVE);
                report.tensor_defect=report.tensor_defect.max((transformed-b[2*row+col]).abs()/scale);
            }}
        }
    }
    if report.material_defect>tolerance {report.rejection_reasons.push(RejectionReason::MaterialMismatch);}
    if report.tensor_defect>tolerance {report.rejection_reasons.push(RejectionReason::TensorCovarianceFailure);}
    if !report.rejection_reasons.is_empty() {return Err(report);}
    let mut input=source.alloc_field();let mut tx=destination.alloc_field();let mut ax=source.alloc_field();let mut tax=destination.alloc_field();let mut atx=destination.alloc_field();
    for seed in 0..2 {
        for (i,v) in input.as_mut_slice().iter_mut().enumerate() {let t=(i+1+seed*19) as f64;*v=cscalar::<B::Real>((t*1.61803398875).sin(),(t*0.754877666).cos());}
        plan.apply_spatial(&input,&mut tx);
        source.apply(&input,&mut ax);plan.apply_spatial(&ax,&mut tax);destination.apply(&tx,&mut atx);
        report.operator_defect=report.operator_defect.max(relative_defect(&tax,&atx));
        source.apply_mass(&input,&mut ax);plan.apply_spatial(&ax,&mut tax);destination.apply_mass(&tx,&mut atx);
        report.mass_defect=report.mass_defect.max(relative_defect(&tax,&atx));report.probe_count+=1;
    }
    if !report.operator_defect.is_finite() || report.operator_defect>tolerance {report.rejection_reasons.push(RejectionReason::OperatorCovarianceFailure);}
    if !report.mass_defect.is_finite() || report.mass_defect>tolerance {report.rejection_reasons.push(RejectionReason::MassCovarianceFailure);}
    if !report.rejection_reasons.is_empty() {return Err(report);}
    report.authorized=true;Ok(QualifiedAction {plan,report})
}
