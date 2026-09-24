//! Fixed-pencil sector benchmark. Full fields and lossless operator identities
//! support an independent audit; export and optional references are untimed.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    drivers::single_solve::{SingleSolveJob, solve as full_solve},
    eigensolver::{EigensolverConfig, refine::certify_without_refinement},
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        cache::SnapshotIdentity,
        galerkin::{self, GalerkinApplication},
        support::FourierSupport,
        support_sectors::{self, SupportSectorOptions},
    },
};
use serde_json::{Value, json};
use std::{f64::consts::TAU, io::Write, time::Instant};

struct Fixture {
    lattice: Lattice2D,
    grid: Grid2D,
    k: [f64; 2],
    group: PlaneGroup,
    analytic_hex: bool,
}
impl Fixture {
    fn new(case: &str, n: usize) -> Self {
        let (group, k) = match case {
            "hex-gamma-c3" => (PlaneGroup::P3, [0.0; 2]),
            "hex-gamma-c6" => (PlaneGroup::P6, [0.0; 2]),
            "hex-gamma-c6v" | "hex-gamma-analytic" => (PlaneGroup::P6m, [0.0; 2]),
            "hex-k" => (PlaneGroup::P6m, [2.0 / 3.0, 1.0 / 3.0]),
            "chiral-c4" => (PlaneGroup::P4, [0.0; 2]),
            "glide-pg" => (PlaneGroup::Pg, [0.5, 0.0]),
            "glide-pgg" => (PlaneGroup::Pgg, [0.5, 0.0]),
            _ => panic!("unknown fixture"),
        };
        Self {
            lattice: if case.starts_with("hex-") {
                Lattice2D::hexagonal(1.0)
            } else {
                Lattice2D::square(1.0)
            },
            grid: Grid2D::new(n, n, 1.0, 1.0),
            k,
            group,
            analytic_hex: case == "hex-gamma-analytic",
        }
    }
    fn operator(&self, pol: Polarization) -> ThetaOperator<CpuBackend<f64>> {
        let reciprocal = self.lattice.reciprocal();
        let d = if matches!(self.group, PlaneGroup::Pg | PlaneGroup::Pgg) {
            let n = self.grid.nx;
            let eps = (0..self.grid.len())
                .map(|i| {
                    let x = TAU * (i % n) as f64 / n as f64;
                    let y = TAU * (i / n) as f64 / n as f64;
                    3.0 + 0.6 * x.sin() * y.sin() + 0.3 * (2.0 * x).cos() * (2.0 * y).cos()
                })
                .collect();
            Dielectric2D::from_sampled_epsilon(self.grid, reciprocal.b1, reciprocal.b2, eps, None)
        } else {
            let atoms = if self.group == PlaneGroup::P4 {
                let mut pos = [0.23, 0.11];
                (0..4)
                    .map(|_| {
                        let atom = BasisAtom {
                            pos,
                            radius: 0.08,
                            eps_inside: 8.9,
                        };
                        pos = [-pos[1], pos[0]];
                        atom
                    })
                    .collect()
            } else {
                vec![BasisAtom {
                    pos: [0.0; 2],
                    radius: 0.2,
                    eps_inside: 8.9,
                }]
            };
            let mut options = DielectricOptions::default();
            if self.group != PlaneGroup::P4 && !self.analytic_hex {
                options.smoothing.mesh_size = 1;
            }
            Dielectric2D::from_geometry(
                &Geometry2D {
                    lattice: self.lattice,
                    eps_bg: 1.0,
                    atoms,
                },
                self.grid,
                &options,
            )
        };
        ThetaOperator::new(
            CpuBackend::new(),
            d,
            pol,
            reciprocal.fractional_to_cartesian(self.k),
        )
    }
}

fn bits(fields: &[Field2D]) -> Vec<Vec<[u64; 2]>> {
    fields
        .iter()
        .map(|f| {
            f.as_slice()
                .iter()
                .map(|z| [z.re.to_bits(), z.im.to_bits()])
                .collect()
        })
        .collect()
}
fn encoded(result: &galerkin::GalerkinResult) -> Value {
    json!({"spectrum":result.eigenvalues, "report":result.report,
        "support_identity":result.support_identity,
        "field_bits":bits(&result.fields), "field_count":result.fields.len(),
        "field_payload_bytes":result.fields.iter().map(|f|f.as_slice().len()*16).sum::<usize>()})
}

fn references(
    fixture: &Fixture,
    support: &FourierSupport,
    pol: Polarization,
    config: &EigensolverConfig,
) -> Value {
    let mut operator = fixture.operator(pol);
    let mut preconditioner = operator.build_homogeneous_preconditioner_adaptive();
    let reference_config = EigensolverConfig {
        n_bands: (config.n_bands + 8).min(support.len()),
        ..config.clone()
    };
    let projected = galerkin::solve_iterative(
        &mut operator,
        support,
        &reference_config,
        None,
        Some(&mut preconditioner),
        GalerkinApplication::DirectFourier,
        Default::default(),
    );
    let projected = match projected {
        Ok(r) => encoded(&r),
        Err(e) => json!({"error":e}),
    };
    drop(preconditioner);
    drop(operator);
    let mut operator = fixture.operator(pol);
    let mut preconditioner = operator.build_homogeneous_preconditioner_adaptive();
    let q = (config.n_bands + 8).min(fixture.grid.len());
    let full = full_solve(
        &mut operator,
        Some(&mut preconditioner),
        &SingleSolveJob::new(q)
            .with_tolerance(config.tol)
            .with_max_iterations(config.max_iter),
    );
    let cert = certify_without_refinement(&mut operator, &full.eigenvalues, &full.eigenvectors);
    json!({"projected":projected, "full_rectangle":{
        "spectrum":full.eigenvalues, "solver_converged":full.converged, "iterations":full.iterations,
        "certificate":{"residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,
            "b_orthogonality_defect":cert.b_orthogonality_defect},
        "field_bits":bits(&full.eigenvectors), "bands":q},
        "scope":"supplementary guarded same-pencil and full-rectangle references; outside all primary clocks, RSS and counters"})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |i, default| args.get(i).map(String::as_str).unwrap_or(default);
    let case = get(1, "hex-gamma-c6v");
    let n: usize = get(2, "33").parse().unwrap();
    let bands: usize = get(3, "8").parse().unwrap();
    let pol = match get(4, "TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("unknown polarization"),
    };
    let mode = get(5, "adaptive");
    let radius: f64 = get(6, "7").parse().unwrap();
    let with_references = get(7, "no-references") == "references";
    let config = EigensolverConfig {
        n_bands: bands,
        tol: get(8, "1e-14").parse().unwrap(),
        max_iter: get(9, "1000").parse().unwrap(),
        ..Default::default()
    };
    let mut options = SupportSectorOptions::default();
    match mode {
        "unreduced" | "adaptive" => {}
        "conservative" => options.adaptive_prefixes = false,
        "isotypic" => options.reconstruct_partners = false,
        "fixed-guards" => options.total_guard_columns = None,
        _ => panic!("unknown solve policy"),
    }
    let start = Instant::now();
    let fixture = Fixture::new(case, n);
    let mut operator = fixture.operator(pol);
    let identity = SnapshotIdentity::from_operator(&operator);
    let support = FourierSupport::metric_ball(
        fixture.grid,
        fixture.lattice,
        fixture.k,
        TAU * radius,
        Default::default(),
    )
    .unwrap();
    let mut preconditioner = operator.build_homogeneous_preconditioner_adaptive();
    let (result, sectors, error) = if mode == "unreduced" {
        match galerkin::solve_iterative(
            &mut operator,
            &support,
            &config,
            None,
            Some(&mut preconditioner),
            GalerkinApplication::DirectFourier,
            options.budget,
        ) {
            Ok(r) => (Some(r), None, None),
            Err(e) => (None, None, Some(e)),
        }
    } else {
        let group = fixture
            .group
            .group()
            .little_group(fixture.k, 1e-12)
            .unwrap();
        match support_sectors::solve(
            &mut operator,
            &support,
            group,
            &config,
            None,
            Some(&mut preconditioner),
            options,
        ) {
            Ok(r) => (Some(r.spectrum), Some(r.report), None),
            Err(e) => (None, None, Some(e)),
        }
    };
    drop(preconditioner);
    drop(operator);
    let seconds = start.elapsed().as_secs_f64();
    let peak_rss_kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u64>().ok())
        });
    let counters = blaze2d_core::profiler::get_counters();
    let stages = blaze2d_core::profiler::get_stage_counters();
    let profile: Value = serde_json::from_str(&blaze2d_core::profiler::get_profile_json()).unwrap();
    // Only lossless serialization follows this clock. Full fields remain live;
    // supplementary references cannot contaminate candidate peak RSS or work.
    let export_start = Instant::now();
    let numerics = json!({"case":case,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),
        "mode":mode,"radius":radius,"precision":"f64","eigensolver":config,
        "operator_identity":identity,"support_identity":support.identity(),
        "sector_options":if mode=="unreduced" {Value::Null} else {serde_json::to_value(options).unwrap()},
        "result":result.as_ref().map(encoded),"sectors":sectors,"error":error});
    let numerical_json = serde_json::to_string(&numerics).unwrap();
    let serialization_seconds = export_start.elapsed().as_secs_f64();
    let reference = if with_references {
        references(&fixture, &support, pol, &config)
    } else {
        Value::Null
    };
    let metadata = json!({"schema":"blaze2d/support-benchmark/1","seconds":seconds,"peak_rss_kib":peak_rss_kib,
        "counters":counters,"stage_counters":stages,"profile":profile,
        "serialization_seconds":serialization_seconds,"numerical_json_bytes":numerical_json.len(),
        "references":reference,"scope":"setup, qualification, basis, all solves and certificates, full field retention, operator teardown; f64; export and references separate"});
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    writeln!(
        out,
        "{{\"numerics\":{numerical_json},\"measurement\":{metadata}}}"
    )
    .unwrap();
}
