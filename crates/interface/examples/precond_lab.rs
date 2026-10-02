//! Matrix-free access to a band job's operators for offline preconditioner studies.
//!
//! `precond_lab serve CASE.toml` reads one JSON request per line on stdin. Requests
//! with a payload are followed by `count` little-endian complex128 vectors in the
//! solver's spatial ordering (row-major ny×nx); replies are one JSON line,
//! optionally followed by a payload of the same layout.
//!
//! Requests: `info`; `set_k {index | k}`; `set_spec {spec, previous?}`;
//! `apply {which: "A"|"B"|"T", count}`; `solve {trace?, vectors?}`; `quit`.

use std::io::{BufRead, Read, Write};

use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::Eigensolver,
    field::Field2D,
    operators::{LinearOperator, ThetaOperator},
    preconditioners::{OperatorPreconditioner, PreconditionerInfo, PreconditionerSpec},
};
use blaze2d_interface::{Config, Plan, Platform};
use num_complex::Complex64;
use serde_json::{Value, json};

type Backend = CpuBackend<f64>;

struct Session {
    job: blaze2d_core::drivers::bandstructure::BandStructureJob,
    dielectric: Dielectric2D,
    spec: PreconditionerSpec,
    previous: Option<Vec<f64>>,
    k: [f64; 2],
    operator: Option<ThetaOperator<Backend>>,
    preconditioner: Option<Box<dyn OperatorPreconditioner<Backend>>>,
    info: Option<PreconditionerInfo>,
}

impl Session {
    fn rebuild(&mut self) -> Value {
        let bloch = self.job.geom.lattice.reciprocal().fractional_to_cartesian(self.k);
        let operator = ThetaOperator::new(Backend::new(), self.dielectric.clone(), self.job.pol, bloch);
        let (preconditioner, info) = operator.build_preconditioner(&self.spec, self.previous.as_deref());
        let (kx, ky) = operator.k_plus_g_components();
        let reply = json!({
            "k": self.k, "bloch": bloch, "is_gamma": operator.analytic_kernel_available(),
            "k_plus_g_x": kx, "k_plus_g_y": ky, "kernel_mask": operator.k_plus_g_clamp_mask(),
            "preconditioner": info,
        });
        self.preconditioner = preconditioner;
        self.info = Some(info);
        self.operator = Some(operator);
        reply
    }
}

fn read_payload(input: &mut impl Read, count: usize, grid: blaze2d_core::grid::Grid2D) -> std::io::Result<Vec<Field2D>> {
    let n = grid.len();
    let mut bytes = vec![0u8; count * n * 16];
    input.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(n * 16).map(|chunk| {
        let data = chunk.chunks_exact(16).map(|c| Complex64::new(
            f64::from_le_bytes(c[..8].try_into().unwrap()), f64::from_le_bytes(c[8..].try_into().unwrap()))).collect();
        Field2D::from_vec(grid, data)
    }).collect())
}

fn write_reply(out: &mut impl Write, header: &Value, payload: &[Field2D]) -> std::io::Result<()> {
    writeln!(out, "{header}")?;
    for field in payload {
        for z in field.as_slice() {
            out.write_all(&z.re.to_le_bytes())?;
            out.write_all(&z.im.to_le_bytes())?;
        }
    }
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "serve" {
        eprintln!("usage: precond_lab serve CASE.toml");
        std::process::exit(2);
    }
    let plan = Plan::new(Config::from_toml(&std::fs::read_to_string(&args[2])?)?, Platform::Native)?;
    let planned = plan.job(0)?;
    let job = planned.bands_job();
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);
    let mut session = Session {
        spec: planned.preconditioner(), previous: None, k: job.k_path[0], operator: None,
        preconditioner: None, info: None, dielectric, job,
    };
    session.rebuild();
    let grid = session.job.grid;

    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let request: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => { write_reply(&mut out, &json!({"ok": false, "error": e.to_string()}), &[])?; continue; }
        };
        match request["op"].as_str().unwrap_or("") {
            "quit" => break,
            "info" => {
                let reply = json!({"ok": true, "nx": grid.nx, "ny": grid.ny, "polarization": format!("{:?}", session.job.pol),
                    "bands": session.job.eigensolver.n_bands, "k_points": session.job.k_path,
                    "eigensolver": session.job.eigensolver, "spec": session.spec,
                    "lattice": {"a1": session.job.geom.lattice.a1, "a2": session.job.geom.lattice.a2},
                    "eps": session.dielectric.eps(), "inv_eps": session.dielectric.inv_eps(),
                    "inv_eps_tensors": session.dielectric.inv_eps_tensors()});
                write_reply(&mut out, &reply, &[])?;
            }
            "set_k" => {
                session.k = if let Some(i) = request["index"].as_u64() {
                    session.job.k_path[i as usize]
                } else {
                    serde_json::from_value(request["k"].clone())?
                };
                let mut reply = session.rebuild();
                reply["ok"] = json!(true);
                write_reply(&mut out, &reply, &[])?;
            }
            "set_spec" => {
                session.spec = serde_json::from_value(request["spec"].clone())?;
                session.previous = serde_json::from_value(request["previous"].clone()).ok();
                let mut reply = session.rebuild();
                reply["ok"] = json!(true);
                write_reply(&mut out, &reply, &[])?;
            }
            "apply" => {
                let count = request["count"].as_u64().unwrap_or(0) as usize;
                let inputs = read_payload(&mut input, count, grid)?;
                let operator = session.operator.as_mut().unwrap();
                let mut outputs = Vec::with_capacity(count);
                for v in inputs {
                    let mut x = operator.alloc_field();
                    x.as_mut_slice().copy_from_slice(v.as_slice());
                    let mut y = operator.alloc_field();
                    match request["which"].as_str() {
                        Some("A") => operator.apply(&x, &mut y),
                        Some("B") => operator.apply_mass(&x, &mut y),
                        Some("T") => {
                            y = x;
                            if let Some(p) = session.preconditioner.as_mut() {
                                p.apply(operator.backend(), &mut y);
                            }
                        }
                        _ => { write_reply(&mut out, &json!({"ok": false, "error": "which must be A, B or T"}), &[])?; outputs.clear(); break; }
                    }
                    outputs.push(y);
                }
                if outputs.len() == count {
                    write_reply(&mut out, &json!({"ok": true, "count": count}), &outputs)?;
                }
            }
            "solve" => {
                let mut config = session.job.eigensolver.clone();
                config.trace = request["trace"].as_bool().unwrap_or(false);
                let vectors = request["vectors"].as_bool().unwrap_or(false);
                let work_start = blaze2d_core::work::snapshot();
                let operator = session.operator.as_mut().unwrap();
                let mut solver = Eigensolver::new(operator, config,
                    session.preconditioner.as_mut().map(|p| &mut **p as &mut dyn OperatorPreconditioner<Backend>), None);
                let result = solver.solve();
                let fields = if vectors { solver.all_eigenvectors() } else { Vec::new() };
                let reply = json!({"ok": true, "eigenvalues": result.eigenvalues, "iterations": result.iterations,
                    "converged": result.converged, "trace": result.trace, "count": fields.len(),
                    "work": blaze2d_core::work::snapshot().since(work_start), "preconditioner": session.info});
                write_reply(&mut out, &reply, &fields)?;
            }
            other => write_reply(&mut out, &json!({"ok": false, "error": format!("unknown op {other}")}), &[])?,
        }
    }
    Ok(())
}
