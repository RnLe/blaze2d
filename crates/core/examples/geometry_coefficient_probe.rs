//! Sampled-coefficient step-halving, including interface events and plateaus.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions, SmoothingMethod},
    geometry::{BasisAtom, Geometry2D},
    geometry_warm_start::difference::DiscreteDifference,
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
};
use serde_json::json;
fn material(
    n: usize,
    hex: bool,
    method: &str,
    radius: f64,
    pol: Polarization,
) -> ThetaOperator<CpuBackend<f64>> {
    let mut options = DielectricOptions::default();
    options.smoothing.method = if method == "voronoi" {
        SmoothingMethod::Voronoi
    } else {
        SmoothingMethod::Analytic
    };
    if method == "none" {
        options.smoothing.mesh_size = 1;
    }
    let d = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice: if hex {
                Lattice2D::hexagonal(1.)
            } else {
                Lattice2D::square(1.)
            },
            eps_bg: 1.,
            atoms: vec![BasisAtom {
                pos: [0., 0.],
                radius,
                eps_inside: 8.9,
            }],
        },
        Grid2D::new(n, n, 1., 1.),
        &options,
    );
    ThetaOperator::new(CpuBackend::<f64>::new(), d, pol, [0., 0.])
}
fn coefficients(d: &DiscreteDifference, pol: Polarization) -> Vec<f64> {
    if pol == Polarization::TM {
        d.delta_epsilon().to_vec()
    } else {
        d.delta_inverse_tensor().iter().flatten().copied().collect()
    }
}
fn norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |n, x| n.hypot(*x))
}
fn main() {
    let mut rows = Vec::new();
    for n in [16, 32, 64] {
        for hex in [false, true] {
            for pol in [Polarization::TM, Polarization::TE] {
                for method in ["none", "analytic", "voronoi"] {
                    for radius in [0.203, 0.25] {
                        let source = material(n, hex, method, radius, pol);
                        let mut samples = Vec::new();
                        let mut derivatives = Vec::new();
                        for level in 0..9 {
                            let h = 0.004 / 2_f64.powi(level);
                            let plus = material(n, hex, method, radius + h, pol);
                            let minus = material(n, hex, method, radius - h, pol);
                            let forward = DiscreteDifference::new(&source, &plus).unwrap();
                            let central = DiscreteDifference::new(&minus, &plus).unwrap();
                            let derivative: Vec<_> = coefficients(&central, pol)
                                .into_iter()
                                .map(|x| x / (2. * h))
                                .collect();
                            samples.push(json!({"step":h,"forward_increment_norm":norm(&coefficients(&forward,pol)),
                    "central_derivative_norm":norm(&derivative),"coefficient_change":forward.report}));
                            derivatives.push(derivative);
                        }
                        let errors: Vec<_> = derivatives
                            .windows(2)
                            .map(|pair| {
                                norm(
                                    &pair[0]
                                        .iter()
                                        .zip(&pair[1])
                                        .map(|(a, b)| a - b)
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect();
                        let orders: Vec<_> = errors
                            .windows(2)
                            .map(|e| {
                                if e[0] > 1e-13 && e[1] > 1e-13 {
                                    Some((e[0] / e[1]).log2())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        rows.push(json!({"grid":n,"lattice":if hex {"triangular"} else {"square"},"polarization":format!("{pol:?}"),
                "smoothing":method,"radius":radius,"samples":samples,"central_successive_difference_norm":errors,
                "central_successive_order":orders}));
                    }
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"finite sampled coefficients; no solver timing or response-order claim",
        "derivative_parameter":"circle radius in units of lattice constant 1","rows":rows})
        )
        .unwrap()
    );
}
