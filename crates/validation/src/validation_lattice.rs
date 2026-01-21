use crate::validation_utils::{classify_to_string, determinant, dot};
use clap::{Args, ValueEnum};
use mpb2d_core::lattice::Lattice2D;
use serde::Serialize;
use std::{f64::consts::TAU, fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct LatticeArgs {
    /// Lattice preset to validate.
    #[arg(long, value_enum)]
    pub lattice: LatticeKind,
    /// Characteristic length a (in lattice units).
    #[arg(long, default_value_t = 1.0)]
    pub a: f64,
    /// Optional output path for the generated JSON. Defaults to stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Tolerance for a_i · b_j ≈ 2π δ_ij.
    #[arg(long, default_value_t = 1e-9)]
    pub dot_tol: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum LatticeKind {
    Square,
    Triangular,
}

impl LatticeKind {
    pub fn build(self, a: f64) -> Lattice2D {
        match self {
            Self::Square => Lattice2D::square(a),
            Self::Triangular => Lattice2D::hexagonal(a),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Square => "square",
            Self::Triangular => "triangular",
        }
    }
}

#[derive(Serialize)]
pub struct LatticeReport {
    lattice: String,
    scale_a: f64,
    real_space: Basis,
    reciprocal_space: Basis,
    classification: String,
    cell_area: f64,
    reciprocal_area: f64,
    dot_products: DotMatrix,
    validation: ValidationSummary,
}

#[derive(Serialize)]
pub struct Basis {
    v1: [f64; 2],
    v2: [f64; 2],
}

#[derive(Serialize)]
pub struct DotMatrix {
    expected: [[f64; 2]; 2],
    actual: [[f64; 2]; 2],
    error: [[f64; 2]; 2],
}

#[derive(Serialize)]
pub struct ValidationSummary {
    target: String,
    tolerance: f64,
    max_abs_error: f64,
    passed: bool,
}

pub fn run(args: LatticeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !(args.a.is_finite() && args.a > 0.0) {
        return Err("characteristic length must be positive and finite".into());
    }

    let lattice = args.lattice.build(args.a);
    let reciprocal = lattice.reciprocal();

    let actual = compute_dot_products(&lattice, &reciprocal);
    let expected = [[TAU, 0.0], [0.0, TAU]];
    let mut error = [[0.0; 2]; 2];
    let mut max_abs_error: f64 = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            error[i][j] = actual[i][j] - expected[i][j];
            max_abs_error = max_abs_error.max(error[i][j].abs());
        }
    }
    let passed = max_abs_error <= args.dot_tol;

    let report = LatticeReport {
        lattice: args.lattice.as_str().to_string(),
        scale_a: args.a,
        real_space: Basis {
            v1: lattice.a1,
            v2: lattice.a2,
        },
        reciprocal_space: Basis {
            v1: reciprocal.b1,
            v2: reciprocal.b2,
        },
        classification: classify_to_string(lattice.classify()),
        cell_area: determinant(lattice.a1, lattice.a2).abs(),
        reciprocal_area: determinant(reciprocal.b1, reciprocal.b2).abs(),
        dot_products: DotMatrix {
            expected,
            actual,
            error,
        },
        validation: ValidationSummary {
            target: "a_i·b_j = 2π δ_ij".to_string(),
            tolerance: args.dot_tol,
            max_abs_error,
            passed,
        },
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json)?;
        eprintln!("Saved lattice validation report to {}", path.display());
    } else {
        println!("{}", json);
    }

    if passed {
        Ok(())
    } else {
        Err(format!(
            "dot-product invariants violated (max abs error = {:.3e} > {:.3e})",
            max_abs_error, args.dot_tol
        )
        .into())
    }
}

fn compute_dot_products(
    lattice: &Lattice2D,
    reciprocal: &mpb2d_core::lattice::ReciprocalLattice2D,
) -> [[f64; 2]; 2] {
    [
        [
            dot(lattice.a1, reciprocal.b1),
            dot(lattice.a1, reciprocal.b2),
        ],
        [
            dot(lattice.a2, reciprocal.b1),
            dot(lattice.a2, reciprocal.b2),
        ],
    ]
}
