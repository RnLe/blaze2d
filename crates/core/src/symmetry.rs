//! Symmetry helpers (Phase 2+).

use serde::{Deserialize, Serialize};

use crate::{
    backend::SpectralBuffer,
    lattice::{Lattice2D, LatticeClass},
};

#[derive(Debug, Clone)]
pub enum PathType {
    Square,
    Hexagonal,
    Custom(Vec<[f64; 2]>),
}

pub fn standard_path(
    lattice: &Lattice2D,
    path: PathType,
    segments_per_leg: usize,
) -> Vec<[f64; 2]> {
    let _ = lattice;
    match path {
        PathType::Custom(seq) => seq,
        PathType::Square => densify_path(&SQUARE_GXMG, segments_per_leg),
        PathType::Hexagonal => densify_path(&HEX_GMK, segments_per_leg),
    }
}

const SQUARE_GXMG: [[f64; 2]; 4] = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]];
const HEX_GMK: [[f64; 2]; 4] = [[0.0, 0.0], [0.5, 0.0], [1.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]];

fn densify_path(nodes: &[[f64; 2]], segments_per_leg: usize) -> Vec<[f64; 2]> {
    if nodes.len() <= 1 {
        return nodes.to_vec();
    }
    let segments = segments_per_leg.max(1);
    let mut path = Vec::with_capacity(nodes.len() * segments);
    path.push(nodes[0]);
    for window in nodes.windows(2) {
        let start = window[0];
        let end = window[1];
        for step in 1..=segments {
            let t = step as f64 / segments as f64;
            let point = [
                (1.0 - t) * start[0] + t * end[0],
                (1.0 - t) * start[1] + t * end[1],
            ];
            if path.last().map(|last| last != &point).unwrap_or(true) {
                path.push(point);
            }
        }
    }
    path
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReflectionAxis {
    X,
    Y,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Parity {
    Even,
    Odd,
}

impl Default for Parity {
    fn default() -> Self {
        Parity::Even
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionConstraint {
    pub axis: ReflectionAxis,
    pub parity: Parity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetryOptions {
    pub reflections: Vec<ReflectionConstraint>,
    pub auto: Option<AutoSymmetry>,
}

impl Default for SymmetryOptions {
    fn default() -> Self {
        Self {
            reflections: Vec::new(),
            auto: Some(AutoSymmetry::default()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutoSymmetry {
    #[serde(default)]
    pub parity: Parity,
}

#[derive(Debug, Clone)]
pub struct SymmetryProjector {
    reflections: Vec<ReflectionConstraint>,
}

impl SymmetryProjector {
    pub fn from_options(opts: &SymmetryOptions) -> Option<Self> {
        if opts.reflections.is_empty() {
            None
        } else {
            Some(Self {
                reflections: opts.reflections.clone(),
            })
        }
    }

    pub fn is_empty(&self) -> bool {
        self.reflections.is_empty()
    }

    pub fn apply<B: SpectralBuffer>(&self, buffer: &mut B) {
        if self.is_empty() {
            return;
        }
        for reflection in &self.reflections {
            apply_reflection(buffer, reflection);
        }
    }
}

fn apply_reflection<B: SpectralBuffer>(buffer: &mut B, reflection: &ReflectionConstraint) {
    let grid = buffer.grid();
    let original = buffer.as_slice().to_vec();
    let data = buffer.as_mut_slice();
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let (mirror_ix, mirror_iy) = match reflection.axis {
                ReflectionAxis::X => (mirror_index(grid.nx, ix), iy),
                ReflectionAxis::Y => (ix, mirror_index(grid.ny, iy)),
            };
            let mirror_idx = grid.idx(mirror_ix, mirror_iy);
            let pair_value = match reflection.parity {
                Parity::Even => (original[idx] + original[mirror_idx]) * 0.5,
                Parity::Odd => (original[idx] - original[mirror_idx]) * 0.5,
            };
            data[idx] = pair_value;
        }
    }
}

fn mirror_index(len: usize, idx: usize) -> usize {
    if len == 0 { 0 } else { (len - idx) % len }
}

impl SymmetryOptions {
    pub fn resolve_with_lattice(&mut self, lattice: &Lattice2D) {
        if !self.reflections.is_empty() {
            return;
        }
        if let Some(auto) = &self.auto {
            self.reflections = reflections_for_lattice(lattice, auto.parity.clone());
        }
    }
}

pub fn reflections_for_lattice(lattice: &Lattice2D, parity: Parity) -> Vec<ReflectionConstraint> {
    match lattice.classify() {
        LatticeClass::Square | LatticeClass::Rectangular => vec![
            ReflectionConstraint {
                axis: ReflectionAxis::X,
                parity: parity.clone(),
            },
            ReflectionConstraint {
                axis: ReflectionAxis::Y,
                parity,
            },
        ],
        LatticeClass::Triangular => vec![
            ReflectionConstraint {
                axis: ReflectionAxis::X,
                parity: parity.clone(),
            },
            ReflectionConstraint {
                axis: ReflectionAxis::Y,
                parity,
            },
        ],
        LatticeClass::Oblique => Vec::new(),
    }
}
