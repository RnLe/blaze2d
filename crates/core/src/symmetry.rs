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
    #[serde(skip)]
    pub(crate) auto_reflections: Vec<ReflectionConstraint>,
}

impl Default for SymmetryOptions {
    fn default() -> Self {
        Self {
            reflections: Vec::new(),
            auto: None,
            auto_reflections: Vec::new(),
        }
    }
}

fn default_bloch_tolerance() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoSymmetry {
    #[serde(default)]
    pub parity: Parity,
    #[serde(default = "default_bloch_tolerance")]
    pub bloch_tolerance: f64,
}

impl Default for AutoSymmetry {
    fn default() -> Self {
        Self {
            parity: Parity::Even,
            bloch_tolerance: default_bloch_tolerance(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymmetrySelection {
    pub reflections: Vec<ReflectionConstraint>,
    pub skipped_auto: usize,
}

impl SymmetrySelection {
    pub fn applied_count(&self) -> usize {
        self.reflections.len()
    }

    pub fn skipped_count(&self) -> usize {
        self.skipped_auto
    }
}

#[derive(Debug, Clone)]
pub struct SymmetryProjector {
    reflections: Vec<ReflectionConstraint>,
}

impl SymmetryProjector {
    pub fn from_reflections(reflections: &[ReflectionConstraint]) -> Option<Self> {
        if reflections.is_empty() {
            None
        } else {
            Some(Self {
                reflections: reflections.to_vec(),
            })
        }
    }

    pub fn from_options(opts: &SymmetryOptions) -> Option<Self> {
        let resolved = opts.all_reflections();
        Self::from_reflections(&resolved)
    }

    pub fn is_empty(&self) -> bool {
        self.reflections.is_empty()
    }

    pub fn len(&self) -> usize {
        self.reflections.len()
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
    pub fn disable_auto(&mut self) {
        self.auto = None;
        self.auto_reflections.clear();
    }

    pub fn resolve_with_lattice(&mut self, lattice: &Lattice2D) {
        self.auto_reflections.clear();
        if !self.reflections.is_empty() {
            return;
        }
        if let Some(auto) = &self.auto {
            self.auto_reflections = reflections_for_lattice(lattice, auto.parity.clone());
        }
    }

    pub fn all_reflections(&self) -> Vec<ReflectionConstraint> {
        let mut refs = self.reflections.clone();
        refs.extend(self.auto_reflections.iter().cloned());
        refs
    }

    pub fn selection_for_bloch(&self, bloch: [f64; 2]) -> SymmetrySelection {
        let mut refs = self.reflections.clone();
        let mut skipped_auto = 0usize;
        if let Some(auto) = &self.auto {
            let tol = auto.bloch_tolerance.max(0.0);
            for reflection in &self.auto_reflections {
                if should_apply_reflection(reflection, bloch, tol) {
                    refs.push(reflection.clone());
                } else {
                    skipped_auto += 1;
                }
            }
        } else {
            skipped_auto = self.auto_reflections.len();
        }
        SymmetrySelection {
            reflections: refs,
            skipped_auto,
        }
    }
}

fn should_apply_reflection(reflection: &ReflectionConstraint, bloch: [f64; 2], tol: f64) -> bool {
    if tol <= 0.0 {
        return true;
    }
    let (component, other) = match reflection.axis {
        ReflectionAxis::X => (bloch[0], bloch[1]),
        ReflectionAxis::Y => (bloch[1], bloch[0]),
    };
    near_axis(component, other, tol)
}

fn near_axis(component: f64, other: f64, tol: f64) -> bool {
    let comp_abs = component.abs();
    if comp_abs <= tol {
        return true;
    }
    let other_abs = other.abs();
    if other_abs <= tol {
        return comp_abs <= tol;
    }
    (comp_abs / other_abs) <= tol
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
