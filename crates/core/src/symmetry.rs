//! Symmetry helpers (Phase 2+).

use crate::lattice::Lattice2D;

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
