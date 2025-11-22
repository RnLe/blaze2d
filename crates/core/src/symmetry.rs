//! Symmetry helpers (Phase 2+).

use crate::lattice::Lattice2D;

pub enum PathType {
    Square,
    Hexagonal,
    Custom(Vec<[f64; 2]>),
}

pub fn standard_path(lattice: &Lattice2D, path: PathType, samples: usize) -> Vec<[f64; 2]> {
    let _ = lattice;
    let mut pts = Vec::new();
    match path {
        PathType::Square => {
            pts.push([0.0, 0.0]);
            pts.push([0.5, 0.0]);
            pts.push([0.5, 0.5]);
            pts.push([0.0, 0.0]);
        }
        PathType::Hexagonal => {
            pts.push([0.0, 0.0]);
            pts.push([0.5, 0.0]);
            pts.push([1.0 / 3.0, 1.0 / 3.0]);
            pts.push([0.0, 0.0]);
        }
        PathType::Custom(seq) => return seq,
    }
    if samples <= pts.len() {
        return pts;
    }
    // Simple interpolation placeholder.
    let mut densified = Vec::new();
    for win in pts.windows(2) {
        let start = win[0];
        let end = win[1];
        for t in 0..samples.max(2) {
            let alpha = t as f64 / (samples.max(2) - 1) as f64;
            densified.push([
                (1.0 - alpha) * start[0] + alpha * end[0],
                (1.0 - alpha) * start[1] + alpha * end[1],
            ]);
        }
    }
    densified
}
