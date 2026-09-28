//! Experimental k-point ordering with restoration of the requested output path.
//! This changes the warm-start trajectory. It is separate from fixed-route
//! symmetry reuse, and cannot be used to attribute a gain to orbit reduction.
use crate::{
    backend::SpectralBackend,
    band_tracking::{apply_permutation, track_bands_with_frequencies},
    dielectric::Dielectric2D,
    drivers::bandstructure::{
        BandStructureJob, BandStructureResult, KPointResult, RunOptions, compute_k_path_distances,
        run_with_continuation,
    },
    field::Field2D,
    lattice::Lattice2D,
    polarization::Polarization,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KTraversal {
    #[default]
    Given,
    Nearest,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraversalReport {
    pub policy: KTraversal,
    pub solve_position: usize,
    pub previous_visited_index: Option<usize>,
    pub peak_pending_field_bytes_so_far: usize,
    pub reserved_field_payload_bytes: usize,
}

/// Deterministic nearest-neighbour route, keeping the first requested point.
/// Distances use the reciprocal metric without changing reciprocal gauges.
/// Duplicate points remain separate output samples. The O(n²) reference is
/// bounded to 4096 samples; larger jobs retain their supplied order.
pub fn nearest_order(lattice: &Lattice2D, points: &[[f64; 2]]) -> Result<Vec<usize>, String> {
    if points.len() > 4096 {
        return Err("nearest traversal is limited to 4096 requested samples".into());
    }
    if points.iter().flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite k point".into());
    }
    let reciprocal = lattice.reciprocal();
    let cartesian: Vec<_> = points
        .iter()
        .map(|&k| reciprocal.fractional_to_cartesian(k))
        .collect();
    if cartesian.iter().flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite reciprocal coordinates".into());
    }
    let mut visited = vec![false; points.len()];
    let mut order = Vec::with_capacity(points.len());
    if points.is_empty() {
        return Ok(order);
    }
    let mut current = 0;
    while order.len() < points.len() {
        visited[current] = true;
        order.push(current);
        let distance = |i: usize| {
            (cartesian[i][0] - cartesian[current][0]).hypot(cartesian[i][1] - cartesian[current][1])
        };
        if let Some(next) = (0..points.len())
            .filter(|&i| !visited[i])
            .min_by(|&a, &b| distance(a).total_cmp(&distance(b)).then(a.cmp(&b)))
        {
            current = next;
        }
    }
    Ok(order)
}

/// All rejection checks precede any solve. Ordered callbacks may be delayed by
/// out-of-order work. Fields needed for original-path tracking are buffered
/// under a separate, explicit payload budget; the numerical cache has its own
/// budget. Reordering never authorizes a new symmetry relation.
pub(crate) fn run_reordered<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    options: RunOptions,
    on_k_point: &mut Option<&mut dyn FnMut(KPointResult)>,
    continuation: Option<&mut super::continuation::BandContinuation>,
) -> Result<BandStructureResult, String> {
    if options.reuse_gamma {
        return Err("nearest traversal requires legacy closing-Gamma reuse to be disabled".into());
    }
    let order = nearest_order(&job.geom.lattice, &job.k_path)?;
    let needs_fields = options.retain_eigenvectors || !options.disable_band_tracking;
    let reserved = if needs_fields {
        order
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_mul(job.grid.len()))
            .and_then(|n| n.checked_mul(job.eigensolver.n_bands))
            .and_then(|n| n.checked_mul(16))
            .ok_or("traversal field payload size overflows")?
    } else {
        0
    };
    if reserved > options.symmetry.max_traversal_field_bytes {
        return Err(format!(
            "traversal field payload {reserved} exceeds budget {}",
            options.symmetry.max_traversal_field_bytes
        ));
    }
    crate::profile_scope!("run.traversal");
    let mut routed = job.clone();
    routed.k_path = order.iter().map(|&i| job.k_path[i]).collect();
    let mut inner = options.clone();
    inner.symmetry.k_traversal = KTraversal::Given;
    inner.disable_band_tracking = true;
    inner.retain_eigenvectors = needs_fields;
    // Sampling the tracking mass here is included in total traversal cost.
    // The inner driver owns its independent material/operator workspace.
    let tracking_mass = if !options.disable_band_tracking && job.pol == Polarization::TM {
        Some(
            Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric)
                .eps()
                .to_vec(),
        )
    } else {
        None
    };
    let distances = compute_k_path_distances(&job.k_path);
    let mut pending: Vec<Option<KPointResult>> = (0..order.len()).map(|_| None).collect();
    let mut bands = vec![Vec::new(); order.len()];
    let mut next = 0;
    let mut pending_bytes = 0;
    let mut peak_pending_bytes = 0;
    let mut previous: Option<(Vec<Field2D>, Vec<f64>)> = None;
    run_with_continuation(backend, &routed, inner, continuation, |mut point| {
        let position = point.k_index;
        let original = order[position];
        point.k_index = original;
        point.distance = distances[original];
        if let Some(source) = point.symmetry.source_index {
            point.symmetry.source_index = Some(order[source]);
        }
        point.symmetry.traversal = Some(TraversalReport {
            policy: KTraversal::Nearest,
            solve_position: position,
            previous_visited_index: position.checked_sub(1).map(|j| order[j]),
            peak_pending_field_bytes_so_far: 0,
            reserved_field_payload_bytes: reserved,
        });
        let field_bytes = point
            .eigenvectors
            .as_ref()
            .map_or(0, |v| v.iter().map(|f| f.len() * 16).sum());
        pending_bytes += field_bytes;
        peak_pending_bytes = peak_pending_bytes.max(pending_bytes);
        pending[original] = Some(point);
        while next < pending.len() && pending[next].is_some() {
            let mut point = pending[next].take().unwrap();
            pending_bytes -= point
                .eigenvectors
                .as_ref()
                .map_or(0, |v| v.iter().map(|f| f.len() * 16).sum());
            if !options.disable_band_tracking {
                let fields = point
                    .eigenvectors
                    .as_mut()
                    .expect("tracking fields requested");
                if let Some((prev, frequencies)) = &previous {
                    let tracking = track_bands_with_frequencies(
                        prev,
                        fields,
                        frequencies,
                        &point.omegas,
                        tracking_mass.as_deref(),
                    );
                    if tracking.had_swaps {
                        apply_permutation(&tracking.permutation, &mut point.omegas, fields);
                        let residuals = point.residuals.clone();
                        let absolute = point.absolute_residuals.clone();
                        point.residuals =
                            tracking.permutation.iter().map(|&i| residuals[i]).collect();
                        point.absolute_residuals =
                            tracking.permutation.iter().map(|&i| absolute[i]).collect();
                    }
                }
                previous = Some((fields.clone(), point.omegas.clone()));
            }
            bands[next] = point.omegas.clone();
            point
                .symmetry
                .traversal
                .as_mut()
                .unwrap()
                .peak_pending_field_bytes_so_far = peak_pending_bytes;
            if !options.retain_eigenvectors {
                point.eigenvectors = None;
            }
            if let Some(callback) = on_k_point.as_mut() {
                callback(point);
            }
            next += 1;
        }
    });
    assert_eq!(next, job.k_path.len());
    Ok(BandStructureResult {
        k_path: job.k_path.clone(),
        distances,
        bands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn route_keeps_duplicates_and_uses_the_oblique_reciprocal_metric() {
        let lattice = Lattice2D::hexagonal(1.0);
        let points = [[0.0, 0.0], [0.2, -0.2], [0.0, 0.0], [0.3, 0.3]];
        assert_eq!(nearest_order(&lattice, &points).unwrap(), vec![0, 2, 3, 1]);
        assert!(nearest_order(&lattice, &[[f64::NAN, 0.0]]).is_err());
        assert!(nearest_order(&lattice, &vec![[0.0; 2]; 4097]).is_err());
    }
}
