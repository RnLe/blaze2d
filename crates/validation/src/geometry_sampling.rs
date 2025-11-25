use mpb2d_core::geometry::Geometry2D;

pub struct SampledField {
    pub grid: Vec<Vec<f64>>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub fill_fraction: f64,
}

pub fn sample_geometry(
    geometry: &Geometry2D,
    resolution: usize,
    eps_inclusion: f64,
) -> SampledField {
    let mut grid = Vec::with_capacity(resolution);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut inclusion_hits = 0usize;
    let total = resolution * resolution;
    let step = 1.0 / resolution as f64;
    let offset = step * 0.5;

    for j in 0..resolution {
        let mut row = Vec::with_capacity(resolution);
        for i in 0..resolution {
            let sample = [offset + i as f64 * step, offset + j as f64 * step];
            let eps = geometry.relative_permittivity_at_fractional(sample);
            min = min.min(eps);
            max = max.max(eps);
            sum += eps;
            if (eps - eps_inclusion).abs() <= 1e-9 {
                inclusion_hits += 1;
            }
            row.push(eps);
        }
        grid.push(row);
    }

    SampledField {
        grid,
        min,
        max,
        mean: sum / total as f64,
        fill_fraction: inclusion_hits as f64 / total as f64,
    }
}
