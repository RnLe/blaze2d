use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedConfig {
    pub config: Config,
    /// Direct lattice vectors, stored as rows for serialization.
    pub lattice_vectors: [[f64; 2]; 2],
    pub resolution: [usize; 2],
    pub k_points_fractional: Vec<[f64; 2]>,
    pub k_points_cartesian: Vec<[f64; 2]>,
    pub distances: Vec<f64>,
    pub k_labels: Vec<String>,
    pub k_label_indices: Vec<usize>,
    pub solved_bands: usize,
}

pub(crate) fn invalid(path: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("invalid_value", path, message)
}

pub(crate) fn positive(value: f64, path: &str) -> InterfaceResult<()> {
    if !value.is_finite() || value <= 0.0 { Err(invalid(path, "must be finite and greater than zero")) }
    else { Ok(()) }
}

pub(crate) fn point(value: &[f64], path: &str) -> InterfaceResult<[f64; 2]> {
    if value.len() != 2 || value.iter().any(|v| !v.is_finite()) {
        Err(invalid(path, "expected two finite coordinates"))
    } else { Ok([value[0], value[1]]) }
}

pub fn lattice_vectors(lattice: &Lattice) -> InterfaceResult<[[f64; 2]; 2]> {
    let vectors = match lattice.kind {
        LatticeKind::Custom => {
            if lattice.a.is_some() || lattice.b.is_some() || lattice.angle_deg.is_some() {
                return Err(invalid("geometry.lattice", "custom vectors cannot be combined with a, b, or angle_deg"));
            }
            let v = lattice.vectors.as_ref().ok_or_else(|| invalid("geometry.lattice.vectors", "required for a custom lattice"))?;
            if v.len() != 2 { return Err(invalid("geometry.lattice.vectors", "expected two direct lattice vectors")); }
            [point(&v[0], "geometry.lattice.vectors[0]")?, point(&v[1], "geometry.lattice.vectors[1]")?]
        }
        kind => {
            if lattice.vectors.is_some() { return Err(invalid("geometry.lattice", "choose a preset or custom vectors")); }
            let a = lattice.a.unwrap_or(1.0);
            positive(a, "geometry.lattice.a")?;
            if matches!(kind, LatticeKind::Square | LatticeKind::Triangular) && (lattice.b.is_some() || lattice.angle_deg.is_some()) {
                return Err(invalid("geometry.lattice", "this preset accepts only a"));
            }
            match kind {
                LatticeKind::Square => [[a, 0.0], [0.0, a]],
                LatticeKind::Triangular => [[a, 0.0], [0.5 * a, 3.0_f64.sqrt() * 0.5 * a]],
                LatticeKind::Rectangular | LatticeKind::Oblique => {
                    let b = lattice.b.ok_or_else(|| invalid("geometry.lattice.b", "required for this lattice"))?;
                    positive(b, "geometry.lattice.b")?;
                    if kind == LatticeKind::Rectangular && lattice.angle_deg.is_some() {
                        return Err(invalid("geometry.lattice.angle_deg", "not used by a rectangular lattice"));
                    }
                    let angle = if kind == LatticeKind::Rectangular { 90.0 }
                        else { lattice.angle_deg.ok_or_else(|| invalid("geometry.lattice.angle_deg", "required for an oblique lattice"))? };
                    if !angle.is_finite() || angle <= 0.0 || angle >= 180.0 {
                        return Err(invalid("geometry.lattice.angle_deg", "must be between 0 and 180 degrees"));
                    }
                    [[a, 0.0], [b * angle.to_radians().cos(), b * angle.to_radians().sin()]]
                }
                LatticeKind::Custom => unreachable!(),
            }
        }
    };
    let [u, v] = vectors;
    let scale = u[0].hypot(u[1]) * v[0].hypot(v[1]);
    let determinant = u[0] * v[1] - u[1] * v[0];
    if !scale.is_finite() || scale == 0.0 || !determinant.is_finite() || determinant.abs() <= 1e-12 * scale {
        return Err(invalid("geometry.lattice", "vectors must be finite and linearly independent"));
    }
    Ok(vectors)
}

pub fn reciprocal_vectors(a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let scale = std::f64::consts::TAU / determinant;
    [[scale * a[1][1], -scale * a[1][0]], [-scale * a[0][1], scale * a[0][0]]]
}

pub fn fractional_to_cartesian(q: [f64; 2], a: [[f64; 2]; 2]) -> [f64; 2] {
    let b = reciprocal_vectors(a);
    [q[0] * b[0][0] + q[1] * b[1][0], q[0] * b[0][1] + q[1] * b[1][1]]
}

pub fn cartesian_to_fractional(k: [f64; 2], a: [[f64; 2]; 2]) -> [f64; 2] {
    [(a[0][0] * k[0] + a[0][1] * k[1]) / std::f64::consts::TAU,
     (a[1][0] * k[0] + a[1][1] * k[1]) / std::f64::consts::TAU]
}

impl Config {
    /// Resolve schema defaults without expanding parameter sweeps.
    pub fn resolve(&self) -> InterfaceResult<ResolvedConfig> {
        if self.schema != CONFIG_SCHEMA {
            return Err(Diagnostic::new("unsupported_schema", "schema", format!("use schema = \"{CONFIG_SCHEMA}\"; legacy TOML is not supported")));
        }
        if self.dimension != 2 { return Err(Diagnostic::new("unsupported_dimension", "dimension", "this release supports two-dimensional calculations")); }
        let mut config = self.clone();
        let lattice_vectors = lattice_vectors(&config.geometry.lattice)?;
        if config.geometry.lattice.kind != LatticeKind::Custom {
            config.geometry.lattice.a.get_or_insert(1.0);
        }
        positive(config.geometry.background_epsilon, "geometry.background_epsilon")?;
        let mut names = HashSet::new();
        for (index, object) in config.geometry.objects.iter_mut().enumerate() {
            let p = format!("geometry.objects[{index}]");
            if object.name.is_empty() || !object.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                return Err(invalid(&format!("{p}.name"), "use a nonempty name containing letters, digits, underscores, or hyphens"));
            }
            if !names.insert(object.name.clone()) { return Err(invalid(&format!("{p}.name"), "object names must be unique")); }
            point(&object.center, &format!("{p}.center"))?;
            object.center.iter_mut().for_each(|v| *v = v.rem_euclid(1.0));
            positive(object.radius, &format!("{p}.radius"))?;
            positive(object.epsilon, &format!("{p}.epsilon"))?;
        }
        let resolution = match &config.grid.resolution {
            Resolution::Uniform(n) => [*n, *n],
            Resolution::Axes(v) if v.len() == 2 => [v[0], v[1]],
            _ => return Err(invalid("grid.resolution", "expected an integer or [nx, ny]")),
        };
        if resolution.iter().any(|&n| n < 4) { return Err(invalid("grid.resolution", "each axis must contain at least four samples")); }
        let grid_size = resolution[0].checked_mul(resolution[1]).ok_or_else(|| invalid("grid.resolution", "grid size overflows"))?;
        config.grid.resolution = Resolution::Axes(resolution.to_vec());
        let is_operator = config.task == Task::Operators;
        let tolerance = *config.eigensolver.tolerance.get_or_insert(if is_operator { 1e-8 } else { 1e-6 });
        positive(tolerance, "eigensolver.tolerance")?;
        let max_iterations = *config.eigensolver.max_iterations.get_or_insert(if is_operator { 300 } else { 200 });
        if max_iterations == 0 { return Err(invalid("eigensolver.max_iterations", "must be greater than zero")); }
        if config.dielectric.mesh_size == 0 { return Err(invalid("dielectric.mesh_size", "must be greater than zero")); }
        positive(config.dielectric.interface_tolerance, "dielectric.interface_tolerance")?;
        let mut k_labels = vec![];
        let mut k_label_indices = vec![];
        let (solved_bands, fractional) = match config.task {
            Task::Bands => {
                if config.operators.is_some() { return Err(invalid("operators", "not valid for task = bands")); }
                let bands = config.bands.get_or_insert_with(Bands::default);
                let path = resolve_path(&mut bands.path, self.geometry.lattice.kind, lattice_vectors)?;
                k_labels = path.labels;
                k_label_indices = path.label_indices;
                (bands.count, path.points)
            }
            Task::Operators => {
                if config.bands.is_some() { return Err(invalid("bands", "not valid for task = operators")); }
                let op = config.operators.as_ref().ok_or_else(|| invalid("operators", "required for task = operators"))?;
                validate_operators(op, config.polarization, &names)?;
                let count = op.band_lo.checked_add(op.retained_bands).and_then(|n| n.checked_add(op.remote_bands))
                    .ok_or_else(|| invalid("operators", "band window overflows"))?;
                let k = point(&op.k_point.value, "operators.k_point.value")?;
                let k = match op.k_point.basis {
                    KCoordinates::ReciprocalFractional => k,
                    KCoordinates::CartesianAngular => cartesian_to_fractional(k, lattice_vectors),
                };
                (count, vec![k])
            }
        };
        if solved_bands == 0 || solved_bands >= grid_size {
            return Err(invalid("grid.resolution", "the grid must have more samples than the solved band window"));
        }
        if config.eigensolver.block_size > 0 && config.eigensolver.block_size < solved_bands {
            return Err(invalid("eigensolver.block_size", "must be zero (automatic) or at least the solved band count"));
        }
        let cartesian: Vec<_> = fractional.iter().map(|&q| fractional_to_cartesian(q, lattice_vectors)).collect();
        if cartesian.iter().flatten().any(|v| !v.is_finite()) { return Err(invalid("geometry.lattice", "reciprocal coordinates overflow")); }
        let mut distances = vec![0.0];
        for pair in cartesian.windows(2) {
            distances.push(distances.last().unwrap() + (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]));
        }
        Ok(ResolvedConfig { config, lattice_vectors, resolution, k_points_fractional: fractional,
            k_points_cartesian: cartesian, distances, k_labels, k_label_indices, solved_bands })
    }
}

fn validate_operators(op: &Operators, pol: Polarization, names: &HashSet<String>) -> InterfaceResult<()> {
    if op.retained_bands == 0 { return Err(invalid("operators.retained_bands", "must be greater than zero")); }
    let requested: HashSet<_> = op.quantities.iter().copied().collect();
    if requested.len() != op.quantities.len() { return Err(invalid("operators.quantities", "duplicate quantity")); }
    if pol == Polarization::TM && requested.contains(&Quantity::SlowCoefficient) {
        return Err(Diagnostic::new("unsupported_quantity", "operators.quantities", "slow_coefficient is available for TE only"));
    }
    if pol == Polarization::TE && requested.contains(&Quantity::ExactTm) {
        return Err(Diagnostic::new("unsupported_quantity", "operators.quantities", "exact_tm is available for TM only"));
    }
    let needs_registry = [Quantity::RDerivatives, Quantity::BornHuang, Quantity::SlowCoefficient, Quantity::ExactTm]
        .iter().any(|q| requested.contains(q));
    if needs_registry && op.registry.is_none() {
        return Err(invalid("operators.registry", "declare the object and finite-difference step for local derivatives"));
    }
    if let Some(registry) = &op.registry {
        if !names.contains(&registry.object) { return Err(invalid("operators.registry.object", "unknown geometry object")); }
        if registry.points.is_empty() { return Err(invalid("operators.registry.points", "at least one registry point is required")); }
        for (i, p) in registry.points.iter().enumerate() { point(p, &format!("operators.registry.points[{i}]"))?; }
        positive(registry.fd_step, "operators.registry.fd_step")?;
    }
    if let Some(stencil) = &op.k_stencil {
        let n = stencil.points_per_axis;
        if n == 0 || n % 2 == 0 || n.checked_mul(n).is_none() {
            return Err(invalid("operators.k_stencil.points_per_axis", "must be a positive odd count whose square does not overflow"));
        }
        if !stencil.half_width.is_finite() || stencil.half_width < 0.0 || (n > 1 && stencil.half_width == 0.0) {
            return Err(invalid("operators.k_stencil.half_width", "must be positive for a multi-point stencil"));
        }
    }
    if op.reference.is_some() && op.k_stencil.is_some() {
        return Err(invalid("operators.reference", "k-stencils generate their own references; external references apply to point extraction"));
    }
    if op.reference.is_some() && !requested.contains(&Quantity::Overlap) {
        return Err(invalid("operators.reference", "external references require the overlap quantity"));
    }
    if requested.contains(&Quantity::Overlap) && op.reference.is_none() && op.k_stencil.as_ref().is_none_or(|s| s.points_per_axis == 1) {
        return Err(invalid("operators.quantities", "overlap requires a multi-point k-stencil; external reference inputs are available through Python"));
    }
    if let Some(limit) = op.fail_on_residual { positive(limit, "operators.fail_on_residual")?; }
    Ok(())
}

struct PathSamples { points: Vec<[f64; 2]>, labels: Vec<String>, label_indices: Vec<usize> }

fn resolve_path(path: &mut KPath, lattice: LatticeKind, vectors: [[f64; 2]; 2]) -> InterfaceResult<PathSamples> {
    let forms = usize::from(path.preset.is_some()) + usize::from(path.vertices.is_some()) + usize::from(path.points.is_some());
    if forms > 1 { return Err(invalid("bands.path", "choose exactly one of preset, vertices, or points")); }
    if forms == 0 {
        path.preset = Some(match lattice {
            LatticeKind::Square => "square", LatticeKind::Triangular => "triangular",
            LatticeKind::Rectangular => "rectangular",
            _ => return Err(invalid("bands.path", "an explicit path is required for this lattice")),
        }.into());
    }
    if path.points.is_some() && path.intervals_per_segment.is_some() {
        return Err(invalid("bands.path.intervals_per_segment", "sampled points are not interpolated"));
    }
    let (values, default_labels, sampled) = if let Some(preset) = &path.preset {
        if path.basis != KCoordinates::ReciprocalFractional { return Err(invalid("bands.path.basis", "presets use reciprocal fractional coordinates")); }
        let (v, l): (Vec<[f64; 2]>, Vec<&str>) = match preset.as_str() {
            "square" => (vec![[0.,0.],[0.5,0.],[0.5,0.5],[0.,0.]], vec!["Γ","X","M","Γ"]),
            "triangular" => (vec![[0.,0.],[0.5,0.],[2./3.,1./3.],[0.,0.]], vec!["Γ","M","K","Γ"]),
            "rectangular" => (vec![[0.,0.],[0.5,0.],[0.5,0.5],[0.,0.5],[0.,0.]], vec!["Γ","X","S","Y","Γ"]),
            _ => return Err(invalid("bands.path.preset", "expected square, triangular, or rectangular")),
        };
        (v, l.into_iter().map(str::to_owned).collect(), false)
    } else {
        let values = path.points.as_ref().or(path.vertices.as_ref()).unwrap();
        let v = values.iter().enumerate().map(|(i,p)| point(p, &format!("bands.path[{i}]"))).collect::<InterfaceResult<Vec<_>>>()?;
        (v, vec![], path.points.is_some())
    };
    if values.is_empty() || (!sampled && values.len() < 2) { return Err(invalid("bands.path", "provide sampled points or at least two vertices")); }
    let labels = if path.labels.is_empty() { default_labels } else { path.labels.clone() };
    if !labels.is_empty() && labels.len() != values.len() { return Err(invalid("bands.path.labels", "labels must match the supplied points or vertices")); }
    let (points, label_indices) = if sampled {
        (values.clone(), (0..values.len()).collect::<Vec<_>>())
    } else {
        let intervals = *path.intervals_per_segment.get_or_insert(15);
        if intervals == 0 { return Err(invalid("bands.path.intervals_per_segment", "must be greater than zero")); }
        let count = (values.len() - 1).checked_mul(intervals).and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("bands.path", "sample count overflows"))?;
        if count > 1_000_000 { return Err(invalid("bands.path", "a single path may contain at most 1,000,000 samples; split larger studies")); }
        let mut points = Vec::with_capacity(count);
        for leg in values.windows(2) {
            for i in 0..intervals {
                let t = i as f64 / intervals as f64;
                points.push([leg[0][0] + t * (leg[1][0] - leg[0][0]), leg[0][1] + t * (leg[1][1] - leg[0][1])]);
            }
        }
        points.push(*values.last().unwrap());
        (points, (0..values.len()).map(|i| i * intervals).collect())
    };
    let points = points.into_iter().map(|k| match path.basis {
        KCoordinates::ReciprocalFractional => k, KCoordinates::CartesianAngular => cartesian_to_fractional(k, vectors),
    }).collect();
    let label_indices = if labels.is_empty() { vec![] } else { label_indices };
    Ok(PathSamples { points, labels, label_indices })
}
