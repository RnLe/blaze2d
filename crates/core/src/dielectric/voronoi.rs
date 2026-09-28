//! Interface averages on the sampling lattice's Wigner–Seitz cell.
//! Circle/polygon areas and integrated interface normals use oriented boundary
//! integrals. The cell rotates with its lattice, including triangular C6 maps.
use super::{SmoothingOptions, build_anisotropic_inv_tensor, isotropic_tensor};
use crate::{geometry::Geometry2D, grid::Grid2D, lattice::Lattice2D};

type Point = [f64; 2];
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: Point, b: Point) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn scaled(a: Point, t: f64) -> Point {
    [a[0] * t, a[1] * t]
}
fn norm(a: Point) -> f64 {
    a[0].hypot(a[1])
}

/// Gauss reduction makes the two-dimensional Voronoi neighbor set finite.
fn reduced_basis(mut a: Point, mut b: Point) -> Result<[Point; 2], String> {
    for _ in 0..128 {
        if dot(b, b) < dot(a, a) {
            std::mem::swap(&mut a, &mut b);
        }
        if !dot(a, a).is_finite()
            || dot(a, a) <= 0.0
            || !cross(a, b).is_finite()
            || cross(a, b) == 0.0
        {
            return Err("Voronoi smoothing requires a finite nonsingular sampling lattice".into());
        }
        if dot(a, b).abs() <= 0.5 * dot(a, a) * (1.0 + 8.0 * f64::EPSILON) {
            return Ok([a, b]);
        }
        let mu = (dot(a, b) / dot(a, a)).round();
        if mu == 0.0 {
            return Ok([a, b]);
        }
        b = sub(b, scaled(a, mu));
    }
    Err("sampling lattice reduction did not converge".into())
}

pub fn cell(lattice: Lattice2D, grid: Grid2D) -> Result<Vec<Point>, String> {
    if grid.nx == 0 || grid.ny == 0 || grid.centered {
        return Err("Voronoi smoothing requires a nonempty node-sampled grid".into());
    }
    let [a, b] = reduced_basis(
        scaled(lattice.a1, 1.0 / grid.nx as f64),
        scaled(lattice.a2, 1.0 / grid.ny as f64),
    )?;
    let radius = norm(a) + norm(b);
    let mut polygon = vec![
        [-radius, -radius],
        [radius, -radius],
        [radius, radius],
        [-radius, radius],
    ];
    for direction in [a, b, [a[0] + b[0], a[1] + b[1]], sub(a, b)] {
        for sign in [-1.0, 1.0] {
            let normal = scaled(direction, sign);
            let bound = dot(normal, normal) * 0.5;
            let mut clipped = Vec::new();
            for i in 0..polygon.len() {
                let p = polygon[i];
                let q = polygon[(i + 1) % polygon.len()];
                let dp = dot(normal, p) - bound;
                let dq = dot(normal, q) - bound;
                if dp <= 0.0 {
                    clipped.push(p);
                }
                if (dp <= 0.0) != (dq <= 0.0) {
                    let t = dp / (dp - dq);
                    clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
                }
            }
            polygon = clipped;
        }
    }
    let area: f64 = (0..polygon.len())
        .map(|i| cross(polygon[i], polygon[(i + 1) % polygon.len()]) * 0.5)
        .sum();
    let expected = lattice.cell_area() / grid.len() as f64;
    if polygon.len() < 3 || !area.is_finite() || (area - expected).abs() > 1e-10 * expected {
        return Err("Voronoi cell failed its lattice-area identity".into());
    }
    Ok(polygon)
}

fn image_offsets(delta: Point, basis: [Point; 2]) -> impl Iterator<Item = Point> {
    let [a, b] = basis;
    let determinant = cross(a, b);
    let u = (cross(delta, b) / determinant).round();
    let v = (cross(a, delta) / determinant).round();
    let wrapped = sub(sub(delta, scaled(a, u)), scaled(b, v));
    (-1..=1).flat_map(move |i| {
        (-1..=1).map(move |j| sub(sub(wrapped, scaled(a, i as f64)), scaled(b, j as f64)))
    })
}

/// Overlapping disks require a separate material-order intersection rule.
/// Reject them explicitly instead of summing overlapping areas twice.
pub fn validate_geometry(geometry: &Geometry2D, grid: Grid2D) -> Result<(), String> {
    cell(geometry.lattice, grid)?;
    let basis = reduced_basis(geometry.lattice.a1, geometry.lattice.a2)?;
    for (i, atom) in geometry.atoms.iter().enumerate() {
        let radius = atom.radius_cartesian(&geometry.lattice);
        if !radius.is_finite() || radius <= 0.0 || 2.0 * radius > norm(basis[0]) * (1.0 + 1e-12) {
            return Err("Voronoi smoothing requires disks without periodic self-overlap".into());
        }
        for other in &geometry.atoms[..i] {
            let delta = geometry
                .lattice
                .fractional_to_cartesian(sub(atom.pos, other.pos));
            let separation = image_offsets(delta, basis)
                .map(norm)
                .fold(f64::INFINITY, f64::min);
            if separation < (radius + other.radius_cartesian(&geometry.lattice)) * (1.0 - 1e-12) {
                return Err("Voronoi smoothing requires nonoverlapping material disks".into());
            }
        }
    }
    Ok(())
}

/// Return intersection area and integral of the circle's outward unit normal.
/// Straight pieces contribute triangle area; exterior pieces contribute arcs.
fn intersection(polygon: &[Point], circle_center: Point, radius: f64) -> (f64, Point) {
    let r2 = radius * radius;
    let mut area = 0.0;
    let mut normal = [0.0; 2];
    for i in 0..polygon.len() {
        let a = sub(polygon[i], circle_center);
        let b = sub(polygon[(i + 1) % polygon.len()], circle_center);
        let d = sub(b, a);
        let dd = dot(d, d);
        if dd == 0.0 {
            continue;
        }
        let ad = dot(a, d);
        let discriminant = ad * ad - dd * (dot(a, a) - r2);
        let mut cuts = vec![0.0, 1.0];
        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            for t in [(-ad - root) / dd, (-ad + root) / dd] {
                if t > 0.0 && t < 1.0 {
                    cuts.push(t);
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        for interval in cuts.windows(2) {
            let [t0, t1] = [interval[0], interval[1]];
            let p = [a[0] + d[0] * t0, a[1] + d[1] * t0];
            let q = [a[0] + d[0] * t1, a[1] + d[1] * t1];
            let middle = [a[0] + d[0] * (t0 + t1) * 0.5, a[1] + d[1] * (t0 + t1) * 0.5];
            if dot(middle, middle) < r2 {
                area += cross(p, q) * 0.5;
                // The closed intersection boundary has zero net normal.
                normal[0] -= q[1] - p[1];
                normal[1] += q[0] - p[0];
            } else {
                area += r2 * cross(p, q).atan2(dot(p, q)) * 0.5;
            }
        }
    }
    (area.max(0.0), normal)
}

pub(super) fn sample(
    geometry: &Geometry2D,
    grid: Grid2D,
    options: &SmoothingOptions,
) -> Result<(Vec<f64>, Vec<f64>, Vec<[f64; 4]>), String> {
    validate_geometry(geometry, grid)?;
    let polygon = cell(geometry.lattice, grid)?;
    let cell_area = geometry.lattice.cell_area() / grid.len() as f64;
    let cell_radius = polygon.iter().copied().map(norm).fold(0.0, f64::max);
    let basis = reduced_basis(geometry.lattice.a1, geometry.lattice.a2)?;
    let mut epsilon = vec![geometry.eps_bg; grid.len()];
    let mut inverse = vec![1.0 / geometry.eps_bg; grid.len()];
    let mut tensors = vec![isotropic_tensor(1.0 / geometry.eps_bg); grid.len()];
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let index = grid.idx(ix, iy);
            let position = [ix as f64 / grid.nx as f64, iy as f64 / grid.ny as f64];
            let mut normal = [0.0; 2];
            for atom in &geometry.atoms {
                let radius = atom.radius_cartesian(&geometry.lattice);
                let delta = geometry
                    .lattice
                    .fractional_to_cartesian(sub(atom.pos, position));
                for center in image_offsets(delta, basis)
                    .filter(|&center| norm(center) < radius + cell_radius)
                {
                    let (area, n) = intersection(&polygon, center, radius);
                    let fill = (area / cell_area).clamp(0.0, 1.0);
                    epsilon[index] += fill * (atom.eps_inside - geometry.eps_bg);
                    inverse[index] += fill * (1.0 / atom.eps_inside - 1.0 / geometry.eps_bg);
                    let contrast = atom.eps_inside - geometry.eps_bg;
                    normal[0] += contrast * n[0];
                    normal[1] += contrast * n[1];
                }
            }
            let length = norm(normal);
            tensors[index] = if length > options.tolerance() * cell_radius * epsilon[index] {
                build_anisotropic_inv_tensor(
                    scaled(normal, 1.0 / length),
                    epsilon[index],
                    inverse[index],
                )
            } else {
                isotropic_tensor(inverse[index])
            };
        }
    }
    Ok((epsilon, inverse, tensors))
}
