/** Geometry in units of the monolayer period a. */
export type Lattice = 'triangular' | 'square';
export type Point = readonly [number, number];

export function rotate([x, y]: Point, radians: number): Point {
  return [x * Math.cos(radians) - y * Math.sin(radians), x * Math.sin(radians) + y * Math.cos(radians)];
}

export function latticeBasis(lattice: Lattice): readonly [Point, Point] {
  return [[1, 0], lattice === 'triangular' ? [0.5, Math.sqrt(3) / 2] : [0, 1]];
}

export function moireGeometry(lattice: Lattice, angle: number) {
  const symmetry = lattice === 'triangular' ? 60 : 90;
  // Use the equivalent orientation nearest alignment. Keeping the sign makes
  // the registry cell rotate correctly on either side of a symmetry angle.
  const reduced = angle <= symmetry / 2 ? angle : angle - symmetry;
  const radians = reduced * Math.PI / 180;
  const aligned = Math.abs(reduced) < 1e-8;
  const eta = aligned ? 0 : 2 * Math.sin(Math.abs(radians) / 2);
  const basis = latticeBasis(lattice);
  // L_i = (I - R(-phi))^-1 a_i. Thus (b_j - R(phi)b_j).L_i = 2pi delta_ij.
  // These are geometric beat vectors, not coincidence-site supercell vectors.
  const cot = aligned ? 0 : 1 / Math.tan(radians / 2);
  const vectors = aligned ? null : basis.map(([x, y]) => [
    (x + cot * y) / 2, (y - cot * x) / 2,
  ] as Point) as [Point, Point];
  return { symmetry, reduced, eta, period: aligned ? Infinity : 1 / eta, vectors, basis };
}

/** A bounded point set large enough to fill the viewport after any rotation. */
export function latticePoints(lattice: Lattice, radius: number): Point[] {
  const [a1, a2] = latticeBasis(lattice);
  const reach = Math.ceil(radius / a2[1]) + 1;
  const points: Point[] = [];
  for (let j = -reach; j <= reach; j++) {
    for (let i = -reach; i <= reach; i++) {
      const point: Point = [i * a1[0] + j * a2[0], j * a2[1]];
      if (Math.hypot(...point) <= radius) points.push(point);
    }
  }
  return points;
}
