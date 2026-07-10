/**
 * Real-space and reciprocal-space lattice geometry for the studio canvas.
 *
 * Mirrors the Rust conventions in `crates/core/src/lattice.rs` and
 * `bravais.rs`: a1 always along x, triangular uses the 60-degree convention
 * (a2 = [a/2, a*sqrt(3)/2]).
 */

import type { StudioLattice, PathPreset } from './configModel';

export type Vec2 = [number, number];

export interface LatticeBasis {
  a1: Vec2;
  a2: Vec2;
}

/** Build the real-space basis vectors from a StudioLattice. */
export function latticeVectors(lat: StudioLattice): LatticeBasis {
  const a = lat.a || 1;
  switch (lat.kind) {
    case 'square':
      return { a1: [a, 0], a2: [0, a] };
    case 'rectangular':
      return { a1: [a, 0], a2: [0, lat.b ?? a * 1.5] };
    case 'triangular':
      return { a1: [a, 0], a2: [a / 2, (a * Math.sqrt(3)) / 2] };
    case 'oblique': {
      const b = lat.b ?? a * 1.5;
      const alpha = ((lat.alpha_deg ?? 75) * Math.PI) / 180;
      return { a1: [a, 0], a2: [b * Math.cos(alpha), b * Math.sin(alpha)] };
    }
    case 'custom':
      return { a1: lat.a1 ?? [1, 0], a2: lat.a2 ?? [0, 1] };
  }
}

/** Fractional -> Cartesian using the basis. */
export function fracToCart(basis: LatticeBasis, frac: Vec2): Vec2 {
  return [
    basis.a1[0] * frac[0] + basis.a2[0] * frac[1],
    basis.a1[1] * frac[0] + basis.a2[1] * frac[1],
  ];
}

/** Cartesian -> fractional (inverse of the 2x2 basis matrix). */
export function cartToFrac(basis: LatticeBasis, cart: Vec2): Vec2 {
  const det = basis.a1[0] * basis.a2[1] - basis.a1[1] * basis.a2[0];
  if (Math.abs(det) < 1e-12) return [0, 0];
  const inv = 1 / det;
  return [
    (basis.a2[1] * cart[0] - basis.a2[0] * cart[1]) * inv,
    (-basis.a1[1] * cart[0] + basis.a1[0] * cart[1]) * inv,
  ];
}

/** Reciprocal basis: a_i . b_j = 2*pi*delta_ij. */
export function reciprocal(basis: LatticeBasis): LatticeBasis {
  const det = basis.a1[0] * basis.a2[1] - basis.a1[1] * basis.a2[0];
  const inv = (2 * Math.PI) / det;
  return {
    a1: [basis.a2[1] * inv, -basis.a2[0] * inv],
    a2: [-basis.a1[1] * inv, basis.a1[0] * inv],
  };
}

/**
 * The first Brillouin zone polygon (Wigner-Seitz cell of the reciprocal
 * lattice), computed as the bounded region around the origin cut by the
 * perpendicular bisectors of the nearest reciprocal lattice vectors.
 */
export function brillouinZone(basis: LatticeBasis): Vec2[] {
  const recip = reciprocal(basis);
  const { a1: b1, a2: b2 } = recip;

  // Candidate G-vectors within a small neighborhood.
  const gs: Vec2[] = [];
  for (let i = -2; i <= 2; i++) {
    for (let j = -2; j <= 2; j++) {
      if (i === 0 && j === 0) continue;
      gs.push([i * b1[0] + j * b2[0], i * b1[1] + j * b2[1]]);
    }
  }

  // Half-planes: point x is inside if x . (G/2) <= |G/2|^2 for all G.
  // Intersect the half-planes by walking candidate vertices (pairwise
  // bisector intersections) and keeping those satisfying every constraint.
  const planes = gs.map((g) => ({ nx: g[0] / 2, ny: g[1] / 2 }));
  const inside = (p: Vec2, eps: number) =>
    planes.every((pl) => p[0] * pl.nx + p[1] * pl.ny <= pl.nx * pl.nx + pl.ny * pl.ny + eps);

  const verts: Vec2[] = [];
  for (let i = 0; i < planes.length; i++) {
    for (let j = i + 1; j < planes.length; j++) {
      const a = planes[i];
      const b = planes[j];
      // Solve [a.n; b.n] . p = [|a.n|^2; |b.n|^2]
      const det = a.nx * b.ny - a.ny * b.nx;
      if (Math.abs(det) < 1e-9) continue;
      const ca = a.nx * a.nx + a.ny * a.ny;
      const cb = b.nx * b.nx + b.ny * b.ny;
      const px = (ca * b.ny - cb * a.ny) / det;
      const py = (a.nx * cb - b.nx * ca) / det;
      if (inside([px, py], 1e-6)) verts.push([px, py]);
    }
  }

  // Sort by angle around the origin, dedupe.
  verts.sort((p, q) => Math.atan2(p[1], p[0]) - Math.atan2(q[1], q[0]));
  const out: Vec2[] = [];
  for (const v of verts) {
    const last = out[out.length - 1];
    if (!last || Math.hypot(last[0] - v[0], last[1] - v[1]) > 1e-6) out.push(v);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Preset high-symmetry paths (fractional reciprocal coords). Mirror the
// Rust brillouin presets and the Python/web generators.
// ---------------------------------------------------------------------------

export const PRESET_CORNERS: Record<string, Vec2[]> = {
  square: [
    [0, 0],
    [0.5, 0],
    [0.5, 0.5],
    [0, 0],
  ],
  triangular: [
    [0, 0],
    [0.5, 0],
    [1 / 3, 1 / 3],
    [0, 0],
  ],
  rectangular: [
    [0, 0],
    [0.5, 0],
    [0.5, 0.5],
    [0, 0.5],
    [0, 0],
  ],
};

export const PRESET_LABELS: Record<string, string[]> = {
  square: ['Γ', 'X', 'M', 'Γ'],
  triangular: ['Γ', 'M', 'K', 'Γ'],
  rectangular: ['Γ', 'X', 'S', 'Y', 'Γ'],
};

/** Resolve preset 'auto' to a concrete preset name from the lattice kind. */
export function resolvePreset(preset: PathPreset, latticeKind: string): string | null {
  if (preset === 'auto') {
    switch (latticeKind) {
      case 'square':
        return 'square';
      case 'rectangular':
        return 'rectangular';
      case 'triangular':
        return 'triangular';
      default:
        return null; // oblique / custom have no standard path
    }
  }
  if (preset === 'hexagonal') return 'triangular';
  return preset;
}
