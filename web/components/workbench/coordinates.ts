export type CoordinateSystem = 'fractional' | 'cartesian';
export type LatticeVectors = readonly [readonly [number, number], readonly [number, number]];
export function toCartesian(point: readonly number[], [a, b]: LatticeVectors): [number, number] {
  return [a[0] * point[0] + b[0] * point[1], a[1] * point[0] + b[1] * point[1]];
}
export function toFractional([x, y]: readonly number[], [a, b]: LatticeVectors): [number, number] {
  const determinant = a[0] * b[1] - a[1] * b[0];
  return [(b[1] * x - b[0] * y) / determinant, (a[0] * y - a[1] * x) / determinant];
}
