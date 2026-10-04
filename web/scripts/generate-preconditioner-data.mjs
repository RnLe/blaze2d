// Regenerates components/preconditioner-page/convergence-data.json, the
// convergence histories shown on the preconditioner page, with the same solver
// code the page documents:
//   pnpm generate:preconditioner
import { writeFile } from 'node:fs/promises';
import { maxwellModel, lobpcg } from '../components/preconditioner-page/linear-algebra.ts';

const grids = [32, 64, 128, 256, 512];
const contrasts = [1, 4, 13];
const kinds = ['none', 'jacobi', 'fourier', 'exact'];
const tolerance = 1e-7;
const maxIterations = 6000;

/** Every early iteration, then log-spaced samples, always keeping the last one. */
function sample(history) {
  const points = [];
  let next = 0;
  history.forEach((eta, iteration) => {
    if (iteration < 24 || iteration >= next || iteration === history.length - 1) {
      points.push([iteration, Number(eta.toPrecision(3))]);
      next = Math.ceil(iteration * 1.12);
    }
  });
  return points;
}

const runs = {};
for (const contrast of contrasts) {
  runs[contrast] = {};
  for (const n of grids) {
    const model = maxwellModel(n, contrast);
    runs[contrast][n] = {};
    for (const kind of kinds) {
      const run = lobpcg(model, kind, tolerance, maxIterations);
      runs[contrast][n][kind] = {
        iterations: run.converged ? run.history.length - 1 : null,
        eigenvalue: Number(run.eigenvalue.toPrecision(6)),
        points: sample(run.history),
      };
      console.log(`ε=${contrast} N=${n} ${kind}: ${run.converged ? run.history.length - 1 : `>${maxIterations}`}`);
    }
  }
}

const data = { tolerance, maxIterations, grids, contrasts, kinds, runs };
await writeFile(new URL('../components/preconditioner-page/convergence-data.json', import.meta.url), JSON.stringify(data) + '\n');
