export type ModuleStatus = 'Done' | 'In Development' | 'Planned';

type Module = {
  title: string;
  anchor: string;
  description: string;
  status: ModuleStatus;
  motivation: string;
};

export const modules = {
  core: {
    title: '2D research core', anchor: '2d-research-core',
    description: 'Rust implementation and theory: LOBPCG, mixed precision, clear API/TOML interfaces, and operator projections.', status: 'Done',
    motivation: 'Make repeated band calculations and projected Maxwell operators available in one reproducible research workflow. This is the foundation the later modules extend.',
  },
  symmetries: {
    title: 'Symmetries', anchor: 'symmetries',
    description: 'Reuse equivalent wavevectors and reduce modes.', status: 'In Development',
    motivation: 'Avoid solving equivalent problems and reduce eligible mode spaces. Using the crystal’s symmetry can save computation before adding new hardware or dimensions.',
  },
  warm: {
    title: 'Configuration warm starts', anchor: 'configuration-warm-starts',
    description: 'Reuse a solution when the geometry changes.', status: 'Planned',
    motivation: '',
  },
  evidence: {
    title: 'Theory and benchmarks', anchor: 'theory-and-benchmarks',
    description: 'Document the science and measure each change.', status: 'In Development',
    motivation: 'This website is being carefully developed as an accessible entry point to Blaze, with guidance on its use and transparent explanations of the science. Make numerical results easier to trust and performance claims easier to reproduce. Profiling also identifies which improvements are worth implementing next.',
  },
  geometry: {
    title: 'General geometry', anchor: 'general-geometry',
    description: 'Add polygons, cutouts, and dielectric grids.', status: 'Planned',
    motivation: 'Support research with arbitrary 2D geometries and integrate into existing workflows, including MPB. Represent etched shapes, composite inclusions, and imported dielectric layouts directly.',
  },
  solver: {
    title: 'Solver optimization', anchor: 'solver-optimization',
    description: 'Lower iteration cost; target chosen frequencies.', status: 'Planned',
    motivation: 'Reduce the cost of each solve and reach defect or cavity modes near a chosen frequency without computing every lower band first.',
  },
  gradients: {
    title: 'Gradients and inverse design', anchor: 'gradients-and-inverse-design',
    description: 'Use sensitivities to optimize a crystal.', status: 'Planned',
    motivation: 'Use the direction of a parameter’s effect to improve a band gap or dispersion curve. This can guide searches more efficiently than exhaustive parameter grids.',
  },
  compatibility: {
    title: 'MPB compatibility', anchor: 'mpb-compatibility',
    description: 'Translate and validate established workflows.', status: 'Planned',
    motivation: 'Lower the effort needed to try Blaze on an existing calculation, and make comparisons with MPB straightforward and reproducible.',
  },
  gpu: {
    title: 'GPU execution', anchor: 'gpu-execution',
    description: 'Accelerate large batches of band calculations with GPU parallelism.', status: 'Planned',
    motivation: '',
  },
  maxwell3d: {
    title: 'Full 3D Maxwell', anchor: 'full-3d-maxwell',
    description: 'Compute band diagrams for 3D photonic crystals.', status: 'Planned',
    motivation: 'Study three-dimensional photonic crystals and polarization coupling beyond the 2D TE/TM split.',
  },
  slabs: {
    title: 'Photonic slabs', anchor: 'photonic-slabs',
    description: 'Use a reduced guided-mode basis for efficient slab band calculations.', status: 'Planned',
    motivation: 'Include vertical confinement and radiation when studying patterned films. A slab-specific method offers these observables without requiring a general 3D grid. Slab support in Blaze would also provide the physics needed to continue the thesis’s two-scale moiré research in photonic crystal slabs.',
  },
  preconditioner: {
    title: 'Learned preconditioner', anchor: 'learned-preconditioner',
    description: 'Learn the corrections Blaze’s physics-based preconditioner misses.', status: 'Planned',
    motivation: 'Preconditioning was Blaze’s most rewarding optimization. Learning works on the same lever, so a learned correction must lower the total time to the same validated eigenpairs, including training and inference.',
  },
  tuning: {
    title: 'Self-tuning solves', anchor: 'self-tuning-solves',
    description: 'Learn the best solver settings during a single band diagram.', status: 'Planned',
    motivation: 'Tuning the preconditioner by hand took much work, and the best settings differ between crystals. A self-tuning run adapts to each crystal without a training set, provided exploration pays for itself within the run.',
  },
  initialization: {
    title: 'Learned starting subspaces', anchor: 'learned-starting-subspaces',
    description: 'Predict a close first subspace for new geometries.', status: 'Planned',
    motivation: 'Cold starts dominate short band diagrams and broad geometry sweeps. A learned predictor has to beat Blaze’s classical reuse of nearby solutions, not only a cold start.',
  },
  surrogates: {
    title: 'Design surrogates', anchor: 'design-surrogates',
    description: 'Learn which configurations deserve a full solve.', status: 'Planned',
    motivation: 'Surrogates reduce the number of full solves rather than the cost of each one, so their benefit is measured per study, separately from solver speed.',
  },
  reduced: {
    title: 'Reduced Bloch solver', anchor: 'reduced-bloch-solver',
    description: 'Solve a whole band diagram in one small shared basis.', status: 'Planned',
    motivation: 'With few iterations left per k-point, the remaining cost is full-size work repeated across the family. Accepted k-points avoid it entirely, which no faster eigensolver can achieve.',
  },
  acceptance: {
    title: 'Residual-checked bases', anchor: 'residual-checked-bases',
    description: 'Accept reduced results only where residuals confirm them.', status: 'Planned',
    motivation: 'Equal accuracy is the condition for every speed-up on this page. Cheap residuals make it hold for each accepted result, and adaptive enrichment decides where full solves are still needed.',
  },
  sweeps: {
    title: 'Families across sweeps', anchor: 'families-across-sweeps',
    description: 'Share one basis across material and geometry sweeps.', status: 'Planned',
    motivation: 'Blaze’s thesis workflow solves thousands of related local Bloch problems, with eigenvectors and projected couplings. Sharing one basis across them targets the largest redundancy in such studies.',
  },
  simultaneous: {
    title: 'Simultaneous iteration', anchor: 'simultaneous-iteration',
    description: 'Iterate on many parameters at once in one shared space.', status: 'Planned',
    motivation: 'Reduced bases share results; a simultaneous iteration would also share the work of finding them. It is the most ambitious step and builds on a validated reduced solver.',
  },
} satisfies Record<string, Module>;

export type ModuleId = keyof typeof modules;

export const stages: { id: '1' | '2' | '3' | 'M' | 'X'; title: string; anchor: string; description: string; items: ModuleId[]; landscape?: true }[] = [
  { id: '1', title: 'Research foundation', anchor: 'stage-1-research-foundation', description: 'Establish a reliable research core.', items: ['core', 'symmetries', 'warm', 'evidence'] },
  { id: '2', title: 'Generalize and optimize 2D', anchor: 'stage-2-generalize-and-optimize-2d', description: 'Fully generalize 2D geometry, improve solving, and support MPB workflows.', items: ['geometry', 'solver', 'gradients', 'compatibility'] },
  { id: '3', title: 'GPU, 3D, and slabs', anchor: 'stage-3-gpu-3d-and-slabs', description: 'Add GPU execution and extend the physical models beyond 2D.', items: ['gpu', 'maxwell3d', 'slabs'] },
  { id: 'M', title: 'Machine learning', anchor: 'stage-m-machine-learning', description: 'Use recent advances in machine learning to accelerate solving, with every result verified.', items: ['preconditioner', 'tuning', 'initialization', 'surrogates'], landscape: true },
  { id: 'X', title: 'Solution families', anchor: 'stage-x-solution-families', description: 'Push Blaze beyond the conventional solver loop into solving whole families at once.', items: ['reduced', 'acceptance', 'sweeps', 'simultaneous'], landscape: true },
];
