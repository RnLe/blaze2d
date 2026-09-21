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
} satisfies Record<string, Module>;

export type ModuleId = keyof typeof modules;

export const stages: { number: 1 | 2 | 3; title: string; anchor: string; description: string; items: ModuleId[] }[] = [
  { number: 1, title: 'Research foundation', anchor: 'stage-1-research-foundation', description: 'Establish a reliable research core.', items: ['core', 'symmetries', 'warm', 'evidence'] },
  { number: 2, title: 'Generalize and optimize 2D', anchor: 'stage-2-generalize-and-optimize-2d', description: 'Fully generalize 2D geometry, improve solving, and support MPB workflows.', items: ['geometry', 'solver', 'gradients', 'compatibility'] },
  { number: 3, title: 'GPU, 3D, and slabs', anchor: 'stage-3-gpu-3d-and-slabs', description: 'Add GPU execution and extend the physical models beyond 2D.', items: ['gpu', 'maxwell3d', 'slabs'] },
];
