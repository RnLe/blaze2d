export const EXAMPLE_CONFIGS = {
  squareTm: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.2
eps_inside = 1.0

[grid]
nx = 24
ny = 24
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 6

[eigensolver]
n_bands = 5
max_iter = 120
tol = 1e-5
`,

  teTmCompare: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.22
eps_inside = 1.0

[grid]
nx = 24
ny = 24
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 5

[eigensolver]
n_bands = 5
max_iter = 120
tol = 1e-5

[[sweeps]]
parameter = "polarization"
values = ["TM", "TE"]
`,

  gammaToX: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.2
eps_inside = 1.0

[grid]
nx = 24
ny = 24
lx = 1.0
ly = 1.0

[path]
k_path = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]

[eigensolver]
n_bands = 5
max_iter = 120
tol = 1e-5
`,

  presetPath: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 10.0

[geometry.lattice]
type = "triangular"
a = 1.0

[[geometry.atoms]]
pos = [0.0, 0.0]
radius = 0.18
eps_inside = 1.0

[grid]
nx = 24
ny = 24
lx = 1.0
ly = 1.0

[path]
preset = "triangular"
segments_per_leg = 6

[eigensolver]
n_bands = 5
max_iter = 120
tol = 1e-5
`,

  radiusSweep: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.18
eps_inside = 1.0

[grid]
nx = 20
ny = 20
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 4

[eigensolver]
n_bands = 4
max_iter = 100
tol = 1e-5

[[sweeps]]
parameter = "atom0.radius"
values = [0.16, 0.22, 0.28]
`,

  resolutionEpsilonSweep: `polarization = "TM"

[bulk]
threads = 8
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 10.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.22
eps_inside = 1.0

[grid]
nx = 16
ny = 16
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 4

[eigensolver]
n_bands = 4
max_iter = 100
tol = 1e-5

[[sweeps]]
parameter = "resolution"
values = [16, 20]

[[sweeps]]
parameter = "eps_bg"
values = [10.0, 12.0]
`,

  streaming: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.2
eps_inside = 1.0

[grid]
nx = 20
ny = 20
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 4

[eigensolver]
n_bands = 4
max_iter = 100
tol = 1e-5

[[sweeps]]
parameter = "atom0.radius"
values = [0.18, 0.24, 0.3]
`,

  selective: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.22
eps_inside = 1.0

[grid]
nx = 24
ny = 24
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 5

[eigensolver]
n_bands = 6
max_iter = 120
tol = 1e-5
`,

  twoAtom: `polarization = "TM"

[bulk]
threads = 1
verbose = false

[solver]
type = "maxwell"

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "triangular"
a = 1.0

[[geometry.atoms]]
pos = [0.0, 0.0]
radius = 0.14
eps_inside = 1.0

[[geometry.atoms]]
pos = [0.333333, 0.333333]
radius = 0.14
eps_inside = 1.0

[grid]
nx = 24
ny = 24
lx = 1.0
ly = 1.0

[path]
preset = "triangular"
segments_per_leg = 5

[eigensolver]
n_bands = 5
max_iter = 120
tol = 1e-5
`,
} as const;
