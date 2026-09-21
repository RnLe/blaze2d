/* Generated from examples/calculations. Run pnpm generate:examples to update. */
export const examples = [
  {
    "slug": "square-rods",
    "title": "Square rods",
    "description": "TM bands of a square lattice of dielectric circles.",
    "category": "Bands",
    "image": "/examples/square-rods.svg",
    "accent": "#59b6ff",
    "source": "schema = \"blaze2d/1\"\ntask = \"bands\"\npolarization = \"TM\"\n\n[geometry]\nbackground_epsilon = 1.0\nlattice = { type = \"square\", a = 1.0 }\n\n[[geometry.objects]]\nname = \"rod\"\nkind = \"circle\"\ncenter = [0.0, 0.0]\nradius = 0.20\nepsilon = 8.9\n\n[grid]\nresolution = 32\n\n[bands]\ncount = 8\npath = { preset = \"square\", intervals_per_segment = 15 }\n\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nresult = blaze.solve(config)\nprint(result[\"frequencies\"])\n"
  },
  {
    "slug": "triangular-holes",
    "title": "Triangular holes",
    "description": "TE bands of air holes in a dielectric background.",
    "category": "Bands",
    "image": "/examples/triangular-holes.svg",
    "accent": "#7ba6ff",
    "source": "schema = \"blaze2d/1\"\ntask = \"bands\"\npolarization = \"TE\"\n\n[geometry]\nbackground_epsilon = 12.0\nlattice = { type = \"triangular\", a = 1.0 }\n\n[[geometry.objects]]\nname = \"hole\"\nkind = \"circle\"\ncenter = [0.0, 0.0]\nradius = 0.30\nepsilon = 1.0\n\n[grid]\nresolution = 32\n\n[bands]\ncount = 8\npath = { preset = \"triangular\", intervals_per_segment = 15 }\n\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nresult = blaze.solve(config)\nprint(result[\"frequencies\"])\n"
  },
  {
    "slug": "rectangular-cell",
    "title": "Rectangular cell",
    "description": "A non-unit lattice and a rectangular sampling grid.",
    "category": "Bands",
    "image": "/examples/rectangular-cell.svg",
    "accent": "#8ed2ff",
    "source": "schema = \"blaze2d/1\"\ntask = \"bands\"\npolarization = \"TM\"\n\n[geometry]\nbackground_epsilon = 1.0\nlattice = { type = \"rectangular\", a = 2.0, b = 1.0 }\n\n[[geometry.objects]]\nname = \"rod\"\nkind = \"circle\"\ncenter = [0.0, 0.0]\nradius = 0.20\nepsilon = 8.9\n\n[grid]\nresolution = [24, 32]\n\n[bands]\ncount = 8\npath = { preset = \"rectangular\", intervals_per_segment = 15 }\n\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nresult = blaze.solve(config)\nprint(result[\"frequencies\"])\n"
  },
  {
    "slug": "oblique-cell",
    "title": "Oblique cell",
    "description": "An explicit path measured in the reciprocal Cartesian metric.",
    "category": "Bands",
    "image": "/examples/oblique-cell.svg",
    "accent": "#5f92e0",
    "source": "schema = \"blaze2d/1\"\ntask = \"bands\"\npolarization = \"TM\"\n\n[geometry]\nbackground_epsilon = 1.0\nlattice = { type = \"oblique\", a = 1.4, b = 0.9, angle_deg = 70.0 }\n\n[[geometry.objects]]\nname = \"rod\"\nkind = \"circle\"\ncenter = [0.0, 0.0]\nradius = 0.20\nepsilon = 8.9\n\n[grid]\nresolution = 32\n\n[bands]\ncount = 8\npath = { vertices = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]], intervals_per_segment = 10 }\n\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nresult = blaze.solve(config)\nprint(result[\"frequencies\"])\n"
  },
  {
    "slug": "radius-sweep",
    "title": "Radius and polarization sweep",
    "description": "Two ordered axes applied to one immutable model.",
    "category": "Studies",
    "image": "/examples/radius-sweep.svg",
    "accent": "#a6ccff",
    "source": "schema = \"blaze2d/1\"\ntask = \"bands\"\npolarization = \"TM\"\n\n[geometry]\nbackground_epsilon = 1.0\nlattice = { type = \"square\", a = 1.0 }\n\n[[geometry.objects]]\nname = \"rod\"\nkind = \"circle\"\ncenter = [0.0, 0.0]\nradius = 0.20\nepsilon = 8.9\n\n[grid]\nresolution = 32\n\n[bands]\ncount = 8\npath = { preset = \"square\", intervals_per_segment = 15 }\n\n[[sweeps]]\nname = \"radius\"\ntarget = \"geometry.objects.rod.radius\"\nlinspace = { start = 0.16, stop = 0.24, count = 5 }\n\n[[sweeps]]\nname = \"polarization\"\ntarget = \"polarization\"\nvalues = [\"TM\", \"TE\"]\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nstudy = blaze.run(config)\nfor result in study[\"results\"]:\n    print(result[\"job_index\"], result[\"metadata\"][\"sweep_parameters\"])\n"
  },
  {
    "slug": "operator-point",
    "title": "Projected operators",
    "description": "Velocity and inverse mass with a nonzero retained-band offset.",
    "category": "Operators",
    "image": "/examples/operator-point.svg",
    "accent": "#4aa6e8",
    "source": "schema = \"blaze2d/1\"\ntask = \"operators\"\npolarization = \"TE\"\n\n[geometry]\nbackground_epsilon = 12.0\nlattice = { type = \"square\" }\n\n[[geometry.objects]]\nname = \"hole\"\nkind = \"circle\"\nradius = 0.2\nepsilon = 1.0\n\n[grid]\nresolution = [24, 32]\n\n[operators]\nband_lo = 2\nretained_bands = 2\nremote_bands = 3\nk_point = { value = [0.25, 0.0], basis = \"reciprocal_fractional\" }\nquantities = [\"velocity\", \"mass_tensor\"]\n\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nresult = blaze.solve(config)\nprint(result[\"eigenvalues\"])\n"
  },
  {
    "slug": "registry-stencil",
    "title": "Registry and k-stencil",
    "description": "Radius sweeps, periodic translations, and transported stencil references.",
    "category": "Operators",
    "image": "/examples/registry-stencil.svg",
    "accent": "#86bdf5",
    "source": "schema = \"blaze2d/1\"\ntask = \"operators\"\npolarization = \"TE\"\n\n[geometry]\nbackground_epsilon = 12.0\nlattice = { type = \"square\" }\n\n[[geometry.objects]]\nname = \"hole\"\nkind = \"circle\"\nradius = 0.2\nepsilon = 1.0\n\n[grid]\nresolution = [24, 32]\n\n[operators]\nband_lo = 2\nretained_bands = 2\nremote_bands = 3\nk_point = { value = [0.25, 0.0], basis = \"reciprocal_fractional\" }\nquantities = [\"velocity\", \"mass_tensor\", \"r_derivatives\"]\nregistry = { object = \"hole\", points = [[0.0, 0.0], [0.25, 0.25]], fd_step = 0.001 }\nk_stencil = { points_per_axis = 3, half_width = 0.02 }\n\n[[sweeps]]\nname = \"radius\"\ntarget = \"geometry.objects.hole.radius\"\nvalues = [0.18, 0.22]\n",
    "python": "from pathlib import Path\nimport blaze\n\nconfig = blaze.Config.from_file(Path(__file__).with_suffix(\".toml\"))\nstudy = blaze.run(config)\nfor result in study[\"results\"]:\n    print(result[\"job_index\"], result[\"metadata\"][\"sweep_parameters\"])\n"
  }
] as const;
