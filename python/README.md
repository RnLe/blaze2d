# Python Reference Data Toolkit

This folder hosts a small mamba environment and helper scripts for generating MPB/Meep reference data.

## Environment setup

```sh
cd python
mamba env create -f environment.yml
mamba activate mpb-reference
```

Key packages:

- `pymeep=*=mpi_mpich_*` – MPI-enabled Meep build exposing the MPB module.
- `mpb` – command line utilities for MPB control files.
- `numpy`, `scipy`, `matplotlib`, `pandas`, `h5py` – analysis and plotting helpers.

## Generating reference data

1. Activate the environment.
2. Run the helper script (MPI-enabled `pymeep` build):

   ```sh
   mamba run -n mpb-reference python generate_square_tm_bands.py \
       --output reference-data/square_tm_uniform_mpb.json \
       --num-bands 5 --resolution 32 --k-density 8 \
       --polarization tm
   ```

   Swap `--polarization te` and adjust `--output` to produce TE datasets (e.g., `reference-data/square_te_uniform_mpb.json`).

3. The script emits JSON containing the sampled k-path and the lowest TM bands. You can modify the script to target other geometries (e.g., air holes, dielectric rods) and store additional datasets under `python/reference-data/`.

## Visualization stubs

In future phases we will add lightweight notebooks/scripts here to visualize both MPB outputs and our Rust solver results for quick regression checks.

## Bundled reference snapshot

- `reference-data/square_tm_uniform_mpb.json` – MPB/Meep TM bands for a uniform ε=12 medium sampled along the Γ–X–M–Γ path (used by Rust validation tests).
- `reference-data/square_te_uniform_mpb.json` – MPB/Meep TE counterpart generated with `--polarization te`.
- `reference-data/square_tm_uniform.json` – Analytic TM bands for quick smoke tests if MPB is unavailable.
