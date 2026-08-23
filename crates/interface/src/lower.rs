//! The only conversion from public calculations to numerical jobs.

use blaze2d_core::{
    dielectric::{DielectricOptions, SmoothingMethod, SmoothingOptions},
    drivers::{bandstructure::BandStructureJob, operator_data::OperatorDataJob},
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operator_data::OperatorDataConfig,
    polarization::Polarization as CorePolarization,
};
use crate::{PlannedJob, Polarization, Quantity, Smoothing, KCoordinates, fractional_to_cartesian};

impl PlannedJob {
    pub fn grid(&self) -> Grid2D {
        let [nx, ny] = self.resolved.resolution;
        Grid2D::new(nx, ny, 1.0, 1.0)
    }

    pub fn geometry(&self) -> Geometry2D {
        let c = &self.resolved.config;
        let [a1, a2] = self.resolved.lattice_vectors;
        let lattice = Lattice2D::oblique(a1, a2);
        let scale = lattice.characteristic_length();
        let moving = c.operators.as_ref().and_then(|o| o.registry.as_ref());
        let atoms = c.geometry.objects.iter().map(|o| {
            let mut pos = [o.center[0], o.center[1]];
            if moving.is_some_and(|r| r.object == o.name) {
                pos = [(pos[0] + self.registry[0]).rem_euclid(1.0),
                       (pos[1] + self.registry[1]).rem_euclid(1.0)];
            }
            BasisAtom { pos, radius: o.radius / scale, eps_inside: o.epsilon }
        }).collect();
        Geometry2D { lattice, eps_bg: c.geometry.background_epsilon, atoms }
    }

    pub fn dielectric(&self) -> DielectricOptions {
        let d = &self.resolved.config.dielectric;
        DielectricOptions { smoothing: SmoothingOptions {
            mesh_size: if d.smoothing == Smoothing::None { 1 } else { d.mesh_size },
            interface_tolerance: d.interface_tolerance,
            method: match d.smoothing {
                Smoothing::Analytic => SmoothingMethod::Analytic,
                _ => SmoothingMethod::Subgrid,
            },
        } }
    }

    pub fn eigensolver(&self) -> EigensolverConfig {
        let e = &self.resolved.config.eigensolver;
        EigensolverConfig {
            n_bands: self.resolved.solved_bands,
            max_iter: e.max_iterations.unwrap(),
            tol: e.tolerance.unwrap(),
            block_size: e.block_size,
            ..Default::default()
        }
    }

    pub fn polarization(&self) -> CorePolarization {
        match self.resolved.config.polarization {
            Polarization::TE => CorePolarization::TE,
            Polarization::TM => CorePolarization::TM,
        }
    }

    pub fn bands_job(&self) -> BandStructureJob {
        BandStructureJob {
            geom: self.geometry(), grid: self.grid(), pol: self.polarization(),
            k_path: self.resolved.k_points_fractional.clone(),
            eigensolver: self.eigensolver(), dielectric: self.dielectric(),
        }
    }

    pub fn operators_job(&self) -> OperatorDataJob {
        let c = &self.resolved.config;
        let o = c.operators.as_ref().expect("validated operator job");
        let q = &o.quantities;
        let needs_derivatives = q.iter().any(|q| matches!(q, Quantity::RDerivatives |
            Quantity::BornHuang | Quantity::SlowCoefficient | Quantity::ExactTm));
        let atom_index = o.registry.as_ref().and_then(|r|
            c.geometry.objects.iter().position(|o| o.name == r.object)).unwrap_or(0);
        let k = [o.k_point.value[0], o.k_point.value[1]];
        let k0 = match o.k_point.basis {
            KCoordinates::ReciprocalFractional => fractional_to_cartesian(k, self.resolved.lattice_vectors),
            KCoordinates::CartesianAngular => k,
        };
        OperatorDataJob {
            geom: self.geometry(), grid: self.grid(), pol: self.polarization(), k0,
            registry: self.registry,
            operator_data_config: OperatorDataConfig {
                band_lo: o.band_lo, n_retained: o.retained_bands, n_remote: o.remote_bands,
                compute_mass_tensor: q.contains(&Quantity::MassTensor),
                compute_born_huang: q.contains(&Quantity::BornHuang),
                compute_slow_coefficient: q.contains(&Quantity::SlowCoefficient),
                compute_overlap: q.contains(&Quantity::Overlap),
                fail_on_residual: o.fail_on_residual,
            },
            eigensolver: self.eigensolver(), dielectric: self.dielectric(),
            fd_step: o.registry.as_ref().map_or(0.001, |r| r.fd_step),
            atom_index, compute_dielectric_derivatives: needs_derivatives,
        }
    }
}
