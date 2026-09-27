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

impl crate::Bands {
    pub(crate) fn run_options(&self, retain_eigenvectors: bool) -> blaze2d_core::drivers::bandstructure::RunOptions {
        use serde_json::json;
        blaze2d_core::drivers::bandstructure::RunOptions {
            reuse_gamma: false, disable_band_tracking: !self.tracking, retain_eigenvectors,
            symmetry: blaze2d_core::symmetry::reuse::SymmetryOptions {
                        k_reuse: match self.symmetry.k_reuse {crate::KReuse::Off=>blaze2d_core::symmetry::reuse::KReuse::Off,crate::KReuse::Identical=>blaze2d_core::symmetry::reuse::KReuse::Identical,crate::KReuse::Unitary=>blaze2d_core::symmetry::reuse::KReuse::Unitary,crate::KReuse::TimeReversal=>blaze2d_core::symmetry::reuse::KReuse::TimeReversal},
                        k_traversal:match self.symmetry.k_traversal {crate::KTraversal::Given=>blaze2d_core::symmetry::traversal::KTraversal::Given,crate::KTraversal::Nearest=>blaze2d_core::symmetry::traversal::KTraversal::Nearest},
                        max_traversal_field_bytes:self.symmetry.max_traversal_field_bytes,
                        max_sector_workspace_bytes:self.symmetry.max_sector_workspace_bytes,
                        coarse_start:self.symmetry.coarse_start.as_ref().map(|c|c.options()),
                        plane_group:self.symmetry.plane_group.as_ref().map(|s|serde_json::from_value(json!(s)).expect("validated plane group")),
                        seed_transfer:match self.symmetry.seed_transfer {crate::SeedTransfer::Copy=>blaze2d_core::symmetry::seed::SeedTransfer::Copy,crate::SeedTransfer::RankAware=>blaze2d_core::symmetry::seed::SeedTransfer::RankAware},
                        sectors:match self.symmetry.sectors {crate::SectorPolicy::Off=>blaze2d_core::symmetry::sectors::SectorPolicy::Off,crate::SectorPolicy::ConservativeMirror=>blaze2d_core::symmetry::sectors::SectorPolicy::ConservativeMirror,crate::SectorPolicy::AdaptiveMirror=>blaze2d_core::symmetry::sectors::SectorPolicy::AdaptiveMirror,crate::SectorPolicy::ConservativeGroup=>blaze2d_core::symmetry::sectors::SectorPolicy::ConservativeGroup,crate::SectorPolicy::AdaptiveGroup=>blaze2d_core::symmetry::sectors::SectorPolicy::AdaptiveGroup},
                        arithmetic:match self.symmetry.arithmetic {crate::SymmetryArithmetic::SpatialComplex=>blaze2d_core::symmetry::fourier::Arithmetic::SpatialComplex,crate::SymmetryArithmetic::FourierComplex=>blaze2d_core::symmetry::fourier::Arithmetic::FourierComplex,crate::SymmetryArithmetic::PtReal=>blaze2d_core::symmetry::fourier::Arithmetic::PtRealTe},
                        sector_kernel:match self.symmetry.sector_kernel {crate::SectorKernel::FullGrid=>blaze2d_core::symmetry::reduced::ReducedKernel::FullGrid,crate::SectorKernel::MirrorTm=>blaze2d_core::symmetry::reduced::ReducedKernel::MirrorTm,crate::SectorKernel::MirrorTmTe=>blaze2d_core::symmetry::reduced::ReducedKernel::MirrorTmTe},
                        sector_storage:match self.symmetry.sector_storage {crate::SectorStorage::FullGrid=>blaze2d_core::symmetry::reduced::SectorStorage::FullGrid,crate::SectorStorage::Orbit=>blaze2d_core::symmetry::reduced::SectorStorage::Orbit},
                        correction_projection_interval:self.symmetry.correction_projection_interval,share_sector_workspace:self.symmetry.share_sector_workspace,reconstruct_partners:self.symmetry.reconstruct_partners,sector_guard_budget:self.symmetry.sector_guard_budget,max_cache_bytes:self.symmetry.max_cache_bytes,..Default::default()
                    },
            ..Default::default()
        }
    }
}

impl crate::Discretization {
    pub(crate) fn metric_options(&self) -> Option<blaze2d_core::drivers::support_bands::MetricBallOptions> {
        use blaze2d_core::{drivers::support_bands::MetricBallOptions, symmetry::{support::SupportBudget, galerkin::GalerkinBudget}};
        match *self {
            Self::FftRectangle => None,
            Self::MetricBall { angular_cutoff, max_coefficients, max_embedding_points, max_enumerated_indices,
                max_estimated_payload_bytes, max_dense_dimension, max_allocation_rounds } => Some(MetricBallOptions {
                angular_cutoff, support_budget: SupportBudget { max_coefficients, max_embedding_points, max_enumerated_indices },
                budget: GalerkinBudget { max_estimated_payload_bytes, max_dense_dimension }, max_allocation_rounds,
            }),
        }
    }
}

impl crate::CoarseStart {
    pub(crate) fn options(&self) -> blaze2d_core::symmetry::coarse::CoarseStartOptions {
        blaze2d_core::symmetry::coarse::CoarseStartOptions {
            resolution:self.resolution,max_seed_field_bytes:self.max_seed_field_bytes,
        }
    }
}
