#![cfg(feature = "profiling")]
use blaze2d_core::profiler::get_counters;
    use blaze2d_core::{backend::SpectralBackend, dielectric::Dielectric2D, grid::Grid2D,
        operators::{LinearOperator, ThetaOperator}, polarization::Polarization};
    use blaze2d_backend_cpu::CpuBackend;

    #[test]
    fn counts_columns_and_actual_transform_elements() {
        let grid = Grid2D::new(5, 3, 1.0, 1.0);
        let backend = CpuBackend::<f64>::new();
        let dielectric = Dielectric2D::from_geometry(&blaze2d_core::geometry::Geometry2D { lattice: blaze2d_core::lattice::Lattice2D::square(1.0), eps_bg: 2.0, atoms: vec![] }, grid, &Default::default());
        let mut op = ThetaOperator::new(backend.clone(), dielectric, Polarization::TE, [0.1, 0.2]);
        let input = backend.alloc_field(grid);
        let mut output = backend.alloc_field(grid);
        op.apply(&input, &mut output);
        op.apply_mass(&input, &mut output);
        op.batch_apply(&[input.clone(), input], &mut [output.clone(), output]);
        let counters = get_counters();
        assert_eq!(counters["operator_A.columns"], 3);
        assert_eq!(counters["operator_B.columns"], 1);
        assert_eq!(counters["fft.forward.vectors"], 9);
        assert_eq!(counters["fft.inverse.vectors"], 9);
        assert_eq!(counters["fft.forward.elements"], 135);
    }

