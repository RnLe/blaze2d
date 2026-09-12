use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    bandstructure::{self, RunOptions},
    dielectric::Dielectric2D,
    eigensolver::refine::certify_without_refinement,
    field::Field2D,
    operators::ThetaOperator,
};
use blaze2d_interface::*;
use num_complex::Complex64;

#[test]
fn fresh_measurement_detects_an_inaccurate_plane_wave_without_changing_it() {
    let mut c = Config::default();
    c.geometry.objects.clear();
    c.geometry.background_epsilon = 4.0;
    c.geometry.lattice.a = Some(2.5);
    c.grid.resolution = Resolution::Axes(vec![12, 16]);
    c.bands.as_mut().unwrap().path = KPath {
        points: Some(vec![vec![0.19, 0.13]]),
        intervals_per_segment: None,
        ..Default::default()
    };
    let job = Plan::new(c, Platform::Native).unwrap().job(0).unwrap();
    let dielectric = Dielectric2D::from_geometry(&job.geometry(), job.grid(), &job.dielectric());
    let k = job.resolved.k_points_cartesian[0];
    let mut theta = ThetaOperator::new(CpuBackend::<f64>::new(), dielectric, job.polarization(), k);
    let field = Field2D::from_f64_vec(
        job.grid(),
        vec![Complex64::new(1.0 / (4.0_f64 * 12.0 * 16.0).sqrt(), 0.0); 12 * 16],
    );
    let original = field.as_slice().to_vec();
    let eigenvalue = 1.1 * (k[0] * k[0] + k[1] * k[1]) / 4.0;
    let certificate =
        certify_without_refinement(&mut theta, &[eigenvalue], std::slice::from_ref(&field));
    assert!((certificate.residuals[0] - 1.0 / 21.0).abs() < 1e-12);
    assert!(certificate.b_orthogonality_defect < 1e-12);
    assert_eq!(field.as_slice(), original.as_slice());
}

#[test]
fn band_certification_preserves_tracked_frequencies_and_reports_unconverged_fields() {
    let mut c = Config::default();
    c.grid.resolution = Resolution::Axes(vec![12, 16]);
    c.bands.as_mut().unwrap().count = 3;
    c.bands.as_mut().unwrap().path = KPath {
        points: Some(vec![vec![0.19, 0.13], vec![0.22, 0.15]]),
        intervals_per_segment: None,
        ..Default::default()
    };
    c.eigensolver.max_iterations = Some(1);
    c.eigensolver.tolerance = Some(1e-12);
    let job = Plan::new(c, Platform::Native).unwrap().job(0).unwrap();
    let original = bandstructure::run_with_options(
        CpuBackend::<f64>::new(),
        &job.bands_job(),
        RunOptions {
            reuse_gamma: false,
            ..Default::default()
        },
    );
    let result = execute(CpuBackend::<f64>::new(), "cpu", &job, |_| {}).unwrap();
    let frequencies: Vec<_> = original
        .bands
        .into_iter()
        .flatten()
        .map(|value| value / std::f64::consts::TAU)
        .collect();
    assert_eq!(frequencies, result.arrays["frequencies"].data);
    assert_eq!(result.metadata["converged"], false);
    assert_eq!(
        result.metadata["certification"]["source"],
        "fresh_operator_application"
    );
    assert!(result.arrays["residuals"].data.iter().any(|r| *r > 1e-6));
}
