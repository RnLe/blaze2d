use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{backend::SpectralBackend,field::{Field2D,FieldScalar},grid::Grid2D,polarization::Polarization,
    symmetry::{affine::{AffineOp,PlaneGroup,Rational},action::BlochActionPlan}};

fn field(grid:Grid2D)->Field2D {Field2D::from_vec(grid,(0..grid.len()).map(|j|FieldScalar::new((j as f64*1.73).sin(),(j as f64*0.93).cos())).collect())}
fn defect(a:&Field2D,b:&Field2D)->f64 {a.as_slice().iter().zip(b.as_slice()).map(|(x,y)|(*x-*y).norm_sqr()).sum::<f64>().sqrt()/a.as_slice().iter().map(|x|x.norm_sqr()).sum::<f64>().sqrt().max(1e-30)}

#[test]
fn spatial_and_fourier_actions_agree_including_axial_sign_glide_and_folding() {
    let backend=CpuBackend::<f64>::new();
    let mut cases=Vec::new();
    for op in PlaneGroup::P4m.group().operations {cases.push((op,[0.13,0.21],op.map_wavevector([0.13,0.21]),Grid2D::new(7,7,1.0,1.0)));}
    cases.push((PlaneGroup::Pg.group().operations[1],[0.5,0.0],[0.5,0.0],Grid2D::new(8,7,1.0,1.0)));
    cases.push((AffineOp::new([[-1,0],[0,1]],[Rational::new(1,3).unwrap(),Rational::ZERO]).unwrap(),[0.0;2],[0.0;2],Grid2D::new(9,7,1.0,1.0)));
    cases.push((AffineOp::new([[-1,0],[0,1]],[Rational::ZERO;2]).unwrap(),[-0.5,0.0],[-0.5,0.0],Grid2D::new(8,7,1.0,1.0)));
    cases.push((PlaneGroup::P2.group().operations[1].with_conjugation(true),[0.13,0.21],[0.13,0.21],Grid2D::new(8,6,1.0,1.0)));
    for pol in [Polarization::TE,Polarization::TM] {for (op,src,dst,grid) in &cases {
        let plan=BlochActionPlan::new(*op,*src,*dst,*grid,pol,1e-12).unwrap();let input=field(*grid);let mut actual=Field2D::zeros(*grid);plan.apply_spatial(&input,&mut actual);
        let mut coefficients=input.clone();backend.forward_fft_2d(&mut coefficients);let mut expected=Field2D::zeros(*grid);plan.apply_fourier(&coefficients,&mut expected);backend.inverse_fft_2d(&mut expected);
        assert!(defect(&expected,&actual)<1e-13,"{op:?} {src:?} {dst:?} {pol:?}");
    }}
}

#[test]
fn glide_square_has_bloch_translation_phase() {
    let grid=Grid2D::new(8,7,1.0,1.0);let input=field(grid);let mut once=Field2D::zeros(grid);let mut twice=once.clone();
    for kx in [0.0,0.23,0.5] {
        let plan=BlochActionPlan::new(PlaneGroup::Pg.group().operations[1],[kx,0.0],[kx,0.0],grid,Polarization::TM,1e-12).unwrap();
        plan.apply_spatial(&input,&mut once);plan.apply_spatial(&once,&mut twice);
        let phase=FieldScalar::from_polar(1.0,-std::f64::consts::TAU*kx);let expected=Field2D::from_vec(grid,input.as_slice().iter().map(|&x|phase*x).collect());
        assert!(defect(&expected,&twice)<1e-14);
    }
}

#[test]
fn refuses_an_unimplemented_centered_sampling_origin() {
    let grid=Grid2D::new_centered(8,8,1.0,1.0);
    assert!(matches!(BlochActionPlan::new(AffineOp::IDENTITY,[0.0;2],[0.0;2],grid,Polarization::TM,1e-12),Err(blaze2d_core::symmetry::action::ActionError::UnsupportedSamplingOrigin)));
}

#[test]
fn translated_antiunitary_actions_keep_their_actual_square_and_semilinearity() {
    let half = Rational::new(1, 2).unwrap();
    let grid = Grid2D::new(8, 7, 1.0, 1.0);
    let backend = CpuBackend::<f64>::new();
    // Positive Nyquist storage closes the folded boundary at kx = -1/2.
    // Translation-time-reversal and glide-time-reversal square to a primitive
    // translation, hence -I here. Shifted PT instead squares to +I.
    let cases = [
        ([[1, 0], [0, 1]], [-0.5, 0.0], -1.0),
        ([[1, 0], [0, -1]], [-0.5, 0.17], -1.0),
        ([[-1, 0], [0, -1]], [0.13, 0.21], 1.0),
    ];
    for pol in [Polarization::TM, Polarization::TE] {
        for (matrix, k, square) in cases {
            let op = AffineOp::new(matrix, [half, Rational::ZERO])
                .unwrap().with_conjugation(true);
            let plan = BlochActionPlan::new(op, k, k, grid, pol, 1e-12).unwrap();
            for fourier in [false, true] {
                let mut input = field(grid);
                if fourier { backend.forward_fft_2d(&mut input); }
                let apply = |source: &Field2D, target: &mut Field2D| {
                    if fourier { plan.apply_fourier(source, target); }
                    else { plan.apply_spatial(source, target); }
                };
                let mut once = Field2D::zeros(grid);
                let mut twice = Field2D::zeros(grid);
                apply(&input, &mut once); apply(&once, &mut twice);
                let expected = Field2D::from_vec(grid,
                    input.as_slice().iter().map(|&z| square*z).collect());
                assert!(defect(&expected, &twice) < 2e-13, "{op:?} {pol:?} fourier={fourier}");
                let scalar = FieldScalar::new(0.37, 0.61);
                let scaled = Field2D::from_vec(grid,
                    input.as_slice().iter().map(|&z| scalar*z).collect());
                apply(&scaled, &mut twice);
                let expected = Field2D::from_vec(grid,
                    once.as_slice().iter().map(|&z| scalar.conj()*z).collect());
                assert!(defect(&expected, &twice) < 2e-13);
            }
        }
    }
}
