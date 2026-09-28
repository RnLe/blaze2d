//! Packed real coordinates on an explicitly selected Galerkin support.
//! Qualified PT is diagonal in Fourier coordinates, so restriction to any
//! admitted subset commutes with its real gauge. The finite pencil is unchanged.
use super::*;
use crate::symmetry::{
    galerkin::{
        self, GalerkinApplication, GalerkinBudget, GalerkinOperator, GalerkinPreconditioner,
        GalerkinResult,
    },
    support::FourierSupport,
};

struct PackedOperator<'a, 'b, B: SpectralBackend + Clone> {
    inner: &'a mut GalerkinOperator<'b, B>,
    backend: CoordinateBackend,
    conversion: Conversion<B>,
    input: B::Buffer,
    output: B::Buffer,
}

impl<B: SpectralBackend + Clone> LinearOperator<CoordinateBackend> for PackedOperator<'_, '_, B> {
    fn apply(&mut self, x: &CoordinateBuffer, y: &mut CoordinateBuffer) {
        self.conversion
            .unpack_fourier(x.as_slice(), &mut self.input);
        self.inner.apply(&self.input, &mut self.output);
        self.conversion
            .pack_fourier(&self.output, y.as_mut_slice(), 1.0, true);
    }
    fn apply_mass(&mut self, x: &CoordinateBuffer, y: &mut CoordinateBuffer) {
        self.conversion
            .unpack_fourier(x.as_slice(), &mut self.input);
        self.inner.apply_mass(&self.input, &mut self.output);
        self.conversion
            .pack_fourier(&self.output, y.as_mut_slice(), 1.0, true);
    }
    fn alloc_field(&self) -> CoordinateBuffer {
        self.backend.alloc_field(self.grid())
    }
    fn backend(&self) -> &CoordinateBackend {
        &self.backend
    }
    fn backend_mut(&mut self) -> &mut CoordinateBackend {
        &mut self.backend
    }
    fn grid(&self) -> Grid2D {
        self.conversion.coordinate_grid()
    }
    fn bloch(&self) -> [f64; 2] {
        self.inner.bloch()
    }
    fn subspace_dimension(&self) -> usize {
        self.inner.subspace_dimension()
    }
    fn owns_gamma_kernel(&self) -> bool {
        self.inner.owns_gamma_kernel()
    }
    fn analytic_kernel_available(&self) -> bool {
        self.inner.analytic_kernel_available()
    }
    fn fill_gamma_kernel(&self, vector: &mut CoordinateBuffer) {
        let mut kernel = self.inner.alloc_field();
        self.inner.fill_gamma_kernel(&mut kernel);
        self.conversion
            .pack_fourier(&kernel, vector.as_mut_slice(), 1.0, false);
    }
    fn seed_initial_vector(&mut self, vector: &mut CoordinateBuffer, phase: f64) {
        // Independent real Fourier coordinates retain both inversion parities.
        // Taking Re(FFT(real spatial noise)) would exclude all odd modes at Γ.
        crate::eigensolver::initialization::seed_random_vector(self.input.as_mut_slice(), phase);
        vector.as_mut_slice().fill(C::ZERO);
        for (i, z) in self.input.as_slice().iter().enumerate() {
            set_scalar(vector.as_mut_slice(), i, z.re.to_accum(), true);
        }
    }
}

struct PackedPreconditioner<'a, B: SpectralBackend + Clone> {
    inner: GalerkinPreconditioner<'a, B>,
    conversion: Conversion<B>,
    scratch: B::Buffer,
}
impl<B: SpectralBackend + Clone> OperatorPreconditioner<CoordinateBackend>
    for PackedPreconditioner<'_, B>
{
    fn apply(&mut self, _: &CoordinateBackend, vector: &mut CoordinateBuffer) {
        self.conversion
            .unpack_fourier(vector.as_slice(), &mut self.scratch);
        self.inner
            .apply(&self.conversion.backend, &mut self.scratch);
        self.conversion
            .pack_fourier(&self.scratch, vector.as_mut_slice(), 1.0, true);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn solve<B: SpectralBackend + Clone>(
    full: &mut ThetaOperator<B>,
    support: &FourierSupport,
    config: &EigensolverConfig,
    warm_physical: Option<&[Field2D]>,
    options: &SymmetryOptions,
    preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    budget: GalerkinBudget,
) -> Result<(GalerkinResult, CoordinateReport), String> {
    crate::profile_scope!("symmetry.galerkin.pt");
    let n = support.len();
    let bytes = galerkin::admit(support, config, false, budget)?
        .checked_add(
            support
                .grid()
                .len()
                .checked_mul(256)
                .ok_or("PT qualification payload overflow")?,
        )
        .ok_or("PT Galerkin payload overflow")?;
    if bytes > budget.max_estimated_payload_bytes {
        return Err("PT Galerkin qualification exceeds its payload allowance".into());
    }
    let setup = CoordinateSetup::qualify(full, &support.lattice(), support.k(), options)?;
    if !setup.real() {
        return Err("projected PT solve requires an explicitly real arithmetic policy".into());
    }
    let conversion = Conversion {
        backend: full.backend().clone(),
        grid: support.coordinate_grid(),
        gauge: Arc::new(
            support
                .embedding_indices()
                .iter()
                .map(|&i| setup.gauge[i])
                .collect(),
        ),
        real_space: true,
        packed: true,
        leakage: Rc::new(Cell::new(0.0)),
    };
    let backend = CoordinateBackend {
        ledger: Arc::default(),
        nreal: n,
        real_space: true,
        packed: true,
        real_vectors: true,
        real_gram: true,
        real_dense: true,
    };
    let application = GalerkinApplication::DirectFourier;
    let mut preconditioner = preconditioner
        .map(|inner| {
            Ok::<_, String>(PackedPreconditioner {
                inner: GalerkinPreconditioner::new(inner, full, support, application)?,
                conversion: conversion.clone(),
                scratch: full.backend().alloc_field(support.coordinate_grid()),
            })
        })
        .transpose()?;
    let mut operator = GalerkinOperator::new(
        full,
        support,
        application,
        budget.max_estimated_payload_bytes,
    )?;
    let warm = warm_physical
        .map(|v| operator.project_physical(v, config.effective_block_size().min(n)))
        .transpose()?;
    let warm = warm.map(|columns| {
        let mut result = Vec::with_capacity(columns.len() * 2);
        for column in columns {
            let mut real = Field2D::zeros(conversion.coordinate_grid());
            let mut imaginary = real.clone();
            for (i, z) in column.as_slice().iter().enumerate() {
                let value = z * conversion.gauge[i].conj();
                set_scalar(real.as_mut_slice(), i, value.re, true);
                set_scalar(imaginary.as_mut_slice(), i, value.im, true);
            }
            result.push(real);
            result.push(imaginary);
        }
        result
    });
    let input = operator.alloc_field();
    let output = operator.alloc_field();
    let mut packed = PackedOperator {
        inner: &mut operator,
        backend: backend.clone(),
        conversion: conversion.clone(),
        input,
        output,
    };
    let mut solver = Eigensolver::new(
        &mut packed,
        config.clone(),
        preconditioner
            .as_mut()
            .map(|p| p as &mut dyn OperatorPreconditioner<CoordinateBackend>),
        warm.as_deref(),
    );
    let result = solver.solve();
    let coordinates = solver.all_eigenvectors();
    drop(solver);
    drop(packed);
    let mut scratch = operator.alloc_field();
    let coefficients = coordinates
        .into_iter()
        .map(|field| {
            conversion.unpack_fourier(field.as_slice(), &mut scratch);
            Field2D::from_vec(
                support.coordinate_grid(),
                scratch
                    .as_slice()
                    .iter()
                    .map(|z| C::new(z.re.to_accum(), z.im.to_accum()))
                    .collect(),
            )
        })
        .collect();
    let mut report = galerkin::report(&operator, bytes, "pt_real_lobpcg");
    report.iterations = result.iterations;
    report.solver_converged = result.converged;
    let spectrum = galerkin::finish_preserving_labels(
        &mut operator,
        result.eigenvalues,
        coefficients,
        config,
        report,
    );
    if !conversion.leakage.get().is_finite()
        || conversion.leakage.get() > options.qualification_tolerance
    {
        return Err(format!(
            "projected PT real subspace leakage: {:e}",
            conversion.leakage.get()
        ));
    }
    if !spectrum.report.accepted {
        return Err(spectrum
            .report
            .rejection
            .clone()
            .unwrap_or_else(|| "projected PT certificate failed".into()));
    }
    let report = CoordinateReport {
        representation: Arithmetic::PtRealTe,
        physical_coefficients: n,
        bytes_per_column: conversion.coordinate_grid().len() * 16,
        peak_solver_vector_bytes: backend.ledger.peak.load(Ordering::Relaxed),
        conversion_scratch_bytes: support.grid().len() * 48
            + n * 16 * (3 + usize::from(preconditioner.is_some())),
        maximum_real_subspace_leakage: conversion.leakage.get(),
        final_sector_projection_defect: None,
        qualification: setup.qualification,
    };
    Ok((spectrum, report))
}
