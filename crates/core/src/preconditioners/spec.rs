//! Serializable preconditioner selection used by the band drivers.

use serde::{Deserialize, Serialize};

use crate::polarization::Polarization;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerKind {
    /// Fourier diagonal for TM, transverse projection for TE.
    #[default]
    Auto,
    None,
    /// Homogeneous-medium Fourier diagonal.
    Diagonal,
    /// MPB-style transverse projection (TE only).
    Transverse,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftRule {
    /// σ = `shift_fraction` · smallest |k+G|² above the near-zero threshold.
    Legacy,
    /// Blend of the legacy shift with the previous k-point's median eigenvalue.
    BandWindow,
    /// σ = `shift_value`.
    Fixed,
    /// σ = `shift_fraction` · min |k+G|² over all modes, in |k+G|² units for every
    /// kind: zero at Γ and continuous in k. For a homogeneous medium both kinds then
    /// equal the symbol of (A + σ_λ B)⁻¹ with σ_λ = σ/ε_eff.
    #[default]
    Continuous,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransverseWeight {
    /// Multiply the gradient by the sampled scalar ε.
    Scalar,
    /// Multiply by the inverse of the smoothed ε⁻¹ tensor the operator uses.
    #[default]
    Tensor,
}

/// Missing fields of a deserialized spec take the `Default` values, so a partial
/// legacy spec must state its `shift_fraction` (the interface fills in 0.5).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreconditionerSpec {
    pub kind: PreconditionerKind,
    pub shift: ShiftRule,
    pub shift_fraction: f64,
    pub shift_value: f64,
    pub band_window_blend: f64,
    pub band_window_scale: f64,
    pub tm_mass_fraction: f64,
    pub transverse_weight: TransverseWeight,
    /// Lowest plane waves (completed to whole |k+G| shells, kernel excluded) in an
    /// additive coarse correction Z (Zᴴ (A + σ_c B) Z)⁻¹ Zᴴ; 0 disables it.
    pub coarse_modes: usize,
    /// σ_c of the coarse operator, in eigenvalue units.
    pub coarse_shift: f64,
}

/// The default: no shift (the Γ-continuous rule at zero) and the tensor-weighted
/// transverse projection; on the preconditioner study matrix this needed 7% (TE) and
/// 11% (TM) fewer FFTs than `legacy()` and never more on any case.
impl Default for PreconditionerSpec {
    fn default() -> Self {
        Self {
            kind: PreconditionerKind::Auto,
            shift: ShiftRule::Continuous,
            shift_fraction: 0.0,
            shift_value: 0.0,
            band_window_blend: super::DEFAULT_BAND_WINDOW_BLEND,
            band_window_scale: super::DEFAULT_BAND_WINDOW_SCALE,
            tm_mass_fraction: crate::operators::TM_PRECONDITIONER_MASS_FRACTION,
            transverse_weight: TransverseWeight::Tensor,
            coarse_modes: 0,
            coarse_shift: 0.0,
        }
    }
}

impl PreconditionerSpec {
    /// The preconditioner used before the default changed: σ = 0.5 · s_min and the
    /// scalar-ε transverse projection.
    pub fn legacy() -> Self {
        Self { shift: ShiftRule::Legacy, shift_fraction: super::SHIFT_SMIN_FRACTION,
            transverse_weight: TransverseWeight::Scalar, ..Self::default() }
    }

    pub fn resolve_kind(&self, polarization: Polarization) -> PreconditionerKind {
        match (self.kind, polarization) {
            (PreconditionerKind::Auto, Polarization::TM) => PreconditionerKind::Diagonal,
            (PreconditionerKind::Auto, Polarization::TE) => PreconditionerKind::Transverse,
            (kind, _) => kind,
        }
    }

    pub fn validate(&self, polarization: Option<Polarization>) -> Result<(), String> {
        let finite = [self.shift_fraction, self.shift_value, self.band_window_blend, self.band_window_scale,
            self.tm_mass_fraction, self.coarse_shift];
        if finite.iter().any(|v| !v.is_finite()) {
            return Err("preconditioner settings must be finite".into());
        }
        if self.shift_fraction < 0.0 || self.band_window_scale < 0.0 || self.tm_mass_fraction < 0.0 {
            return Err("shift_fraction, band_window_scale and tm_mass_fraction must be non-negative".into());
        }
        if !(0.0..=1.0).contains(&self.band_window_blend) {
            return Err("band_window_blend must lie in [0, 1]".into());
        }
        if self.shift == ShiftRule::Fixed && self.shift_value < 0.0 {
            return Err("a fixed shift must be non-negative".into());
        }
        if self.kind == PreconditionerKind::Transverse && polarization == Some(Polarization::TM) {
            return Err("the transverse projection approximates the TE operator only".into());
        }
        if self.coarse_modes > MAX_COARSE_MODES {
            return Err(format!("coarse_modes must not exceed {MAX_COARSE_MODES}"));
        }
        if self.coarse_shift < 0.0 {
            return Err("coarse_shift must be non-negative".into());
        }
        if self.coarse_modes > 0 && self.kind == PreconditionerKind::None {
            return Err("a coarse correction needs a smoothing preconditioner".into());
        }
        Ok(())
    }

    /// The diagnostics enum for recorded runs.
    pub fn diagnostics_type(&self, polarization: Polarization) -> crate::diagnostics::PreconditionerType {
        use crate::diagnostics::PreconditionerType as T;
        match self.resolve_kind(polarization) {
            PreconditionerKind::None => T::None,
            PreconditionerKind::Transverse => T::TransverseProjection,
            _ => T::FourierDiagonalKernelCompensated,
        }
    }
}

/// Bound on the dense coarse space (its Cholesky factor is rebuilt per k-point).
pub const MAX_COARSE_MODES: usize = 512;

/// What a built preconditioner used at one k-point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PreconditionerInfo {
    pub kind: PreconditionerKind,
    pub shift: f64,
    pub s_min: f64,
    /// Plane waves in the coarse correction after shell completion (0 when off or
    /// when its Galerkin matrix was not positive definite).
    #[serde(default)]
    pub coarse_modes: usize,
}
