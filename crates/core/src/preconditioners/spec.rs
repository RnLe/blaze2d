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
    #[default]
    Legacy,
    /// Blend of the legacy shift with the previous k-point's median eigenvalue.
    BandWindow,
    /// σ = `shift_value`.
    Fixed,
}

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
}

impl Default for PreconditionerSpec {
    fn default() -> Self {
        Self {
            kind: PreconditionerKind::Auto,
            shift: ShiftRule::Legacy,
            shift_fraction: super::SHIFT_SMIN_FRACTION,
            shift_value: 0.0,
            band_window_blend: super::DEFAULT_BAND_WINDOW_BLEND,
            band_window_scale: super::DEFAULT_BAND_WINDOW_SCALE,
            tm_mass_fraction: crate::operators::TM_PRECONDITIONER_MASS_FRACTION,
        }
    }
}

impl PreconditionerSpec {
    pub fn resolve_kind(&self, polarization: Polarization) -> PreconditionerKind {
        match (self.kind, polarization) {
            (PreconditionerKind::Auto, Polarization::TM) => PreconditionerKind::Diagonal,
            (PreconditionerKind::Auto, Polarization::TE) => PreconditionerKind::Transverse,
            (kind, _) => kind,
        }
    }

    pub fn validate(&self, polarization: Option<Polarization>) -> Result<(), String> {
        let finite = [self.shift_fraction, self.shift_value, self.band_window_blend, self.band_window_scale, self.tm_mass_fraction];
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

/// What a built preconditioner used at one k-point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PreconditionerInfo {
    pub kind: PreconditionerKind,
    pub shift: f64,
    pub s_min: f64,
}
