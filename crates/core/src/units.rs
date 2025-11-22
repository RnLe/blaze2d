//! Physical units and handy constants.

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Placeholder for future unit conversions.
#[derive(Debug, Clone, Copy, Default)]
pub struct Units {
    pub length_scale: f64,
    pub frequency_scale: f64,
}
