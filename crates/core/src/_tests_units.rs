#![cfg(test)]

use super::units::{SPEED_OF_LIGHT, Units};

#[test]
fn speed_of_light_matches_si_definition() {
    // Exact integer definition since 1983; make sure we keep the canonical value.
    assert!((SPEED_OF_LIGHT - 299_792_458.0).abs() < f64::EPSILON);
}

#[test]
fn default_units_have_zero_scales() {
    let units = Units::default();
    assert_eq!(units.length_scale, 0.0);
    assert_eq!(units.frequency_scale, 0.0);
}

#[test]
fn units_store_custom_scales() {
    let units = Units {
        length_scale: 1e-6,
        frequency_scale: 2.0 * std::f64::consts::PI * 100e12,
    };
    assert!((units.length_scale - 1e-6).abs() < f64::EPSILON);
    assert!((units.frequency_scale - 2.0 * std::f64::consts::PI * 100e12).abs() < f64::EPSILON);
}
