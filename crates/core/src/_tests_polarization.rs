#![cfg(test)]

use super::polarization::Polarization;

#[test]
fn serde_accepts_uppercase_variants() {
    let tm: Polarization = serde_json::from_str("\"TM\"").expect("TM should deserialize");
    let te: Polarization = serde_json::from_str("\"TE\"").expect("TE should deserialize");
    assert!(matches!(tm, Polarization::TM));
    assert!(matches!(te, Polarization::TE));
}

#[test]
fn serde_rejects_lowercase_variants() {
    let err = serde_json::from_str::<Polarization>("\"tm\"").unwrap_err();
    assert!(err.is_data());
}

#[test]
fn serde_roundtrip_preserves_variant_names() {
    let tm_json = serde_json::to_string(&Polarization::TM).expect("serialize TM");
    let te_json = serde_json::to_string(&Polarization::TE).expect("serialize TE");
    assert_eq!(tm_json, "\"TM\"");
    assert_eq!(te_json, "\"TE\"");
}
