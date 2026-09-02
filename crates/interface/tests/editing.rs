use blaze2d_interface::*;

#[test]
fn structured_changes_preserve_comments_and_unmodified_representations() {
    let source = include_str!("fixtures/bands.toml").replace("radius = 0.20","radius = 0.20 # physical radius");
    let source = format!("# Authored calculation\n{source}\n# End of study\n");
    let mut candidate = Config::from_toml(&source).unwrap();
    candidate.geometry.objects[0].radius = 0.25;
    let edited = edit::apply_config(&source,candidate,Platform::Browser).unwrap();
    assert!(edited.starts_with("# Authored calculation\n"));
    assert!(edited.contains("# physical radius"));
    assert!(edited.ends_with("# End of study\n"));
    assert_eq!(Config::from_toml(&edited).unwrap().geometry.objects[0].radius,0.25);
    assert!(!edited.contains("[eigensolver]"));
}

#[test]
fn no_change_preserves_source_exactly_and_invalid_changes_fail() {
    let source = include_str!("fixtures/bands.toml");
    let c = Config::from_toml(source).unwrap();
    assert_eq!(edit::apply_config(source,c.clone(),Platform::Native).unwrap(),source);
    let mut invalid=c;invalid.geometry.objects[0].epsilon=-1.0;
    assert!(edit::apply_config(source,invalid,Platform::Native).is_err());
}
