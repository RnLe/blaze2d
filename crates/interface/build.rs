use std::{env, process::Command};
fn main() {
    println!("cargo:rerun-if-env-changed=BLAZE_SOURCE_REVISION");
    println!("cargo:rerun-if-changed=source-revision.txt");
    let revision = env::var("BLAZE_SOURCE_REVISION").ok().or_else(|| {
        Command::new("git").args(["rev-parse", "HEAD"]).output().ok()
            .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }).or_else(|| std::fs::read_to_string("source-revision.txt").ok().map(|s|s.trim().to_owned()))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=BLAZE_SOURCE_REVISION={revision}");
    // Git's HEAD points at a branch; watch both the symbolic ref and its value.
    let branch = Command::new("git").args(["symbolic-ref", "-q", "HEAD"]).output().ok()
        .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    for name in [Some("HEAD".to_string()), Some("packed-refs".to_string()), branch].into_iter().flatten() {
        if let Ok(out) = Command::new("git").args(["rev-parse", "--git-path", &name]).output() {
            if out.status.success() { println!("cargo:rerun-if-changed={}", String::from_utf8_lossy(&out.stdout).trim()); }
        }
    }
}
