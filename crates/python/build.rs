//! Embed build provenance (git SHA + dirty flag) at compile time (B8).
//!
//! The values are exposed to the crate as `BLAZE2D_GIT_SHA` /
//! `BLAZE2D_GIT_DIRTY` compile-time env vars and surfaced to Python via
//! `blaze.build_info()`. If git is unavailable at build time the env vars are
//! left unset and `build_info()` falls back to a runtime `git rev-parse`,
//! defaulting to "unknown".

use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_string())
}

fn main() {
    // Re-run when HEAD moves or the index changes so the SHA stays honest.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");

    if let Some(sha) = git_output(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=BLAZE2D_GIT_SHA={}", sha);
    }
    if let Some(status) = git_output(&["status", "--porcelain"]) {
        let dirty = if status.is_empty() { "clean" } else { "dirty" };
        println!("cargo:rustc-env=BLAZE2D_GIT_DIRTY={}", dirty);
    }
}
