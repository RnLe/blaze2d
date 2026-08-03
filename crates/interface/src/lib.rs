//! Shared calculation contract for native and browser interfaces.

pub mod config;
pub mod diagnostic;

pub use config::*;
pub use diagnostic::*;

pub const CONFIG_SCHEMA: &str = "blaze2d/1";
pub const RESULT_SCHEMA: &str = "blaze2d/result/1";
pub const RUN_SCHEMA: &str = "blaze2d/run/1";

pub fn schema() -> schemars::Schema {
    schemars::schema_for!(Config)
}
