//! Shared calculation contract for native and browser interfaces.

pub mod config;
pub mod diagnostic;
pub mod normalize;
pub mod plan;

pub use config::*;
pub use diagnostic::*;
pub use normalize::{ResolvedConfig, reciprocal_vectors, fractional_to_cartesian, cartesian_to_fractional};
pub use plan::{Plan, PlannedJob, PlanSummary, Platform, capabilities, set_parameter, validate_toml, ValidationReport};

pub const CONFIG_SCHEMA: &str = "blaze2d/1";
pub const RESULT_SCHEMA: &str = "blaze2d/result/1";
pub const RUN_SCHEMA: &str = "blaze2d/run/1";

pub fn schema() -> schemars::Schema {
    schemars::schema_for!(Config)
}
