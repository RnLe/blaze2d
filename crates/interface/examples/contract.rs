use blaze2d_interface::*;
use schemars::JsonSchema;

#[allow(dead_code)]
#[derive(JsonSchema)]
struct BrowserContract {
    config: Config,
    validation: ValidationReport,
    result: ResultRecord,
    event: Event,
    job: PlannedJob,
}

fn main() {
    println!("{}",serde_json::to_string_pretty(&schemars::schema_for!(BrowserContract)).unwrap());
}
