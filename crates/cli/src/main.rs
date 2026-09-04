use std::{path::PathBuf, process::ExitCode};
use clap::{Parser, Subcommand};
use blaze2d_interface::{Config, Plan, Platform};

#[derive(Parser)]
#[command(name="blaze2d", version, about="Photonic bands and projected operators")]
struct Cli { #[command(subcommand)] command: Command }

#[derive(Subcommand)]
enum Command {
    /// Run a TOML calculation or parameter study.
    Run {
        file: PathBuf,
        #[arg(long)] progress: bool,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long, default_value_t=0)] threads: usize,
        #[arg(long, default_value="stop", value_parser=["stop","continue"])] error_policy: String,
    },
    /// Validate and inspect the shared calculation contract.
    Config { #[command(subcommand)] command: ConfigCommand },
}

#[derive(Subcommand)]
enum ConfigCommand { Validate {file:PathBuf}, Normalize {file:PathBuf}, Describe }

fn plan(path: &PathBuf) -> Result<Plan, Box<dyn std::error::Error>> {
    Ok(Plan::new(Config::from_toml(&std::fs::read_to_string(path)?)?, Platform::Native)?)
}

fn execute(cli: Cli) -> Result<bool, Box<dyn std::error::Error>> {
    match cli.command {
        Command::Config {command:ConfigCommand::Describe} => println!("{}",serde_json::to_string_pretty(&blaze2d_interface::schema())?),
        Command::Config {command:ConfigCommand::Validate {file}} => println!("{}",serde_json::to_string_pretty(&plan(&file)?.summary)?),
        Command::Config {command:ConfigCommand::Normalize {file}} => print!("{}",plan(&file)?.config.to_toml()?),
        Command::Run {file,output,threads,error_policy,progress} => {
            let mut study = blaze2d_runner::collect_with_progress(plan(&file)?, blaze2d_runner::Options {threads,
                error_policy: if error_policy=="stop" {blaze2d_runner::ErrorPolicy::Stop} else {blaze2d_runner::ErrorPolicy::Continue},
                ..Default::default()}, |event| {
                    if progress {
                        match event {
                            blaze2d_interface::Event::Result {result} => eprintln!("Job {} completed", result.job_index),
                            blaze2d_interface::Event::JobFailure {error} => eprintln!("Job {} failed: {}",error.job_index,error.diagnostic),
                            _ => {}
                        }
                    }
                })?;
            study.statistics["runtime"]["progress"] = serde_json::json!(progress);
            study.statistics["runtime"]["output"] = serde_json::json!(output.as_ref().map(|p| p.display().to_string()));
            if let Some(path) = output { blaze2d_runner::export::write(&study,&path)?; }
            else { println!("{}",serde_json::to_string(&study)?); }
            return Ok(study.statistics["status"] == "completed");
        }
    }
    Ok(true)
}

fn main() -> ExitCode {
    match execute(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => { eprintln!("{error}"); ExitCode::from(2) }
    }
}
