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
        /// Suppress the final execution summary on stderr.
        #[arg(long)] quiet: bool,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long, default_value_t=0)] threads: usize,
        #[arg(long, default_value="stop", value_parser=["stop","continue"])] error_policy: String,
        /// Write explanatory counters and timings from a profiling build.
        #[arg(long)] profile_output: Option<PathBuf>,
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
        Command::Run {file,output,threads,error_policy,progress,quiet,profile_output} => {
            if profile_output.is_some() && !cfg!(feature="profiling") {
                return Err("--profile-output requires a build with the profiling feature".into());
            }
            #[cfg(feature="profiling")]
            blaze2d_core::profiler::reset();
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
            if let Some(path) = profile_output {
                std::fs::write(path, serde_json::to_vec_pretty(&serde_json::json!({
                    "timings":serde_json::from_str::<serde_json::Value>(&blaze2d_core::profiler::get_profile_json())?,
                    "counters":blaze2d_core::profiler::get_all_counters(),
                    "stages":blaze2d_core::profiler::get_all_stage_counters(),
                    "timing_role":"instrumented explanatory profile; primary latency uses an unprofiled executable"
                }))?)?;
            }
            study.statistics["runtime"]["progress"] = serde_json::json!(progress);
            study.statistics["runtime"]["output"] = serde_json::json!(output.as_ref().map(|p| p.display().to_string()));
            if !quiet {
                let summary = blaze2d_interface::summary::format_results(study.results.iter().chain(study.errors.iter().filter_map(|e|e.partial_result.as_deref())));
                if !summary.is_empty() { eprintln!("{summary}"); }
            }
            if let Some(path) = output { blaze2d_runner::export::write(&study,&path)?; }
            else { println!("{}",serde_json::to_string(&study)?); }
            return Ok(study.statistics["status"] == "completed");
        }
    }
    Ok(true)
}

/// Keep freed solver-sized blocks inside the process. Every wavevector allocates and frees
/// tens of megabytes of fields; with glibc's default thresholds they go back to the kernel
/// and the next wavevector faults in fresh pages (about a tenth of a TM band diagram).
/// Blocks below 32 MiB come from the heap, which is trimmed only above 512 MiB free.
#[cfg(all(target_os = "linux", target_env = "gnu", target_pointer_width = "64"))]
fn retain_freed_memory() {
    // SAFETY: mallopt only adjusts allocator parameters; it runs once, before any threads.
    unsafe {
        // Setting the trim threshold also freezes the mmap threshold, so it is only set
        // when glibc accepted the larger mmap threshold (at most half its heap size).
        if libc::mallopt(libc::M_MMAP_THRESHOLD, 32 << 20) == 1 {
            libc::mallopt(libc::M_TRIM_THRESHOLD, 512 << 20);
        }
    }
}

fn main() -> ExitCode {
    #[cfg(all(target_os = "linux", target_env = "gnu", target_pointer_width = "64"))]
    retain_freed_memory();
    match execute(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => { eprintln!("{error}"); ExitCode::from(2) }
    }
}
