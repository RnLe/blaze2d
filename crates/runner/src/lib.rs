//! Native scheduling across independent configurations, with bounded delivery.

use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::thread;
use crossbeam_channel::{Receiver, bounded};
use serde::{Serialize, Deserialize};
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::{Diagnostic, Event, InterfaceResult, JobFailure, Plan, Precision, RunStatus, execute};
pub mod export;

#[derive(Debug, Serialize, Deserialize)]
pub struct Study {
    pub schema: String,
    pub config: blaze2d_interface::Config,
    pub results: Vec<blaze2d_interface::ResultRecord>,
    pub errors: Vec<JobFailure>,
    pub statistics: serde_json::Value,
}

pub fn collect(plan: Plan, options: Options) -> InterfaceResult<Study> {
    let start_time = std::time::Instant::now();
    let mut study = Study {schema: blaze2d_interface::RUN_SCHEMA.into(), config: plan.config.clone(),
        results: vec![], errors: vec![], statistics: serde_json::json!({})};
    let stream = start(plan, options)?;
    while let Some(event) = stream.next_event() {
        match event {
            Event::Result {result} => study.results.push(*result),
            Event::JobFailure {error} => study.errors.push(*error),
            Event::Terminal {status,completed,failed} => {
                study.statistics = serde_json::json!({"status":status,"completed":completed,"failed":failed,
                    "elapsed_seconds":start_time.elapsed().as_secs_f64()});
            }, _=>{}
        }
    }
    study.results.sort_by_key(|r|r.job_index);
    Ok(study)
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum ErrorPolicy { #[default] Stop, Continue }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    /// Zero selects the available CPU count, capped at the number of jobs.
    #[serde(default)]
    pub threads: usize,
    #[serde(default)]
    pub error_policy: ErrorPolicy,
    /// Zero selects twice the effective thread count.
    #[serde(default)]
    pub queue_capacity: usize,
}

pub struct RunStream {
    receiver: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    pub options: Options,
}
impl RunStream {
    pub fn next_event(&self) -> Option<Event> { self.receiver.recv().ok() }
    pub fn cancel(&self) { self.cancel.store(true, Ordering::Release); }
}
impl Drop for RunStream { fn drop(&mut self) { self.cancel(); } }

pub fn start(plan: Plan, mut options: Options) -> InterfaceResult<RunStream> {
    if options.threads > 1024 || options.queue_capacity > 65536 {
        return Err(Diagnostic::new("runner_options", "", "Use at most 1024 threads and 65536 queued events"));
    }
    options.threads = if options.threads == 0 { thread::available_parallelism().map_or(1, |n| n.get()) } else { options.threads }
        .min(plan.summary.jobs).max(1);
    if options.queue_capacity == 0 { options.queue_capacity = 2 * options.threads; }
    let (sender, receiver) = bounded(options.queue_capacity);
    let cancel = Arc::new(AtomicBool::new(false));
    let run_cancel = cancel.clone();
    let effective = options.clone();
    let plan = Arc::new(plan);
    thread::Builder::new().name("blaze-study".into()).spawn(move || {
        let _ = sender.send(Event::RunStart { jobs: plan.summary.jobs, solves: plan.summary.solves });
        let next = AtomicUsize::new(0);
        let completed = AtomicUsize::new(0);
        let failed = AtomicUsize::new(0);
        let halt = AtomicBool::new(false);
        thread::scope(|scope| {
            for _ in 0..effective.threads {
                let (sender, plan, cancel, effective) = (&sender, &plan, &run_cancel, &effective);
                let (next, completed, failed, halt) = (&next, &completed, &failed, &halt);
                scope.spawn(move || {
                    while !cancel.load(Ordering::Acquire) && !halt.load(Ordering::Acquire) {
                        let Ok(index) = next.fetch_update(Ordering::AcqRel, Ordering::Acquire,
                            |i| (i < plan.summary.jobs).then(|| i + 1)) else { break };
                        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let job = plan.job(index).map_err(|diagnostic| JobFailure { job_index: index, diagnostic, partial_result: None })?;
                            let emit = |event| { if sender.send(event).is_err() { cancel.store(true, Ordering::Release); } };
                            match job.resolved.config.eigensolver.precision {
                                Precision::F64 => execute(CpuBackend::<f64>::new(), "cpu", &job, emit),
                                Precision::F32 => execute(CpuBackend::<f32>::new(), "cpu", &job, emit),
                            }
                        })).unwrap_or_else(|panic| {
                            let msg = panic.downcast_ref::<String>().map(String::as_str)
                                .or_else(|| panic.downcast_ref::<&str>().copied()).unwrap_or("Internal solver failure");
                            Err(JobFailure { job_index: index, diagnostic: Diagnostic::new("solver", "", msg), partial_result: None })
                        });
                        let event = match outcome {
                            Ok(mut result) => {
                                completed.fetch_add(1, Ordering::Relaxed);
                                result.metadata["runner"] = serde_json::json!({"threads":effective.threads,
                                    "queue_capacity":effective.queue_capacity, "error_policy":effective.error_policy,
                                    "parallelism":"independent_configurations", "nested_threads":false});
                                Event::Result { result: Box::new(result) }
                            },
                            Err(error) => {
                                failed.fetch_add(1, Ordering::Relaxed);
                                if effective.error_policy == ErrorPolicy::Stop { halt.store(true, Ordering::Release); }
                                Event::JobFailure { error: Box::new(error) }
                            }
                        };
                        if sender.send(event).is_err() { cancel.store(true, Ordering::Release); break; }
                    }
                });
            }
        });
        let completed = completed.load(Ordering::Relaxed);
        let failed = failed.load(Ordering::Relaxed);
        let status = if run_cancel.load(Ordering::Acquire) { RunStatus::Cancelled }
            else if failed > 0 && effective.error_policy == ErrorPolicy::Stop { RunStatus::Failed }
            else if failed > 0 { RunStatus::CompletedWithErrors } else { RunStatus::Completed };
        let _ = sender.send(Event::Terminal { status, completed, failed });
    }).map_err(|e| Diagnostic::new("runner_start", "", e.to_string()))?;
    Ok(RunStream { receiver, cancel, options })
}
