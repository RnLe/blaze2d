//! Per-worker inclusive/exclusive timing and numerical work accounting.
//!
//! Profiling builds accumulate in thread-local storage. A worker publishes once
//! when it exits (or explicitly calls `flush_worker`); snapshots include the
//! calling worker and published workers. Join live workers before taking a full
//! run snapshot. Hot events never acquire the aggregation mutex.
use std::{cell::RefCell, collections::BTreeMap, sync::{Mutex, OnceLock}, time::Duration};
#[cfg(not(target_arch="wasm32"))]
use std::time::Instant;
#[cfg(target_arch="wasm32")]
use web_time::Instant;
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct TimingStats { pub calls: u64, pub nanos: u128, pub exclusive_nanos: u128 }
#[derive(Default)]
#[cfg_attr(not(feature="profiling"), allow(dead_code))]
struct Frame { name: &'static str, start: Option<Instant>, child: Duration }
#[derive(Default)]
struct Worker {
    timings: BTreeMap<&'static str, TimingStats>,
    counters: BTreeMap<&'static str,u64>,
    stages: BTreeMap<&'static str,BTreeMap<&'static str,u64>>,
    stack: Vec<Frame>,
}
#[derive(Default)]
struct Aggregate { timings: BTreeMap<&'static str,TimingStats>, counters: BTreeMap<&'static str,u64>, stages: BTreeMap<&'static str,BTreeMap<&'static str,u64>> }
static COMPLETED: OnceLock<Mutex<Aggregate>> = OnceLock::new();
fn completed() -> &'static Mutex<Aggregate> { COMPLETED.get_or_init(||Mutex::new(Aggregate::default())) }
thread_local! { static LOCAL: RefCell<Worker> = RefCell::new(Worker::default()); }
fn merge(a: &mut Aggregate, w: &Worker) {
    for (&name,s) in &w.timings { let d=a.timings.entry(name).or_default(); d.calls+=s.calls; d.nanos+=s.nanos; d.exclusive_nanos+=s.exclusive_nanos; }
    for (&name,&n) in &w.counters { *a.counters.entry(name).or_default()+=n; }
    for (&stage, counters) in &w.stages { for (&name,&n) in counters { *a.stages.entry(stage).or_default().entry(name).or_default()+=n; } }
}
impl Drop for Worker { fn drop(&mut self) { if let Ok(mut a)=completed().lock() { merge(&mut a,self); } } }

pub struct ScopeTimer { name: &'static str }
impl ScopeTimer { pub fn new(name: &'static str) -> Self { start_timer(name); Self {name} } }
impl Drop for ScopeTimer { fn drop(&mut self) { stop_timer(self.name); } }

#[inline]
pub fn start_timer(name: &'static str) {
    #[cfg(feature="profiling")]
    LOCAL.with(|w|w.borrow_mut().stack.push(Frame {name,start:Some(Instant::now()),child:Duration::ZERO}));
    #[cfg(not(feature="profiling"))] let _=name;
}
#[inline]
pub fn stop_timer(name: &'static str) {
    #[cfg(feature="profiling")]
    LOCAL.with(|w| {
        let mut w=w.borrow_mut();
        if w.stack.last().is_none_or(|frame|frame.name!=name) {
            *w.counters.entry("profiler.unbalanced_timers").or_default()+=1;
            return;
        }
        let frame=w.stack.pop().unwrap(); let elapsed=frame.start.unwrap().elapsed();
        let s=w.timings.entry(name).or_default();s.calls+=1;s.nanos+=elapsed.as_nanos();s.exclusive_nanos+=elapsed.saturating_sub(frame.child).as_nanos();
        if let Some(parent)=w.stack.last_mut() { parent.child+=elapsed; }
    });
    #[cfg(not(feature="profiling"))] let _=name;
}
#[inline]
pub fn count(name: &'static str, amount: usize) {
    #[cfg(feature="profiling")]
    LOCAL.with(|w| {
        let mut w=w.borrow_mut(); *w.counters.entry(name).or_default()+=amount as u64;
        // These stage counts are inclusive; the top-level total is counted once.
        let Worker {stack,stages,..}=&mut *w;
        for frame in stack { *stages.entry(frame.name).or_default().entry(name).or_default()+=amount as u64; }
    });
    #[cfg(not(feature="profiling"))] let _=(name,amount);
}
/// Snapshot of this worker. Suitable for differences around one serial solve.
pub fn get_counters() -> BTreeMap<&'static str,u64> { LOCAL.with(|w|w.borrow().counters.clone()) }
pub fn get_stage_counters() -> BTreeMap<&'static str,BTreeMap<&'static str,u64>> { LOCAL.with(|w|w.borrow().stages.clone()) }
/// Publish this worker once; call only outside timed scopes.
pub fn flush_worker() { LOCAL.with(|w| { let mut w=w.borrow_mut(); assert!(w.stack.is_empty(),"cannot flush active scopes"); if let Ok(mut a)=completed().lock() { merge(&mut a,&w); w.timings.clear();w.counters.clear();w.stages.clear(); } }); }
/// Reset outside active scopes and after joining other workers, before a new run.
pub fn reset() { LOCAL.with(|w| { let mut w=w.borrow_mut();assert!(w.stack.is_empty(),"cannot reset active scopes");w.timings.clear();w.counters.clear();w.stages.clear(); }); *completed().lock().unwrap()=Aggregate::default(); }
pub fn get_profile_json() -> String {
    LOCAL.with(|w| {
        let w=w.borrow();let a=completed().lock().unwrap(); let mut snapshot=Aggregate {timings:a.timings.clone(),counters:a.counters.clone(),stages:a.stages.clone()};merge(&mut snapshot,&w);
        serde_json::to_string(&snapshot.timings).expect("serializable profile")
    })
}
pub fn get_all_counters() -> BTreeMap<&'static str,u64> { let mut a=completed().lock().unwrap().counters.clone();for (name,n) in get_counters() { *a.entry(name).or_default()+=n; }a }
/// Inclusive stage work from joined workers and the calling worker. A parent
/// makespan scope does not inherit child-thread counters; worker stages do.
pub fn get_all_stage_counters() -> BTreeMap<&'static str,BTreeMap<&'static str,u64>> {
    let mut a=completed().lock().unwrap().stages.clone();
    for (stage,counters) in get_stage_counters() {for (name,n) in counters {*a.entry(stage).or_default().entry(name).or_default()+=n;}}
    a
}
pub fn print_stats() { println!("{}",get_profile_json()); }
#[macro_export]
macro_rules! profile_scope { ($name:expr) => { #[cfg(feature="profiling")] let _timer=$crate::profiler::ScopeTimer::new($name); }; }

#[cfg(all(test,feature="profiling"))]
mod tests {
    use super::*;
    #[test]
    fn nested_names_workers_and_reset() {
        reset();
        start_timer("outer");start_timer("outer");count("test.columns",3);stop_timer("outer");stop_timer("outer");
        for _ in 0..3 { std::thread::spawn(|| { let _t=ScopeTimer::new("worker");count("test.columns",2); }).join().unwrap(); }
        let v:serde_json::Value=serde_json::from_str(&get_profile_json()).unwrap();
        assert_eq!(v["outer"]["calls"],2);assert_eq!(v["worker"]["calls"],3);
        assert!(v["outer"]["exclusive_nanos"].as_u64().unwrap()<=v["outer"]["nanos"].as_u64().unwrap());
        assert_eq!(get_all_counters()["test.columns"],9);
        assert_eq!(get_all_stage_counters()["worker"]["test.columns"],6);
        flush_worker();assert_eq!(get_all_counters()["test.columns"],9);
        reset();assert_eq!(get_profile_json(),"{}");assert!(get_all_counters().is_empty());
    }
}
