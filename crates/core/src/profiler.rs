use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::sync::OnceLock;

// Simple global registry for profiling stats
static PROFILER: OnceLock<Mutex<HashMap<&'static str, (u64, Duration)>>> = OnceLock::new();

fn get_profiler() -> &'static Mutex<HashMap<&'static str, (u64, Duration)>> {
    PROFILER.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct ScopeTimer {
    name: &'static str,
    start: Instant,
}

impl ScopeTimer {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if let Ok(mut lock) = get_profiler().lock() {
            let entry = lock.entry(self.name).or_insert((0, Duration::ZERO));
            entry.0 += 1;
            entry.1 += elapsed;
        }
    }
}

pub fn print_stats() {
    if let Ok(lock) = get_profiler().lock() {
        let mut stats: Vec<_> = lock.iter().collect();
        // Sort by duration desc
        stats.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        
        println!("\n=== Profiling Stats ===");
        println!("{:<30} | {:<10} | {:<15} | {:<15}", "Function", "Calls", "Total Time", "Avg Time");
        println!("{}", "-".repeat(80));
        
        for (name, (count, duration)) in stats {
            let avg = if *count > 0 { *duration / *count as u32 } else { Duration::ZERO };
            println!("{:<30} | {:<10} | {:<15?} | {:<15?}", name, count, duration, avg);
        }
        println!("=======================\n");
    }
}

pub fn get_profile_json() -> String {
    if let Ok(lock) = get_profiler().lock() {
        let mut json = String::from("{");
        for (i, (name, (count, duration))) in lock.iter().enumerate() {
            if i > 0 { json.push(','); }
            let nanos = duration.as_nanos();
            json.push_str(&format!("\"{}\":{{\"calls\":{},\"nanos\":{}}}", name, count, nanos));
        }
        json.push('}');
        return json;
    }
    String::from("{}")
}


// Re-add macro
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        #[cfg(feature = "profiling")]
        let _timer = $crate::profiler::ScopeTimer::new($name);
    };
}

use std::cell::RefCell;

thread_local! {
    static ACTIVE_TIMERS: RefCell<HashMap<&'static str, Instant>> = RefCell::new(HashMap::new());
}

#[cfg(feature = "profiling")]
pub fn start_timer(name: &'static str) {
    ACTIVE_TIMERS.with(|timers| {
        timers.borrow_mut().insert(name, Instant::now());
    });
}

#[cfg(feature = "profiling")]
pub fn stop_timer(name: &'static str) {
    let start_time = ACTIVE_TIMERS.with(|timers| {
        timers.borrow_mut().remove(name)
    });

    if let Some(start) = start_time {
        let elapsed = start.elapsed();
        if let Ok(mut lock) = get_profiler().lock() {
            let entry = lock.entry(name).or_insert((0, Duration::ZERO));
            entry.0 += 1;
            entry.1 += elapsed;
        }
    }
}

#[cfg(not(feature = "profiling"))]
pub fn start_timer(_name: &'static str) {}

#[cfg(not(feature = "profiling"))]
pub fn stop_timer(_name: &'static str) {}
