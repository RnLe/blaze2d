#![cfg(feature = "profiling")]

use blaze2d_core::profiler;
use blaze2d_interface::*;
use blaze2d_runner::{Options, start};
use serde_json::json;

#[test]
fn terminal_event_contains_every_workers_profile() {
    // Keep reset/aggregation isolated from other profile-producing tests.
    for continuation in [ContinuationPolicy::Off, ContinuationPolicy::PreviousGeometry] {
        profiler::reset();
        let mut config = Config::default();
        config.grid.resolution = Resolution::Uniform(12);
        config.eigensolver.tolerance = Some(1e-13);
        let bands = config.bands.as_mut().unwrap();
        bands.count = 4;
        bands.symmetry.arithmetic = SymmetryArithmetic::PtReal;
        bands.symmetry.k_reuse = KReuse::Identical;
        bands.symmetry.continuation = continuation;
        bands.path = KPath { points: Some(vec![vec![0.0,0.0], vec![0.19,0.13], vec![-0.19,-0.13], vec![0.0,0.0]]),
            intervals_per_segment: None, ..Default::default() };
        config.sweeps = vec![Sweep { name: "radius".into(), target: "geometry.objects.rod.radius".into(),
            values: Some(vec![json!(0.17),json!(0.18),json!(0.19),json!(0.20),json!(0.21),json!(0.22)]), linspace: None }];
        let stream = start(Plan::new(config, Platform::Native).unwrap(),
            Options { threads: 4, queue_capacity: 1, ..Default::default() }).unwrap();
        let mut iterations = 0;
        let mut terminal = false;
        while let Some(event) = stream.next_event() {
            match event {
                Event::Result { result } => {
                    iterations += result.metadata["iterations"].as_array().unwrap()
                        .iter().map(|v|v.as_u64().unwrap()).sum::<u64>();
                }
                Event::JobFailure { error } => panic!("{}",error.diagnostic),
                Event::Terminal { status,completed,failed } => {
                    assert_eq!((status,completed,failed),(RunStatus::Completed,6,0));
                    let counters = profiler::get_all_counters();
                    assert!(iterations > 0);
                    assert_eq!(counters["iterations"], iterations);
                    assert_eq!(counters.get("profiler.unbalanced_timers").copied().unwrap_or(0),0);
                    terminal = true;
                }
                _ => {}
            }
        }
        assert!(terminal);
    }
}
