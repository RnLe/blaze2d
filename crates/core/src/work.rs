//! Always-on numerical work counters for one worker thread.
//!
//! These mirror selected profiler counters so that unprofiled builds report
//! deterministic work (iterations, operator and preconditioner columns, FFTs,
//! dense multiply terms). Snapshot differences around a solve give its work.
use serde::{Deserialize, Serialize};
use std::cell::Cell;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub iterations: u64,
    pub a_columns: u64,
    pub b_columns: u64,
    pub m_columns: u64,
    pub fft_vectors: u64,
    pub gemm_terms: u64,
    pub dense_cubes: u64,
    pub svqb_dropped: u64,
}

impl Work {
    pub fn since(self, start: Work) -> Work {
        Work {
            iterations: self.iterations.wrapping_sub(start.iterations),
            a_columns: self.a_columns.wrapping_sub(start.a_columns),
            b_columns: self.b_columns.wrapping_sub(start.b_columns),
            m_columns: self.m_columns.wrapping_sub(start.m_columns),
            fft_vectors: self.fft_vectors.wrapping_sub(start.fft_vectors),
            gemm_terms: self.gemm_terms.wrapping_sub(start.gemm_terms),
            dense_cubes: self.dense_cubes.wrapping_sub(start.dense_cubes),
            svqb_dropped: self.svqb_dropped.wrapping_sub(start.svqb_dropped),
        }
    }

    pub fn add(&mut self, other: Work) {
        self.iterations += other.iterations;
        self.a_columns += other.a_columns;
        self.b_columns += other.b_columns;
        self.m_columns += other.m_columns;
        self.fft_vectors += other.fft_vectors;
        self.gemm_terms += other.gemm_terms;
        self.dense_cubes += other.dense_cubes;
        self.svqb_dropped += other.svqb_dropped;
    }
}

thread_local! { static LOCAL: Cell<Work> = const { Cell::new(Work { iterations: 0, a_columns: 0, b_columns: 0, m_columns: 0, fft_vectors: 0, gemm_terms: 0, dense_cubes: 0, svqb_dropped: 0 }) }; }

/// Work recorded on the calling worker so far.
pub fn snapshot() -> Work { LOCAL.with(Cell::get) }

/// Add work completed by a joined child worker.
pub fn absorb(work: Work) { LOCAL.with(|w| { let mut v = w.get(); v.add(work); w.set(v); }); }

#[inline]
pub(crate) fn record(name: &str, amount: usize) {
    let amount = amount as u64;
    LOCAL.with(|w| {
        let mut v = w.get();
        match name {
            "iterations" => v.iterations += amount,
            "operator_A.columns" => v.a_columns += amount,
            "operator_B.columns" => v.b_columns += amount,
            "preconditioner.columns" => v.m_columns += amount,
            "fft.forward.vectors" | "fft.inverse.vectors" => v.fft_vectors += amount,
            "ritz_gram.multiply_terms" | "svqb_gram.multiply_terms" => v.gemm_terms += amount,
            "dense_ritz.dimension_cubes" | "real_dense_ritz.dimension_cubes" => v.dense_cubes += amount,
            "subspace.dropped_columns" => v.svqb_dropped += amount,
            _ => return,
        }
        w.set(v);
    });
}
