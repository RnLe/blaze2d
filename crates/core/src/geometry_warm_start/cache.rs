//! Worker-local rolling history. Donors remain immutable while borrowed.
use super::innovation::Innovations;
use super::snapshot::{BasisGeneration, FiberKey, NativeSnapshot, Quality};
use crate::field::Real;
use serde::Serialize;
use std::{collections::VecDeque, sync::Arc};

#[derive(Debug, Clone, Default, Serialize)]
pub struct BankReport {
    pub stored: bool,
    pub entries: usize,
    pub payload_bytes: usize,
    pub innovation_columns: usize,
    pub innovation_payload_bytes: usize,
    pub innovation_trimmed_columns: usize,
    pub evicted: Vec<BasisGeneration>,
    pub reason: Option<String>,
}

pub struct NativeBank<R: Real> {
    entries: VecDeque<Entry<R>>,
    limit: usize,
    bytes: usize,
    history_per_fiber: usize,
    generation: u64,
}
struct Entry<R: Real> {
    snapshot: Arc<NativeSnapshot<R>>,
    innovation: Option<Arc<Innovations<R>>>,
}
impl<R: Real> std::ops::Deref for Entry<R> {
    type Target = NativeSnapshot<R>;
    fn deref(&self) -> &Self::Target {
        &self.snapshot
    }
}
impl<R: Real> Entry<R> {
    fn payload_bytes(&self) -> usize {
        self.snapshot.payload_bytes() + self.innovation.as_ref().map_or(0, |i| i.payload_bytes())
    }
}
impl<R: Real> NativeBank<R> {
    pub fn new(limit: usize, history_per_fiber: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            limit,
            bytes: 0,
            history_per_fiber: history_per_fiber.max(1),
            generation: 0,
        }
    }
    pub fn payload_bytes(&self) -> usize {
        self.bytes
    }
    pub fn innovation_payload(&self) -> (usize, usize) {
        self.entries
            .iter()
            .filter_map(|e| e.innovation.as_ref())
            .fold((0, 0), |(columns, bytes), i| {
                (columns + i.fields().len(), bytes + i.payload_bytes())
            })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
    pub fn next_generation(&mut self, job: usize, k_index: usize) -> BasisGeneration {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("snapshot generation overflow");
        BasisGeneration {
            job,
            k_index,
            accepted_generation: self.generation,
        }
    }
    pub fn donors(&self, key: &FiberKey, count: usize) -> Vec<Arc<NativeSnapshot<R>>> {
        self.entries
            .iter()
            .rev()
            .filter(|x| x.key == *key)
            .take(count)
            .map(|e| e.snapshot.clone())
            .collect()
    }
    pub fn innovations_for(&self, source: &NativeSnapshot<R>) -> Option<Arc<Innovations<R>>> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.basis_generation() == source.basis_generation() && e.key() == source.key())
            .and_then(|e| e.innovation.clone())
            .filter(|i| i.matches(source))
    }
    pub fn resize(&mut self, limit: usize, history_per_fiber: usize) {
        self.limit = limit;
        self.history_per_fiber = history_per_fiber.max(1);
        let mut kept: Vec<(FiberKey, usize)> = Vec::new();
        let mut remove = Vec::new();
        for (i, snapshot) in self.entries.iter().enumerate().rev() {
            if let Some((_, count)) = kept.iter_mut().find(|(key, _)| *key == snapshot.key) {
                *count += 1;
                if *count > self.history_per_fiber {
                    remove.push(i);
                }
            } else {
                kept.push((snapshot.key.clone(), 1));
            }
        }
        for i in remove {
            self.bytes -= self.entries.remove(i).unwrap().payload_bytes();
        }
        while self.bytes > limit {
            let index = self
                .entries
                .iter()
                .position(|s| !s.key.is_gamma())
                .unwrap_or(0);
            self.bytes -= self.entries.remove(index).unwrap().payload_bytes();
        }
    }
    /// Commit only a directly accepted target. Failed candidates do not evict
    /// the last good donor. Gamma history is protected from unrelated k writes.
    pub fn insert(&mut self, snapshot: NativeSnapshot<R>, quality: Quality) -> BankReport {
        self.insert_with_innovation(snapshot, quality, None)
    }
    pub fn insert_with_innovation(
        &mut self,
        snapshot: NativeSnapshot<R>,
        quality: Quality,
        innovation: Option<Innovations<R>>,
    ) -> BankReport {
        self.insert_shared(Arc::new(snapshot), quality, innovation)
    }
    pub(crate) fn insert_shared(
        &mut self,
        snapshot: Arc<NativeSnapshot<R>>,
        quality: Quality,
        mut innovation: Option<Innovations<R>>,
    ) -> BankReport {
        let mut report = BankReport {
            entries: self.len(),
            payload_bytes: self.bytes,
            innovation_columns: self.innovation_payload().0,
            innovation_payload_bytes: self.innovation_payload().1,
            ..Default::default()
        };
        if !snapshot.admitted(quality) {
            report.reason =
                Some("source block failed convergence, count or direct accuracy admission".into());
            return report;
        }
        if innovation
            .as_ref()
            .is_some_and(|i| !i.matches(&snapshot) || i.fields().is_empty())
        {
            innovation = None;
        }
        let protected = if snapshot.key.is_gamma() {
            0
        } else {
            self.entries
                .iter()
                .filter(|e| e.key.is_gamma())
                .map(Entry::payload_bytes)
                .sum()
        };
        if let Some(frame) = innovation.as_mut() {
            let prepared = frame.fields().len();
            frame.fit_payload(
                self.limit
                    .saturating_sub(snapshot.payload_bytes())
                    .saturating_sub(protected),
            );
            report.innovation_trimmed_columns = prepared - frame.fields().len();
            if report.innovation_trimmed_columns > 0 {
                report.reason = Some(
                    "innovation rank reduced to preserve the newest full snapshot within budget"
                        .into(),
                );
            }
        }
        if innovation.as_ref().is_some_and(|i| i.fields().is_empty()) {
            innovation = None;
        }
        let size = snapshot.payload_bytes() + innovation.as_ref().map_or(0, |i| i.payload_bytes());
        if size > self.limit {
            report.reason = Some("snapshot exceeds the worker cache budget".into());
            return report;
        }
        let mut remove = vec![false; self.entries.len()];
        // A repeated path vertex updates one geometry, not the secant history.
        for (i, old) in self.entries.iter().enumerate() {
            if old.key == snapshot.key && old.parameters == snapshot.parameters {
                remove[i] = true;
            }
        }
        let matching: Vec<_> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(i, s)| !remove[*i] && s.key == snapshot.key)
            .map(|(i, _)| i)
            .collect();
        for &i in matching
            .iter()
            .take(matching.len().saturating_sub(self.history_per_fiber - 1))
        {
            remove[i] = true;
        }
        let mut retained = self.bytes
            - self
                .entries
                .iter()
                .enumerate()
                .filter(|(i, _)| remove[*i])
                .map(|(_, s)| s.payload_bytes())
                .sum::<usize>();
        for (i, old) in self.entries.iter().enumerate() {
            if retained + size <= self.limit {
                break;
            }
            if !remove[i] && !old.key.is_gamma() {
                remove[i] = true;
                retained -= old.payload_bytes();
            }
        }
        if snapshot.key.is_gamma() {
            for (i, old) in self.entries.iter().enumerate() {
                if retained + size <= self.limit {
                    break;
                }
                if !remove[i] {
                    remove[i] = true;
                    retained -= old.payload_bytes();
                }
            }
        }
        if retained + size > self.limit {
            report.reason = Some("rolling insertion would evict protected Gamma history".into());
            return report;
        }
        let mut index = 0;
        self.entries.retain(|old| {
            let discard = remove[index];
            index += 1;
            if discard {
                report.evicted.push(old.basis_generation);
            }
            !discard
        });
        self.entries.push_back(Entry {
            snapshot,
            innovation: innovation.map(Arc::new),
        });
        self.bytes = retained + size;
        report.stored = true;
        report.entries = self.len();
        report.payload_bytes = self.bytes;
        report.innovation_columns = self
            .entries
            .iter()
            .filter_map(|e| e.innovation.as_ref())
            .map(|i| i.fields().len())
            .sum();
        report.innovation_payload_bytes = self
            .entries
            .iter()
            .filter_map(|e| e.innovation.as_ref())
            .map(|i| i.payload_bytes())
            .sum();
        report
    }
}
