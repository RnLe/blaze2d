//! Bounded multiple-geometry initialization on an unchanged physical fiber.
//! Previous geometries provide trial directions, never exact spectral reuse.
//! Source entries are freshly certified, and every prediction is rank-compressed
//! and Rayleigh-Ritz refined in the target mass metric before correction.
use super::{
    cache::SnapshotIdentity,
    parameter::ParameterMetric,
    reuse::{CachedSpectrum, SymmetryOptions},
    seed::{SeedReport, rank_aware_seed},
};
use crate::{
    backend::SpectralBackend,
    eigensolver::refine::certify_without_refinement,
    field::Field2D,
    operators::{LinearOperator, ThetaOperator},
};
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedHistory {
    Latest,
    Recent {
        snapshots: usize,
    },
    /// Requires target parameter coordinates through `prepare_tracked`.
    Nearest {
        snapshots: usize,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryEviction {
    Fifo,
    LeastRecentlyUsed,
    FarthestFromIncoming { metric: ParameterMetric },
}

pub struct ExactGeometryReuse {
    pub source_id: String,
    pub spectrum: CachedSpectrum,
}

struct GeometrySnapshot {
    id: String,
    parameters: Vec<f64>,
    identity: SnapshotIdentity,
    spectrum: CachedSpectrum,
    payload_bytes: usize,
    last_used: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InsertionReport {
    pub stored: bool,
    pub evicted_entries: usize,
    pub evicted_ids: Vec<String>,
    pub cache_entries: usize,
    pub cache_payload_bytes: usize,
    pub entry_payload_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContinuationReport {
    pub sources: Vec<String>,
    pub source_parameters: Vec<Vec<f64>>,
    pub requested_history: SeedHistory,
    pub cache_payload_bytes: usize,
    pub source_field_payload_bytes: usize,
    pub prepared_columns: usize,
    pub rank: SeedReport,
}

pub struct ContinuationSeed {
    pub fields: Vec<Field2D>,
    pub report: ContinuationReport,
}

/// Bounded geometry cache, FIFO by default. Capacity covers numerical/string
/// payloads, not allocator or policy bookkeeping. Seed compatibility compares
/// the physical fiber; exact reuse additionally compares every operator bit.
pub struct GeometryContinuationCache {
    entries: VecDeque<GeometrySnapshot>,
    payload_bytes: usize,
    limit: usize,
    eviction: GeometryEviction,
    clock: u64,
}

impl GeometryContinuationCache {
    pub fn new(payload_limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            payload_bytes: 0,
            limit: payload_limit,
            eviction: GeometryEviction::Fifo,
            clock: 0,
        }
    }
    pub fn with_eviction(payload_limit: usize, eviction: GeometryEviction) -> Self {
        Self {
            eviction,
            ..Self::new(payload_limit)
        }
    }
    pub fn resident_ids(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.id.as_str()).collect()
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Source certification is part of insertion, including its operator work.
    /// Duplicate IDs replace their old entry. Oversized entries do not evict a
    /// usable cache. The caller supplies parameter coordinates as provenance;
    /// numerical compatibility is established from the actual operator.
    pub fn insert<B: SpectralBackend>(
        &mut self,
        operator: &mut ThetaOperator<B>,
        id: String,
        parameters: Vec<f64>,
        mut spectrum: CachedSpectrum,
        options: &SymmetryOptions,
    ) -> Result<InsertionReport, String> {
        crate::profile_scope!("continuation.capture");
        if std::mem::size_of::<B::Real>() != 8 {
            return Err("geometry continuation currently requires f64".into());
        }
        if id.is_empty() || parameters.is_empty() || parameters.iter().any(|p| !p.is_finite()) {
            return Err("source ID and finite parameter coordinates are required".into());
        }
        if let GeometryEviction::FarthestFromIncoming { metric } = &self.eviction {
            metric.validate(&parameters)?;
            // Validate all distances before an insertion can evict any entry.
            for e in &self.entries {
                metric.distance(&parameters, &e.parameters)?;
            }
        }
        if spectrum.fields.is_empty()
            || spectrum.fields.len() != spectrum.eigenvalues.len()
            || spectrum.fields.iter().any(|f| {
                !same_grid(f.grid(), operator.grid())
                    || f.as_slice()
                        .iter()
                        .any(|z| !z.re.is_finite() || !z.im.is_finite())
            })
        {
            return Err("invalid source field dimensions or values".into());
        }
        let d = operator.dielectric();
        for axis in 0..2 {
            let cart =
                spectrum.k[0] * d.reciprocal_b1()[axis] + spectrum.k[1] * d.reciprocal_b2()[axis];
            if !cart.is_finite()
                || (cart - operator.bloch()[axis]).abs() > 8.0 * f64::EPSILON * cart.abs().max(1.0)
            {
                return Err("source k metadata differs from the actual operator".into());
            }
        }
        let certificate =
            certify_without_refinement(operator, &spectrum.eigenvalues, &spectrum.fields);
        spectrum.residuals = certificate.residuals;
        spectrum.absolute_residuals = certificate.absolute_residuals;
        spectrum.orthogonality_defect = certificate.b_orthogonality_defect;
        if !spectrum.passes(options) {
            return Err(format!(
                "source geometry snapshot failed fresh certification: residuals={:?}, orthogonality={:e}",
                spectrum.residuals, spectrum.orthogonality_defect
            ));
        }
        let identity = SnapshotIdentity::from_operator(operator);
        let payload_bytes = spectrum
            .bytes()
            .checked_add(identity.operator_bits.len() * 8)
            .and_then(|n| n.checked_add(id.len()))
            .and_then(|n| n.checked_add(parameters.len() * 8))
            .ok_or("geometry cache payload size overflows")?;
        let mut evicted_ids = Vec::new();
        let stored = payload_bytes <= self.limit;
        if stored {
            let stamp = self
                .clock
                .checked_add(1)
                .ok_or("geometry usage clock overflow")?;
            if let Some(position) = self.entries.iter().position(|e| e.id == id) {
                self.payload_bytes -= self.entries.remove(position).unwrap().payload_bytes;
            }
            while payload_bytes > self.limit - self.payload_bytes {
                let victim = match &self.eviction {
                    GeometryEviction::Fifo => 0,
                    GeometryEviction::LeastRecentlyUsed => {
                        self.entries
                            .iter()
                            .enumerate()
                            .min_by_key(|(i, e)| (e.last_used, *i))
                            .unwrap()
                            .0
                    }
                    GeometryEviction::FarthestFromIncoming { metric } => {
                        self.entries
                            .iter()
                            .enumerate()
                            .max_by(|(i, a), (j, b)| {
                                metric
                                    .distance(&parameters, &a.parameters)
                                    .unwrap()
                                    .total_cmp(
                                        &metric.distance(&parameters, &b.parameters).unwrap(),
                                    )
                                    .then(j.cmp(i))
                            })
                            .unwrap()
                            .0
                    }
                };
                let removed = self.entries.remove(victim).unwrap();
                self.payload_bytes -= removed.payload_bytes;
                evicted_ids.push(removed.id);
            }
            self.clock = stamp;
            self.payload_bytes += payload_bytes;
            self.entries.push_back(GeometrySnapshot {
                id,
                parameters,
                identity,
                spectrum,
                payload_bytes,
                last_used: stamp,
            });
        }
        Ok(InsertionReport {
            stored,
            evicted_entries: evicted_ids.len(),
            evicted_ids,
            cache_entries: self.entries.len(),
            cache_payload_bytes: self.payload_bytes,
            entry_payload_bytes: payload_bytes,
        })
    }

    /// Return at most `columns` lowest Ritz directions from the selected union.
    /// No extra outer-solver guards are added by enrichment. Rank and target
    /// residuals remain explicit and do not assert lowest-spectrum coverage.
    /// This read-only replay does not refresh LRU; scheduled jobs should use
    /// `prepare_tracked` to record their successful seed accesses.
    pub fn prepare<B: SpectralBackend>(
        &self,
        target: &mut ThetaOperator<B>,
        history: SeedHistory,
        columns: usize,
        relative_rank_tolerance: f64,
    ) -> Result<Option<ContinuationSeed>, String> {
        crate::profile_scope!("continuation.prepare");
        let snapshots = match history {
            SeedHistory::Latest => 1,
            SeedHistory::Recent { snapshots } => snapshots,
            SeedHistory::Nearest { .. } => {
                return Err("nearest seeds require target coordinates and prepare_tracked".into());
            }
        };
        if snapshots == 0 || columns == 0 {
            return Err("positive history and target column limits are required".into());
        }
        let identity = SnapshotIdentity::from_operator(target);
        if identity.storage_bits != 64 {
            return Err("geometry continuation currently requires f64".into());
        }
        let chosen: Vec<_> = self
            .entries
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, e)| same_fiber(&e.identity, &identity))
            .take(snapshots)
            .map(|(i, _)| i)
            .collect();
        self.prepare_indices(target, history, columns, relative_rank_tolerance, &chosen)
    }

    /// Select compatible geometry seeds and record successful uses for LRU.
    /// Parameter distance selects a seed only; it never authorizes exact reuse.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_tracked<B: SpectralBackend>(
        &mut self,
        target: &mut ThetaOperator<B>,
        parameters: &[f64],
        metric: &ParameterMetric,
        history: SeedHistory,
        columns: usize,
        relative_rank_tolerance: f64,
    ) -> Result<Option<ContinuationSeed>, String> {
        crate::profile_scope!("continuation.prepare");
        metric.validate(parameters)?;
        let snapshots = match history {
            SeedHistory::Latest => 1,
            SeedHistory::Recent { snapshots } | SeedHistory::Nearest { snapshots } => snapshots,
        };
        if snapshots == 0 || columns == 0 {
            return Err("positive history and target column limits are required".into());
        }
        let identity = SnapshotIdentity::from_operator(target);
        if identity.storage_bits != 64 {
            return Err("geometry continuation currently requires f64".into());
        }
        let mut chosen: Vec<_> = self
            .entries
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, e)| {
                same_fiber(&e.identity, &identity) && e.parameters.len() == parameters.len()
            })
            .map(|(i, e)| metric.distance(parameters, &e.parameters).map(|d| (i, d)))
            .collect::<Result<_, _>>()?;
        if matches!(history, SeedHistory::Nearest { .. }) {
            chosen.sort_by(|(a, da), (b, db)| da.total_cmp(db).then(b.cmp(a)));
        }
        let chosen: Vec<_> = chosen.into_iter().take(snapshots).map(|(i, _)| i).collect();
        let stamp = self
            .clock
            .checked_add(1)
            .ok_or("geometry usage clock overflow")?;
        let seed =
            self.prepare_indices(target, history, columns, relative_rank_tolerance, &chosen)?;
        if seed.is_some() {
            self.clock = stamp;
            for i in chosen {
                self.entries[i].last_used = stamp;
            }
        }
        Ok(seed)
    }

    /// Exact sampled-operator identity, k bits, requested width and a fresh
    /// certificate are all required. Different geometries are seed-only unless
    /// their actual sampled operators are identical. No parameter approximation
    /// or hash match can authorize skipping a solve.
    pub fn reuse_exact<B: SpectralBackend>(
        &mut self,
        target: &mut ThetaOperator<B>,
        k: [f64; 2],
        columns: usize,
        options: &SymmetryOptions,
    ) -> Result<Option<ExactGeometryReuse>, String> {
        crate::profile_scope!("continuation.reuse");
        if columns == 0 || k.iter().any(|x| !x.is_finite()) {
            return Err("invalid exact geometry request".into());
        }
        let identity = SnapshotIdentity::from_operator(target);
        if identity.storage_bits != 64 {
            return Err("geometry continuation currently requires f64".into());
        }
        let position = self.entries.iter().rposition(|e| {
            e.identity == identity
                && e.spectrum.k.map(f64::to_bits) == k.map(f64::to_bits)
                && e.spectrum.fields.len() == columns
        });
        let Some(i) = position else {
            return Ok(None);
        };
        let mut spectrum = self.entries[i].spectrum.clone();
        let cert = certify_without_refinement(target, &spectrum.eigenvalues, &spectrum.fields);
        spectrum.residuals = cert.residuals;
        spectrum.absolute_residuals = cert.absolute_residuals;
        spectrum.orthogonality_defect = cert.b_orthogonality_defect;
        if !spectrum.passes(options) {
            return Err("exact geometry cache failed fresh certification".into());
        }
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or("geometry usage clock overflow")?;
        self.entries[i].last_used = self.clock;
        Ok(Some(ExactGeometryReuse {
            source_id: self.entries[i].id.clone(),
            spectrum,
        }))
    }

    fn prepare_indices<B: SpectralBackend>(
        &self,
        target: &mut ThetaOperator<B>,
        history: SeedHistory,
        columns: usize,
        relative_rank_tolerance: f64,
        chosen: &[usize],
    ) -> Result<Option<ContinuationSeed>, String> {
        if chosen.is_empty() {
            return Ok(None);
        }
        let fields: Vec<_> = chosen
            .iter()
            .flat_map(|&i| self.entries[i].spectrum.fields.iter().cloned())
            .collect();
        let source_field_payload_bytes = fields.iter().map(|f| f.len() * 16).sum();
        let mut seed =
            rank_aware_seed(target, &fields, relative_rank_tolerance).map_err(|e| e.to_string())?;
        seed.fields.truncate(columns);
        Ok(Some(ContinuationSeed {
            report: ContinuationReport {
                sources: chosen.iter().map(|&i| self.entries[i].id.clone()).collect(),
                source_parameters: chosen
                    .iter()
                    .map(|&i| self.entries[i].parameters.clone())
                    .collect(),
                requested_history: history,
                cache_payload_bytes: self.payload_bytes,
                source_field_payload_bytes,
                prepared_columns: seed.fields.len(),
                rank: seed.report,
            },
            fields: seed.fields,
        }))
    }
}

fn same_fiber(a: &SnapshotIdentity, b: &SnapshotIdentity) -> bool {
    a.nx == b.nx
        && a.ny == b.ny
        && a.centered == b.centered
        && a.storage_bits == b.storage_bits
        && a.operator_bits.get(..9) == b.operator_bits.get(..9)
}

fn same_grid(a: crate::grid::Grid2D, b: crate::grid::Grid2D) -> bool {
    (a.nx, a.ny, a.centered, a.lx.to_bits(), a.ly.to_bits())
        == (b.nx, b.ny, b.centered, b.lx.to_bits(), b.ly.to_bits())
}
