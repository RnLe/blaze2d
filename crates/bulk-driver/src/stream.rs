//! Real-time streaming for Python/WASM consumers.
//!
//! This module provides streaming output channels for real-time data consumption.
//! Use cases include:
//!
//! - **Python plotting**: Stream band diagrams to matplotlib as they're computed
//! - **WASM/React**: Push updates to web UI components
//! - **Inter-process communication**: Pipe results to other processes
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────┐     ┌────────────────┐     ┌──────────────────┐
//! │  Solver Pool   │────▶│ StreamChannel  │────▶│  Subscribers     │
//! │  (N threads)   │     │  (broadcast)   │     │  - Callback      │
//! └────────────────┘     └────────────────┘     │  - Channel       │
//!                                               │  - Custom        │
//!                                               └──────────────────┘
//! ```
//!
//! ## Filtered Streaming
//!
//! For selective mode, use `FilteredStreamChannel` to apply k-point/band
//! filtering at the emission point, reducing bandwidth:
//!
//! ```text
//! ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
//! │  Solver Pool   │────▶│ SelectiveFilter│────▶│ StreamChannel  │
//! │  (full data)   │     │ (k/band filter)│     │ (filtered)     │
//! └────────────────┘     └────────────────┘     └────────────────┘
//! ```
//!
//! ## Subscriber Types
//!
//! - **CallbackSubscriber**: Invokes a closure for each result
//! - **ChannelSubscriber**: Sends results to a crossbeam channel
//! - **Custom implementations**: Implement `Subscriber` trait

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use log::{debug, warn};
use parking_lot::RwLock;

use crate::channel::{
    BackpressurePolicy, ChannelError, ChannelStats, CompactBandResult, OutputChannelSink,
    StreamConfig,
};
use crate::config::SelectiveSpec;

// ============================================================================
// Subscriber Trait
// ============================================================================

/// Trait for subscribers that receive streaming band structure results.
///
/// Implement this trait to create custom consumers for real-time data.
///
/// # Thread Safety
///
/// Subscribers must be `Send + Sync` as they may be called from multiple
/// solver threads concurrently.
///
/// # Example
///
/// ```ignore
/// struct PrintSubscriber;
///
/// impl Subscriber for PrintSubscriber {
///     fn on_result(&self, result: &CompactBandResult) {
///         println!("Job {} complete: {} k-points", 
///             result.job_index, result.num_k_points());
///     }
/// }
/// ```
pub trait Subscriber: Send + Sync {
    /// Called when a new result is available.
    ///
    /// This method should be non-blocking or very fast. For expensive
    /// processing, use a channel-based subscriber and process in a
    /// separate thread.
    fn on_result(&self, result: &CompactBandResult);

    /// Called when all jobs are complete.
    fn on_complete(&self, stats: &ChannelStats) {
        let _ = stats; // Default: ignore
    }

    /// Called when an error occurs.
    fn on_error(&self, error: &ChannelError) {
        let _ = error; // Default: ignore
    }
}

// ============================================================================
// Callback Subscriber
// ============================================================================

/// Subscriber that invokes a callback closure for each result.
///
/// Useful for Python/FFI bindings where a function pointer is passed in.
pub struct CallbackSubscriber<F>
where
    F: Fn(&CompactBandResult) + Send + Sync,
{
    callback: F,
}

impl<F> CallbackSubscriber<F>
where
    F: Fn(&CompactBandResult) + Send + Sync,
{
    /// Create a new callback subscriber.
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F> Subscriber for CallbackSubscriber<F>
where
    F: Fn(&CompactBandResult) + Send + Sync,
{
    fn on_result(&self, result: &CompactBandResult) {
        (self.callback)(result);
    }
}

// ============================================================================
// Channel Subscriber
// ============================================================================

/// Subscriber that sends results to a crossbeam channel.
///
/// This allows asynchronous consumption of results in a separate thread.
/// The consumer receives owned clones of results.
pub struct ChannelSubscriber {
    sender: Sender<CompactBandResult>,
    backpressure: BackpressurePolicy,
    dropped: AtomicUsize,
}

impl ChannelSubscriber {
    /// Create a new channel subscriber with the specified capacity and policy.
    pub fn new(capacity: usize, backpressure: BackpressurePolicy) -> (Self, Receiver<CompactBandResult>) {
        let (sender, receiver) = bounded(capacity);
        (
            Self {
                sender,
                backpressure,
                dropped: AtomicUsize::new(0),
            },
            receiver,
        )
    }

    /// Get the number of dropped results (due to backpressure).
    pub fn dropped_count(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Subscriber for ChannelSubscriber {
    fn on_result(&self, result: &CompactBandResult) {
        let result = result.clone();

        match self.backpressure {
            BackpressurePolicy::Block => {
                // Block until space is available
                if self.sender.send(result).is_err() {
                    warn!("channel subscriber: receiver dropped");
                }
            }
            BackpressurePolicy::DropNewest => {
                // Try to send, drop if full
                if let Err(TrySendError::Full(_)) = self.sender.try_send(result) {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
            BackpressurePolicy::DropOldest => {
                // If full, drain one and send
                loop {
                    match self.sender.try_send(result.clone()) {
                        Ok(()) => break,
                        Err(TrySendError::Full(_)) => {
                            // Try to receive (drop oldest)
                            let _ = self.sender.try_send(result.clone());
                            self.dropped.fetch_add(1, Ordering::Relaxed);
                            break;
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            warn!("channel subscriber: receiver dropped");
                            break;
                        }
                    }
                }
            }
        }
    }

    fn on_complete(&self, stats: &ChannelStats) {
        debug!(
            "channel subscriber complete: {} results, {} dropped",
            stats.results_sent,
            self.dropped.load(Ordering::Relaxed)
        );
    }
}

// ============================================================================
// Collecting Subscriber
// ============================================================================

/// Subscriber that collects all results into a vector.
///
/// Useful for testing and small workloads where you want all results in memory.
pub struct CollectingSubscriber {
    results: RwLock<Vec<CompactBandResult>>,
}

impl CollectingSubscriber {
    /// Create a new collecting subscriber.
    pub fn new() -> Self {
        Self {
            results: RwLock::new(Vec::new()),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            results: RwLock::new(Vec::with_capacity(capacity)),
        }
    }

    /// Get the collected results.
    ///
    /// Note: This clones the results. For large datasets, consider using
    /// a channel-based subscriber instead.
    pub fn results(&self) -> Vec<CompactBandResult> {
        self.results.read().clone()
    }

    /// Take the collected results, leaving an empty vector.
    pub fn take_results(&self) -> Vec<CompactBandResult> {
        std::mem::take(&mut *self.results.write())
    }

    /// Get the number of collected results.
    pub fn len(&self) -> usize {
        self.results.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.results.read().is_empty()
    }
}

impl Default for CollectingSubscriber {
    fn default() -> Self {
        Self::new()
    }
}

impl Subscriber for CollectingSubscriber {
    fn on_result(&self, result: &CompactBandResult) {
        self.results.write().push(result.clone());
    }
}

// ============================================================================
// Selective Filter
// ============================================================================

/// High-performance filter for selective streaming.
///
/// Pre-computes index sets for O(1) lookup during filtering.
/// This is applied at the emission point to minimize data transfer.
#[derive(Debug, Clone)]
pub struct SelectiveFilter {
    /// K-point indices to include (sorted, deduplicated)
    k_indices: Vec<usize>,

    /// Band indices to include (0-based internally, sorted, deduplicated)
    band_indices: Vec<usize>,

    /// Whether filtering is actually needed
    is_active: bool,
}

impl SelectiveFilter {
    /// Create a filter from a SelectiveSpec.
    ///
    /// Note: `k_labels` are not resolved here - they should be converted
    /// to indices by the caller using the k-path information.
    pub fn from_spec(spec: &SelectiveSpec) -> Self {
        let mut k_indices: Vec<usize> = spec.k_indices.clone();
        k_indices.sort_unstable();
        k_indices.dedup();

        // Convert 1-based band indices to 0-based
        let mut band_indices: Vec<usize> = spec
            .bands
            .iter()
            .filter(|&&b| b > 0)
            .map(|&b| b - 1)
            .collect();
        band_indices.sort_unstable();
        band_indices.dedup();

        let is_active = !k_indices.is_empty() || !band_indices.is_empty();

        Self {
            k_indices,
            band_indices,
            is_active,
        }
    }

    /// Create a filter with explicit k-point and band indices.
    ///
    /// Band indices are 0-based here (unlike SelectiveSpec which is 1-based).
    pub fn new(k_indices: Vec<usize>, band_indices: Vec<usize>) -> Self {
        let mut k_indices = k_indices;
        k_indices.sort_unstable();
        k_indices.dedup();

        let mut band_indices = band_indices;
        band_indices.sort_unstable();
        band_indices.dedup();

        let is_active = !k_indices.is_empty() || !band_indices.is_empty();

        Self {
            k_indices,
            band_indices,
            is_active,
        }
    }

    /// Create a pass-through filter (no filtering).
    pub fn none() -> Self {
        Self {
            k_indices: Vec::new(),
            band_indices: Vec::new(),
            is_active: false,
        }
    }

    /// Check if this filter is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Apply the filter to a result, returning a filtered copy.
    ///
    /// If the filter is not active, returns a clone of the original.
    /// This is optimized for the common case where filtering is applied.
    /// 
    /// Note: Filtering only applies to Maxwell results. EA results are returned unchanged.
    pub fn apply(&self, result: &CompactBandResult) -> CompactBandResult {
        use crate::channel::{CompactResultType, MaxwellResult};

        if !self.is_active {
            return result.clone();
        }

        // Filtering only applies to Maxwell results
        let maxwell = match &result.result_type {
            CompactResultType::Maxwell(m) => m,
            CompactResultType::EA(_) => return result.clone(),
        };

        // Determine which k-indices to include
        let k_filter: Vec<usize> = if self.k_indices.is_empty() {
            // No k-filter: include all
            (0..maxwell.k_path.len()).collect()
        } else {
            // Filter to valid indices
            self.k_indices
                .iter()
                .copied()
                .filter(|&i| i < maxwell.k_path.len())
                .collect()
        };

        // Determine which band indices to include
        let num_bands = maxwell.bands.first().map(|b| b.len()).unwrap_or(0);
        let band_filter: Vec<usize> = if self.band_indices.is_empty() {
            // No band filter: include all
            (0..num_bands).collect()
        } else {
            // Filter to valid indices
            self.band_indices
                .iter()
                .copied()
                .filter(|&i| i < num_bands)
                .collect()
        };

        // Build filtered result
        let k_path: Vec<[f64; 2]> = k_filter.iter().map(|&i| maxwell.k_path[i]).collect();
        let distances: Vec<f64> = k_filter.iter().map(|&i| maxwell.distances[i]).collect();
        let bands: Vec<Vec<f64>> = k_filter
            .iter()
            .map(|&k_idx| {
                band_filter
                    .iter()
                    .map(|&b_idx| maxwell.bands[k_idx][b_idx])
                    .collect()
            })
            .collect();

        CompactBandResult {
            job_index: result.job_index,
            params: result.params.clone(),
            result_type: CompactResultType::Maxwell(MaxwellResult {
                k_path,
                distances,
                bands,
            }),
        }
    }

    /// Get the number of k-points that will be in filtered output.
    pub fn k_count(&self) -> Option<usize> {
        if self.k_indices.is_empty() {
            None // All k-points
        } else {
            Some(self.k_indices.len())
        }
    }

    /// Get the number of bands that will be in filtered output.
    pub fn band_count(&self) -> Option<usize> {
        if self.band_indices.is_empty() {
            None // All bands
        } else {
            Some(self.band_indices.len())
        }
    }
}

impl Default for SelectiveFilter {
    fn default() -> Self {
        Self::none()
    }
}

// ============================================================================
// Filtered Stream Channel
// ============================================================================

/// Stream channel with server-side selective filtering.
///
/// Applies k-point and band filtering at the emission point, before
/// broadcasting to subscribers. This reduces bandwidth and memory usage
/// when only a subset of data is needed.
///
/// ## Example
///
/// ```ignore
/// // Stream only Gamma, X, M points and first 4 bands
/// let filter = SelectiveFilter::new(
///     vec![0, 10, 15],  // k-indices for Gamma, X, M
///     vec![0, 1, 2, 3], // bands 1-4 (0-based)
/// );
/// let channel = FilteredStreamChannel::new(filter, StreamConfig::default());
/// ```
pub struct FilteredStreamChannel {
    /// The underlying stream channel
    inner: StreamChannel,

    /// Filter to apply
    filter: SelectiveFilter,

    /// Original results received (before filtering)
    results_received: AtomicUsize,
}

impl FilteredStreamChannel {
    /// Create a new filtered stream channel.
    pub fn new(filter: SelectiveFilter, config: StreamConfig) -> Self {
        Self {
            inner: StreamChannel::new(config),
            filter,
            results_received: AtomicUsize::new(0),
        }
    }

    /// Create from a SelectiveSpec.
    pub fn from_spec(spec: &SelectiveSpec, config: StreamConfig) -> Self {
        Self::new(SelectiveFilter::from_spec(spec), config)
    }

    /// Subscribe to receive filtered results.
    pub fn subscribe(&self, subscriber: Arc<dyn Subscriber>) {
        self.inner.subscribe(subscriber);
    }

    /// Add a callback subscriber.
    pub fn on_result<F>(&self, callback: F)
    where
        F: Fn(&CompactBandResult) + Send + Sync + 'static,
    {
        self.inner.on_result(callback);
    }

    /// Add a channel subscriber and return the receiver.
    pub fn add_channel_subscriber(&self) -> Receiver<CompactBandResult> {
        self.inner.add_channel_subscriber()
    }

    /// Add a collecting subscriber and return it.
    pub fn add_collecting_subscriber(&self) -> Arc<CollectingSubscriber> {
        self.inner.add_collecting_subscriber()
    }

    /// Get the number of subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.inner.subscriber_count()
    }

    /// Get the filter being applied.
    pub fn filter(&self) -> &SelectiveFilter {
        &self.filter
    }

    /// Get the number of results received (before filtering).
    pub fn results_received(&self) -> usize {
        self.results_received.load(Ordering::Relaxed)
    }
}

impl OutputChannelSink for FilteredStreamChannel {
    fn send(&self, result: CompactBandResult) -> Result<(), ChannelError> {
        self.results_received.fetch_add(1, Ordering::Relaxed);

        // Apply filter and forward to inner channel
        let filtered = self.filter.apply(&result);
        self.inner.send(filtered)
    }

    fn flush(&self) -> Result<(), ChannelError> {
        self.inner.flush()
    }

    fn close(&self) -> Result<ChannelStats, ChannelError> {
        self.inner.close()
    }

    fn is_open(&self) -> bool {
        self.inner.is_open()
    }
}

// ============================================================================
// Stream Channel
// ============================================================================

/// Streaming output channel with multiple subscriber support.
///
/// Results are broadcast to all registered subscribers as they arrive.
/// Subscribers can be added before starting the computation.
pub struct StreamChannel {
    /// Registered subscribers
    subscribers: RwLock<Vec<Arc<dyn Subscriber>>>,

    /// Configuration
    config: StreamConfig,

    /// Statistics
    results_sent: AtomicUsize,
    results_dropped: AtomicUsize,

    /// Channel is closed
    closed: AtomicBool,
}

impl StreamChannel {
    /// Create a new stream channel.
    pub fn new(config: StreamConfig) -> Self {
        Self {
            subscribers: RwLock::new(Vec::new()),
            config,
            results_sent: AtomicUsize::new(0),
            results_dropped: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(StreamConfig::default())
    }

    /// Subscribe to receive results.
    ///
    /// The subscriber will receive all results emitted after subscription.
    pub fn subscribe(&self, subscriber: Arc<dyn Subscriber>) {
        self.subscribers.write().push(subscriber);
    }

    /// Add a callback subscriber.
    pub fn on_result<F>(&self, callback: F)
    where
        F: Fn(&CompactBandResult) + Send + Sync + 'static,
    {
        self.subscribe(Arc::new(CallbackSubscriber::new(callback)));
    }

    /// Add a channel subscriber and return the receiver.
    ///
    /// This is the recommended way to consume results asynchronously.
    pub fn add_channel_subscriber(&self) -> Receiver<CompactBandResult> {
        let (subscriber, receiver) = ChannelSubscriber::new(
            self.config.channel_capacity,
            self.config.backpressure,
        );
        self.subscribe(Arc::new(subscriber));
        receiver
    }

    /// Add a collecting subscriber and return it.
    pub fn add_collecting_subscriber(&self) -> Arc<CollectingSubscriber> {
        let subscriber = Arc::new(CollectingSubscriber::new());
        self.subscribe(subscriber.clone());
        subscriber
    }

    /// Get the number of subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.read().len()
    }
}

impl OutputChannelSink for StreamChannel {
    fn send(&self, result: CompactBandResult) -> Result<(), ChannelError> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(ChannelError::Closed);
        }

        // Broadcast to all subscribers
        let subscribers = self.subscribers.read();
        for subscriber in subscribers.iter() {
            subscriber.on_result(&result);
        }

        self.results_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn flush(&self) -> Result<(), ChannelError> {
        // Streaming mode has no buffering to flush
        Ok(())
    }

    fn close(&self) -> Result<ChannelStats, ChannelError> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Err(ChannelError::Closed);
        }

        let stats = ChannelStats {
            results_sent: self.results_sent.load(Ordering::Relaxed),
            results_dropped: self.results_dropped.load(Ordering::Relaxed),
            ..Default::default()
        };

        // Notify subscribers of completion
        let subscribers = self.subscribers.read();
        for subscriber in subscribers.iter() {
            subscriber.on_complete(&stats);
        }

        Ok(stats)
    }

    fn is_open(&self) -> bool {
        !self.closed.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Arc Wrapper for Shared Use
// ============================================================================

/// Thread-safe shared stream channel.
///
/// This is a convenience type for cases where the channel needs to be
/// shared across threads (e.g., passed to both driver and consumer).
pub type SharedStreamChannel = Arc<StreamChannel>;

/// Create a shared stream channel.
pub fn shared_stream(config: StreamConfig) -> SharedStreamChannel {
    Arc::new(StreamChannel::new(config))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expansion::{AtomParams, JobParams};
    use mpb2d_core::polarization::Polarization;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Duration;

    fn make_test_result(index: usize) -> CompactBandResult {
        use crate::channel::{CompactResultType, MaxwellResult};

        CompactBandResult {
            job_index: index,
            params: JobParams {
                eps_bg: 12.0,
                resolution: 32,
                polarization: Polarization::TM,
                lattice_type: Some("square".to_string()),
                atoms: vec![AtomParams {
                    index: 0,
                    pos: [0.5, 0.5],
                    radius: 0.3,
                    eps_inside: 1.0,
                }],
                sweep_values: vec![],
            },
            result_type: CompactResultType::Maxwell(MaxwellResult {
                k_path: vec![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
                distances: vec![0.0, 0.5, 1.0],
                bands: vec![
                    vec![0.1, 0.2, 0.3],
                    vec![0.15, 0.25, 0.35],
                    vec![0.2, 0.3, 0.4],
                ],
            }),
        }
    }

    #[test]
    fn test_callback_subscriber() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let subscriber = CallbackSubscriber::new(move |_result| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        });

        subscriber.on_result(&make_test_result(0));
        subscriber.on_result(&make_test_result(1));
        subscriber.on_result(&make_test_result(2));

        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_channel_subscriber() {
        let (subscriber, receiver) = ChannelSubscriber::new(10, BackpressurePolicy::Block);

        // Send some results
        subscriber.on_result(&make_test_result(0));
        subscriber.on_result(&make_test_result(1));

        // Receive them
        let r0 = receiver.recv().unwrap();
        let r1 = receiver.recv().unwrap();

        assert_eq!(r0.job_index, 0);
        assert_eq!(r1.job_index, 1);
    }

    #[test]
    fn test_channel_subscriber_backpressure() {
        let (subscriber, _receiver) = ChannelSubscriber::new(2, BackpressurePolicy::DropNewest);

        // Overflow the channel
        for i in 0..10 {
            subscriber.on_result(&make_test_result(i));
        }

        // Should have dropped some
        assert!(subscriber.dropped_count() > 0);
    }

    #[test]
    fn test_collecting_subscriber() {
        let subscriber = CollectingSubscriber::new();

        subscriber.on_result(&make_test_result(0));
        subscriber.on_result(&make_test_result(1));

        assert_eq!(subscriber.len(), 2);

        let results = subscriber.take_results();
        assert_eq!(results.len(), 2);
        assert!(subscriber.is_empty());
    }

    #[test]
    fn test_stream_channel() {
        let channel = StreamChannel::new(StreamConfig::default());

        // Add multiple subscribers
        let collector = channel.add_collecting_subscriber();
        let receiver = channel.add_channel_subscriber();

        assert_eq!(channel.subscriber_count(), 2);

        // Send results
        for i in 0..5 {
            channel.send(make_test_result(i)).unwrap();
        }

        // Close
        let stats = channel.close().unwrap();
        assert_eq!(stats.results_sent, 5);

        // Check collector
        assert_eq!(collector.len(), 5);

        // Check channel
        let mut received = 0;
        while receiver.try_recv().is_ok() {
            received += 1;
        }
        assert_eq!(received, 5);
    }

    #[test]
    fn test_stream_channel_closed() {
        let channel = StreamChannel::new(StreamConfig::default());
        channel.close().unwrap();

        let result = channel.send(make_test_result(0));
        assert!(matches!(result, Err(ChannelError::Closed)));
    }

    #[test]
    fn test_stream_channel_concurrent() {
        let channel = Arc::new(StreamChannel::new(StreamConfig::default()));
        let receiver = channel.add_channel_subscriber();

        let channel_clone = channel.clone();
        let sender_thread = thread::spawn(move || {
            for i in 0..100 {
                channel_clone.send(make_test_result(i)).unwrap();
            }
            channel_clone.close().unwrap()
        });

        let receiver_thread = thread::spawn(move || {
            let mut count = 0;
            while let Ok(_) = receiver.recv_timeout(Duration::from_millis(100)) {
                count += 1;
            }
            count
        });

        let stats = sender_thread.join().unwrap();
        let count = receiver_thread.join().unwrap();

        assert_eq!(stats.results_sent, 100);
        assert_eq!(count, 100);
    }

    // ========================================================================
    // SelectiveFilter Tests
    // ========================================================================

    #[test]
    fn test_selective_filter_none() {
        let filter = SelectiveFilter::none();
        assert!(!filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        // Should be identical
        assert_eq!(filtered.k_path().unwrap().len(), result.k_path().unwrap().len());
        assert_eq!(filtered.bands().unwrap().len(), result.bands().unwrap().len());
    }

    #[test]
    fn test_selective_filter_k_only() {
        // Filter to only k-index 0 and 2
        let filter = SelectiveFilter::new(vec![0, 2], vec![]);
        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        assert_eq!(filtered.k_path().unwrap().len(), 2);
        assert_eq!(filtered.k_path().unwrap()[0], result.k_path().unwrap()[0]);
        assert_eq!(filtered.k_path().unwrap()[1], result.k_path().unwrap()[2]);
        assert_eq!(filtered.bands().unwrap().len(), 2);
        // All bands preserved
        assert_eq!(filtered.bands().unwrap()[0].len(), result.bands().unwrap()[0].len());
    }

    #[test]
    fn test_selective_filter_bands_only() {
        // Filter to only bands 0 and 2 (0-based)
        let filter = SelectiveFilter::new(vec![], vec![0, 2]);
        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        // All k-points preserved
        assert_eq!(filtered.k_path().unwrap().len(), result.k_path().unwrap().len());
        // Only 2 bands
        assert_eq!(filtered.bands().unwrap()[0].len(), 2);
        assert_eq!(filtered.bands().unwrap()[0][0], result.bands().unwrap()[0][0]);
        assert_eq!(filtered.bands().unwrap()[0][1], result.bands().unwrap()[0][2]);
    }

    #[test]
    fn test_selective_filter_both() {
        // Filter to k-index 1 and bands 1,2 (0-based)
        let filter = SelectiveFilter::new(vec![1], vec![1, 2]);
        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        assert_eq!(filtered.k_path().unwrap().len(), 1);
        assert_eq!(filtered.k_path().unwrap()[0], result.k_path().unwrap()[1]);
        assert_eq!(filtered.bands().unwrap().len(), 1);
        assert_eq!(filtered.bands().unwrap()[0].len(), 2);
        assert_eq!(filtered.bands().unwrap()[0][0], result.bands().unwrap()[1][1]);
        assert_eq!(filtered.bands().unwrap()[0][1], result.bands().unwrap()[1][2]);
    }

    #[test]
    fn test_selective_filter_from_spec() {
        use crate::config::SelectiveSpec;

        let spec = SelectiveSpec {
            k_indices: vec![0, 2],
            k_labels: vec![], // Not resolved here
            bands: vec![1, 3], // 1-based
        };
        let filter = SelectiveFilter::from_spec(&spec);

        assert!(filter.is_active());

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        assert_eq!(filtered.k_path().unwrap().len(), 2);
        assert_eq!(filtered.bands().unwrap()[0].len(), 2);
        // Band 1 -> index 0, Band 3 -> index 2
        assert_eq!(filtered.bands().unwrap()[0][0], result.bands().unwrap()[0][0]);
        assert_eq!(filtered.bands().unwrap()[0][1], result.bands().unwrap()[0][2]);
    }

    #[test]
    fn test_selective_filter_out_of_bounds() {
        // Request indices that don't exist
        let filter = SelectiveFilter::new(vec![0, 100], vec![0, 100]);

        let result = make_test_result(0);
        let filtered = filter.apply(&result);

        // Should only include valid indices
        assert_eq!(filtered.k_path().unwrap().len(), 1); // Only index 0 is valid
        assert_eq!(filtered.bands().unwrap()[0].len(), 1); // Only band 0 is valid
    }

    // ========================================================================
    // FilteredStreamChannel Tests
    // ========================================================================

    #[test]
    fn test_filtered_stream_channel() {
        let filter = SelectiveFilter::new(vec![0, 2], vec![0, 1]);
        let channel = FilteredStreamChannel::new(filter, StreamConfig::default());

        let collector = channel.add_collecting_subscriber();

        // Send a full result
        channel.send(make_test_result(0)).unwrap();

        // Should receive filtered version
        let results = collector.results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].k_path().unwrap().len(), 2);
        assert_eq!(results[0].bands().unwrap()[0].len(), 2);

        // Check received count
        assert_eq!(channel.results_received(), 1);
    }

    #[test]
    fn test_filtered_stream_channel_multiple() {
        let filter = SelectiveFilter::new(vec![1], vec![]);
        let channel = FilteredStreamChannel::new(filter, StreamConfig::default());

        let receiver = channel.add_channel_subscriber();

        for i in 0..5 {
            channel.send(make_test_result(i)).unwrap();
        }
        channel.close().unwrap();

        let mut count = 0;
        while let Ok(result) = receiver.try_recv() {
            assert_eq!(result.k_path().unwrap().len(), 1);
            count += 1;
        }
        assert_eq!(count, 5);
    }
}
