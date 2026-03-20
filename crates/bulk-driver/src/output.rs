//! Output handling for bulk driver results.
//!
//! Supports two output modes:
//! - **Full mode**: One CSV file per job with complete band structure data
//! - **Selective mode**: Single merged CSV with specific k-points and bands
//!
//! Both modes include swept parameter values as columns.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use log::{debug, info};

use crate::config::{OutputConfig, OutputMode};
use crate::driver::JobResult;
use crate::expansion::ExpandedJob;

// ============================================================================
// Re-export (for public API if needed)
// ============================================================================

// ============================================================================
// Output Writer
// ============================================================================

/// Handles writing job results to CSV files.
pub struct OutputWriter {
    /// Output configuration
    config: OutputConfig,

    /// For full mode: output directory
    output_dir: PathBuf,

    /// For selective mode: accumulated rows
    selective_rows: Vec<SelectiveRow>,

    /// Parameter column names (determined from first job)
    param_columns: Vec<String>,

    /// K-point index to label mapping (for selective mode)
    k_labels: HashMap<usize, String>,

    /// Jobs reference for header generation
    first_job_seen: bool,
}

/// A row in selective mode output.
#[derive(Debug, Clone)]
struct SelectiveRow {
    /// Job index
    job_index: usize,

    /// Parameter values (matching param_columns order)
    param_values: Vec<String>,

    /// K-point index
    k_index: usize,

    /// K-point label (if any)
    k_label: Option<String>,

    /// K-point fractional coordinates
    k_frac: [f64; 2],

    /// K-point distance along path
    k_distance: f64,

    /// Band values (matching band indices from config)
    band_values: Vec<f64>,

    /// Band indices (1-based)
    band_indices: Vec<usize>,
}

impl OutputWriter {
    /// Create a new output writer.
    pub fn new(config: &OutputConfig, jobs: &[ExpandedJob]) -> Result<Self, OutputError> {
        // Create output directory for full mode
        let output_dir = config.directory.clone();
        if matches!(config.mode, OutputMode::Full) {
            fs::create_dir_all(&output_dir)?;
            info!("output directory: {}", output_dir.display());
        }

        // Build k-label mapping for selective mode
        let mut k_labels = HashMap::new();
        for (i, label) in config.selective.k_labels.iter().enumerate() {
            // Map labels to indices - this is a simplification
            // In practice, you'd want to look up actual k-point labels from the path
            k_labels.insert(i, label.clone());
        }

        // Determine parameter columns from first job
        let param_columns = if let Some(first) = jobs.first() {
            first
                .params
                .to_columns()
                .iter()
                .map(|(name, _)| name.to_string())
                .collect()
        } else {
            vec![]
        };

        Ok(Self {
            config: config.clone(),
            output_dir,
            selective_rows: Vec::new(),
            param_columns,
            k_labels,
            first_job_seen: false,
        })
    }

    /// Write a job result.
    pub fn write_result(
        &mut self,
        job: &ExpandedJob,
        result: &JobResult,
    ) -> Result<(), OutputError> {
        match (&self.config.mode, &result.result) {
            (OutputMode::Full, crate::driver::JobResultType::Maxwell(_)) => {
                self.write_full_maxwell(job, result)
            }
            (OutputMode::Full, crate::driver::JobResultType::EA(_)) => {
                self.write_full_ea(job, result)
            }
            (OutputMode::Selective, crate::driver::JobResultType::Maxwell(_)) => {
                self.accumulate_selective(job, result)
            }
            (OutputMode::Selective, crate::driver::JobResultType::EA(_)) => {
                // EA doesn't have k-points, so selective mode just writes eigenvalues
                self.write_full_ea(job, result)
            }
        }
    }

    /// Write full mode output for Maxwell (one file per job).
    fn write_full_maxwell(&mut self, job: &ExpandedJob, result: &JobResult) -> Result<(), OutputError> {
        let band_result = result.maxwell().expect("expected Maxwell result");

        let filename = format!(
            "{}_{:06}.csv",
            self.config.prefix,
            job.index
        );
        let path = self.output_dir.join(filename);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write header with parameters
        let param_cols = job.params.to_columns();
        write!(writer, "job_index")?;
        for (name, _) in &param_cols {
            write!(writer, ",{}", name)?;
        }
        write!(writer, ",k_index,kx,ky,k_distance")?;

        let max_bands = band_result.bands.iter().map(|b| b.len()).max().unwrap_or(0);
        for i in 1..=max_bands {
            write!(writer, ",band{}", i)?;
        }
        writeln!(writer)?;

        // Write data rows
        for (k_idx, ((kx, ky), bands)) in band_result
            .k_path
            .iter()
            .map(|k| (k[0], k[1]))
            .zip(band_result.bands.iter())
            .enumerate()
        {
            let distance = band_result.distances.get(k_idx).copied().unwrap_or(0.0);

            write!(writer, "{}", job.index)?;
            for (_, value) in &param_cols {
                write!(writer, ",{}", value)?;
            }
            write!(writer, ",{},{},{},{}", k_idx, kx, ky, distance)?;

            for omega in bands {
                let normalized = omega / (2.0 * std::f64::consts::PI);
                write!(writer, ",{}", normalized)?;
            }

            // Pad with empty columns
            for _ in bands.len()..max_bands {
                write!(writer, ",")?;
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        debug!("wrote {}", path.display());

        Ok(())
    }

    /// Write full mode output for EA (one file per job).
    fn write_full_ea(&mut self, job: &ExpandedJob, result: &JobResult) -> Result<(), OutputError> {
        let ea_result = result.ea().expect("expected EA result");

        // Write eigenvalues CSV
        let csv_filename = format!(
            "{}_{:06}_ea.csv",
            self.config.prefix,
            job.index
        );
        let csv_path = self.output_dir.join(csv_filename);

        let file = File::create(&csv_path)?;
        let mut writer = BufWriter::new(file);

        // Write header with parameters
        let param_cols = job.params.to_columns();
        write!(writer, "job_index")?;
        for (name, _) in &param_cols {
            write!(writer, ",{}", name)?;
        }
        write!(writer, ",n_iterations,converged,band_index,eigenvalue")?;
        writeln!(writer)?;

        // Write data rows (one per eigenvalue)
        for (band_idx, eigenvalue) in ea_result.eigenvalues.iter().enumerate() {
            write!(writer, "{}", job.index)?;
            for (_, value) in &param_cols {
                write!(writer, ",{}", value)?;
            }
            write!(
                writer,
                ",{},{},{},{}",
                ea_result.n_iterations,
                ea_result.converged,
                band_idx + 1,  // 1-based band index
                eigenvalue
            )?;
            writeln!(writer)?;
        }

        writer.flush()?;
        debug!("wrote {}", csv_path.display());

        // Write eigenvectors to binary file
        // Format: [n_bands: u64][nx: u64][ny: u64] followed by 
        // n_bands × (nx × ny × 2) f64 values (interleaved real/imag)
        let eigvec_filename = format!(
            "{}_{:06}_eigenvectors.bin",
            self.config.prefix,
            job.index
        );
        let eigvec_path = self.output_dir.join(eigvec_filename);

        let eigvec_file = File::create(&eigvec_path)?;
        let mut eigvec_writer = BufWriter::new(eigvec_file);

        let n_bands = ea_result.eigenvectors.len() as u64;
        let grid = ea_result.eigenvectors.first()
            .map(|f| f.grid())
            .unwrap_or_else(|| mpb2d_core::grid::Grid2D::new(1, 1, 1.0, 1.0));
        let nx = grid.nx as u64;
        let ny = grid.ny as u64;

        // Write header
        eigvec_writer.write_all(&n_bands.to_le_bytes())?;
        eigvec_writer.write_all(&nx.to_le_bytes())?;
        eigvec_writer.write_all(&ny.to_le_bytes())?;

        // Write eigenvector data (interleaved real/imag)
        for field in &ea_result.eigenvectors {
            for c in field.as_slice() {
                eigvec_writer.write_all(&c.re.to_le_bytes())?;
                eigvec_writer.write_all(&c.im.to_le_bytes())?;
            }
        }

        eigvec_writer.flush()?;
        debug!("wrote {}", eigvec_path.display());

        Ok(())
    }

    /// Accumulate data for selective mode.
    fn accumulate_selective(
        &mut self,
        job: &ExpandedJob,
        result: &JobResult,
    ) -> Result<(), OutputError> {
        let band_result = result.maxwell().expect("selective mode requires Maxwell result");

        let param_cols = job.params.to_columns();
        let param_values: Vec<String> = param_cols.iter().map(|(_, v)| v.clone()).collect();

        // Determine which k-points to include
        let k_indices: Vec<usize> = if self.config.selective.k_indices.is_empty() {
            // If no indices specified, use all
            (0..band_result.k_path.len()).collect()
        } else {
            self.config
                .selective
                .k_indices
                .iter()
                .filter(|&&i| i < band_result.k_path.len())
                .copied()
                .collect()
        };

        // Determine which bands to include
        let band_indices: Vec<usize> = self.config.selective.bands.clone();

        for &k_idx in &k_indices {
            let k_point = band_result.k_path.get(k_idx);
            let bands = band_result.bands.get(k_idx);
            let distance = band_result.distances.get(k_idx).copied().unwrap_or(0.0);

            if let (Some(k), Some(b)) = (k_point, bands) {
                let band_values: Vec<f64> = band_indices
                    .iter()
                    .filter_map(|&band_idx| {
                        if band_idx > 0 && band_idx <= b.len() {
                            Some(b[band_idx - 1] / (2.0 * std::f64::consts::PI))
                        } else {
                            None
                        }
                    })
                    .collect();

                self.selective_rows.push(SelectiveRow {
                    job_index: job.index,
                    param_values: param_values.clone(),
                    k_index: k_idx,
                    k_label: self.k_labels.get(&k_idx).cloned(),
                    k_frac: *k,
                    k_distance: distance,
                    band_values,
                    band_indices: band_indices.clone(),
                });
            }
        }

        if !self.first_job_seen {
            self.param_columns = param_cols.iter().map(|(n, _)| n.to_string()).collect();
            self.first_job_seen = true;
        }

        Ok(())
    }

    /// Finalize output (write selective mode file if applicable).
    pub fn finalize(&mut self) -> Result<(), OutputError> {
        if !matches!(self.config.mode, OutputMode::Selective) {
            return Ok(());
        }

        if self.selective_rows.is_empty() {
            return Ok(());
        }

        // Ensure output directory exists
        if let Some(parent) = self.config.filename.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let file = File::create(&self.config.filename)?;
        let mut writer = BufWriter::new(file);

        // Write header
        write!(writer, "job_index")?;
        for col in &self.param_columns {
            write!(writer, ",{}", col)?;
        }
        write!(writer, ",k_index,k_label,kx,ky,k_distance")?;

        // Determine band columns from first row
        if let Some(first) = self.selective_rows.first() {
            for &band_idx in &first.band_indices {
                write!(writer, ",band{}", band_idx)?;
            }
        }
        writeln!(writer)?;

        // Write data rows
        for row in &self.selective_rows {
            write!(writer, "{}", row.job_index)?;
            for value in &row.param_values {
                write!(writer, ",{}", value)?;
            }
            write!(
                writer,
                ",{},{},{},{},{}",
                row.k_index,
                row.k_label.as_deref().unwrap_or(""),
                row.k_frac[0],
                row.k_frac[1],
                row.k_distance
            )?;
            for value in &row.band_values {
                write!(writer, ",{}", value)?;
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        info!(
            "wrote selective output: {} rows to {}",
            self.selective_rows.len(),
            self.config.filename.display()
        );

        Ok(())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Output errors.
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV write error: {0}")]
    Csv(String),
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get standard k-point labels for a path type.
///
/// Returns a mapping from k-point index to label.
pub fn get_k_labels(path_type: &str, segments_per_leg: usize) -> HashMap<usize, String> {
    let labels = match path_type.to_lowercase().as_str() {
        "square" => vec!["Γ", "X", "M", "Γ"],
        "rectangular" => vec!["Γ", "X", "S", "Y", "Γ"],
        "triangular" | "hexagonal" => vec!["Γ", "M", "K", "Γ"],
        _ => vec![],
    };

    let mut result = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        let idx = i * segments_per_leg;
        result.insert(idx, label.to_string());
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_labels_square() {
        let labels = get_k_labels("square", 10);
        assert_eq!(labels.get(&0), Some(&"Γ".to_string()));
        assert_eq!(labels.get(&10), Some(&"X".to_string()));
        assert_eq!(labels.get(&20), Some(&"M".to_string()));
        assert_eq!(labels.get(&30), Some(&"Γ".to_string()));
    }
}
