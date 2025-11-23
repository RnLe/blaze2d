//! Helpers for dumping operator inspection artifacts (snapshots + iteration traces).

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::bandstructure::InspectionOptions;
use crate::eigensolver::IterationDiagnostics;
use crate::field::Field2D;
use crate::operator::{OperatorSnapshotData, ThetaOperator};
use crate::polarization::Polarization;

pub(crate) fn dump_iteration_trace(
    opts: &InspectionOptions,
    pol: Polarization,
    k_index: usize,
    k_frac: [f64; 2],
    iterations: &[IterationDiagnostics],
) -> io::Result<Option<PathBuf>> {
    if !opts.operator.dump_iteration_traces || opts.output_dir.is_none() {
        return Ok(None);
    }
    let Some(dir) = opts.output_dir.as_ref() else {
        return Ok(None);
    };
    fs::create_dir_all(dir)?;
    let filename = format!(
        "operator_iteration_trace_{}_k{index:03}.csv",
        pol_label(pol),
        index = k_index
    );
    let path = dir.join(filename);
    let mut writer = BufWriter::new(File::create(&path)?);
    writeln!(
        writer,
        "k_index,kx,ky,iteration,max_residual,avg_residual,max_relative_residual,avg_relative_residual,block_size,new_directions,preconditioner_trials,preconditioner_avg_before,preconditioner_avg_after"
    )?;
    for info in iterations {
        writeln!(
            writer,
            "{k_index},{kx},{ky},{iter},{:.6e},{:.6e},{:.6e},{:.6e},{},{},{},{:.6e},{:.6e}",
            info.max_residual,
            info.avg_residual,
            info.max_relative_residual,
            info.avg_relative_residual,
            info.block_size,
            info.new_directions,
            info.preconditioner_trials,
            info.preconditioner_avg_before,
            info.preconditioner_avg_after,
            kx = k_frac[0],
            ky = k_frac[1],
            iter = info.iteration,
        )?;
    }
    writer.flush()?;
    Ok(Some(path))
}

pub(crate) fn dump_operator_snapshots<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    pol: Polarization,
    k_index: usize,
    k_frac: [f64; 2],
    modes: &[Field2D],
    opts: &InspectionOptions,
) -> io::Result<Vec<PathBuf>> {
    if !opts.operator.should_dump_snapshots(k_index) {
        return Ok(Vec::new());
    }
    let Some(dir) = opts.output_dir.as_ref() else {
        return Ok(Vec::new());
    };
    fs::create_dir_all(dir)?;
    let mut paths = Vec::new();
    let limit = opts.operator.snapshot_mode_limit.max(1);
    for (mode_idx, mode) in modes.iter().take(limit).enumerate() {
        let mut buffer = theta.alloc_field();
        buffer.as_mut_slice().copy_from_slice(mode.as_slice());
        let snapshot = theta.capture_snapshot(&buffer);
        let base = format!(
            "operator_snapshot_{}_k{index:03}_mode{mode:02}",
            pol_label(pol),
            index = k_index,
            mode = mode_idx + 1
        );
        let snapshot_path = dir.join(format!("{base}.csv"));
        write_snapshot_csv(&snapshot_path, k_frac, &snapshot)?;
        paths.push(snapshot_path.clone());
        let spectrum_path = dir.join(format!("{base}_spectrum.csv"));
        write_spectrum_csv(
            &spectrum_path,
            theta.kx_shifted(),
            theta.ky_shifted(),
            &snapshot,
        )?;
        paths.push(spectrum_path);
    }
    Ok(paths)
}

fn write_snapshot_csv(
    path: &Path,
    k_frac: [f64; 2],
    snapshot: &OperatorSnapshotData,
) -> io::Result<()> {
    let grid = snapshot.grid;
    let mut writer = BufWriter::new(File::create(path)?);
    let grad_present = snapshot.grad_x.is_some() && snapshot.grad_y.is_some();
    let eps_grad_present = snapshot.eps_grad_x.is_some() && snapshot.eps_grad_y.is_some();
    match (grad_present, eps_grad_present) {
        (true, true) => {
            writeln!(
                writer,
                "k_frac_x,k_frac_y,ix,iy,x,y,re_field,im_field,re_grad_x,im_grad_x,re_grad_y,im_grad_y,re_eps_grad_x,im_eps_grad_x,re_eps_grad_y,im_eps_grad_y,re_theta,im_theta"
            )?;
        }
        (true, false) => {
            writeln!(
                writer,
                "k_frac_x,k_frac_y,ix,iy,x,y,re_field,im_field,re_grad_x,im_grad_x,re_grad_y,im_grad_y,re_theta,im_theta"
            )?;
        }
        (false, _) => {
            writeln!(
                writer,
                "k_frac_x,k_frac_y,ix,iy,x,y,re_field,im_field,re_theta,im_theta"
            )?;
        }
    }
    let grad_x = snapshot.grad_x.as_ref();
    let grad_y = snapshot.grad_y.as_ref();
    let eps_grad_x = snapshot.eps_grad_x.as_ref();
    let eps_grad_y = snapshot.eps_grad_y.as_ref();
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let [x, y] = grid.cartesian_coords(ix, iy);
            let field = snapshot.field_spatial[idx];
            let theta_val = snapshot.theta_spatial[idx];
            match (grad_x, grad_y, eps_grad_x, eps_grad_y) {
                (Some(gx_vec), Some(gy_vec), Some(ex_vec), Some(ey_vec)) => {
                    let gx = gx_vec[idx];
                    let gy = gy_vec[idx];
                    let ex = ex_vec[idx];
                    let ey = ey_vec[idx];
                    writeln!(
                        writer,
                        "{},{},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                        k_frac[0],
                        k_frac[1],
                        ix,
                        iy,
                        x,
                        y,
                        field.re,
                        field.im,
                        gx.re,
                        gx.im,
                        gy.re,
                        gy.im,
                        ex.re,
                        ex.im,
                        ey.re,
                        ey.im,
                        theta_val.re,
                        theta_val.im,
                    )?;
                }
                (Some(gx_vec), Some(gy_vec), _, _) => {
                    let gx = gx_vec[idx];
                    let gy = gy_vec[idx];
                    writeln!(
                        writer,
                        "{},{},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                        k_frac[0],
                        k_frac[1],
                        ix,
                        iy,
                        x,
                        y,
                        field.re,
                        field.im,
                        gx.re,
                        gx.im,
                        gy.re,
                        gy.im,
                        theta_val.re,
                        theta_val.im,
                    )?;
                }
                _ => {
                    writeln!(
                        writer,
                        "{},{},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e}",
                        k_frac[0],
                        k_frac[1],
                        ix,
                        iy,
                        x,
                        y,
                        field.re,
                        field.im,
                        theta_val.re,
                        theta_val.im,
                    )?;
                }
            }
        }
    }
    writer.flush()
}

fn write_spectrum_csv(
    path: &Path,
    kx_shifted: &[f64],
    ky_shifted: &[f64],
    snapshot: &OperatorSnapshotData,
) -> io::Result<()> {
    let grid = snapshot.grid;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "ix,iy,kx_plus_g,ky_plus_g,field_hat_mag,theta_hat_mag"
    )?;
    for iy in 0..grid.ny {
        let ky = ky_shifted.get(iy).copied().unwrap_or(0.0);
        for ix in 0..grid.nx {
            let kx = kx_shifted.get(ix).copied().unwrap_or(0.0);
            let idx = grid.idx(ix, iy);
            let field_mag = snapshot.field_fourier[idx].norm();
            let theta_mag = snapshot.theta_fourier[idx].norm();
            writeln!(
                writer,
                "{ix},{iy},{kx},{ky},{:.6e},{:.6e}",
                field_mag, theta_mag
            )?;
        }
    }
    writer.flush()
}

fn pol_label(pol: Polarization) -> &'static str {
    match pol {
        Polarization::TE => "te",
        Polarization::TM => "tm",
    }
}
