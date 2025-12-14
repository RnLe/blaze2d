use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{backend::SpectralBackend, field::Field2D, grid::Grid2D};
use num_complex::Complex64;

fn seeded_field(grid: Grid2D) -> Field2D {
    let mut field = Field2D::zeros(grid);
    let nx = grid.nx;
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        let ix = idx % nx;
        let iy = idx / nx;
        let real = ((ix as f64 + 1.0) * 0.31).sin();
        let imag = ((iy as f64 + 1.0) * 0.47).cos();
        *value = Complex64::new(real, imag);
    }
    field
}

fn bench_cpu_fft(c: &mut Criterion) {
    let configs = vec![
        ("serial", CpuBackend::new()),
        (
            "parallel",
            CpuBackend::new_parallel().with_parallel_threshold(1),
        ),
    ];
    let sizes = [16usize, 24, 32, 48, 64, 96, 128, 192, 256];
    let mut group = c.benchmark_group("cpu_fft_forward_inverse");
    group.sample_size(10);
    for &size in &sizes {
        let grid = Grid2D::new(size, size, 1.0, 1.0);
        let template = seeded_field(grid);
        for (label, backend) in &configs {
            let template_for_run = template.clone();
            group.throughput(Throughput::Elements((grid.len() * 2) as u64));
            group.bench_function(BenchmarkId::new(*label, size), move |b| {
                b.iter(|| {
                    let mut field = template_for_run.clone();
                    backend.forward_fft_2d(&mut field);
                    backend.inverse_fft_2d(&mut field);
                    black_box(field);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(fft_benches, bench_cpu_fft);
criterion_main!(fft_benches);
