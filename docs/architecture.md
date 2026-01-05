# Architecture

## Overview

Blaze2D solves the 2D Maxwell eigenvalue problem in the frequency domain using the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) algorithm.

## Mathematical Formulation

For a 2D photonic crystal with periodic dielectric function ε(**r**), the master equation for TM polarization is:

$$\nabla \times \left( \frac{1}{\varepsilon(\mathbf{r})} \nabla \times \mathbf{H} \right) = \frac{\omega^2}{c^2} \mathbf{H}$$

In 2D with out-of-plane fields, this reduces to a scalar eigenvalue problem discretized on a real-space grid.

## Algorithm: LOBPCG

Blaze2D uses LOBPCG rather than the Davidson or conjugate-gradient approaches used in MPB.
LOBPCG offers:

- Block computation of multiple eigenvalues simultaneously
- Efficient for the lowest eigenvalues of large sparse systems
- Amenable to preconditioning

### Iteration Structure

```
1. Initialize random block of trial vectors X
2. Apply operator: W = A·X
3. Rayleigh-Ritz projection onto {X, W, P}
4. Update X, P from optimal linear combination
5. Check convergence; repeat if needed
```

## Key Distinction: Job-Level Parallelism & Subspace deflation

### MPB's Approach
MPB parallelizes **within** each k-point solve:
- FFTs are distributed across threads (FFTW + OpenMP)
- Thread synchronization at each FFT call
- Overhead significant for small grids

Additionally, MPB typically relies on the convergence of the entire subspace trace. This prevents early locking of individual eigenpairs that may converge faster than the rest of the block.

### Blaze2D's Approach
Blaze2D parallelizes **across** k-points:
- Each thread handles independent k-point calculations
- No synchronization between jobs
- Linear scaling with job count

This architectural choice makes Blaze2D particularly efficient for:
- Large parameter sweeps
- Band diagram calculations (many k-points)
- Moderate grid resolutions (≤128×128)

Furthermore, Blaze2D employs **active subspace deflation**. Once an eigenpair reaches the target tolerance, it is locked, and the search subspace is orthogonalized against it. This focuses computational effort solely on the remaining unconverged bands, significantly accelerating the final iterations.

## Operator Implementation

The Maxwell operator Θ = ∇ × (1/ε) ∇ × is applied as:

1. **Curl in Fourier space**: Multiply by ik
2. **Inverse epsilon in real space**: Pointwise division
3. **Curl in Fourier space**: Multiply by ik

FFTs convert between real and Fourier representations.
For TM mode, this simplifies to a scalar Laplacian-like operator weighted by 1/ε.

## Memory Layout

Field data uses row-major ordering with dimensions `(nx, ny)`.
Complex arrays use interleaved real/imaginary format for cache efficiency.

## Future Optimizations

### Dynamic Block Size Reduction
Currently, deflation locks converged eigenvectors but maintains the full block size for the Rayleigh-Ritz projection.
**Potential optimization:** Implement dynamic block size reduction, explicitly shrinking the dimensions of the trial vectors ($X$), residuals ($W$), and search directions ($P$) as eigenpairs converge. This would reduce the dense matrix operations by removing up to $3 \times N_{locked}$ columns per iteration.

### GPU Acceleration
The operator application (FFT + pointwise multiply) maps naturally to GPU:
- cuFFT for transforms
- Batched k-point processing
- Expected 10-100× speedup for grids ≥64×64
