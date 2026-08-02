= Conclusion <sec_conclusion>

Blaze was built to solve 2D photonic crystal band structures quickly,
repeatedly, and without surprises. The benchmarks underline this behavior. In
mixed precision Blaze is roughly 3× faster than MPB on a single core, scales
almost linearly across threads where MPB's threaded mode stalls, and uses up to
20× less memory at the resolutions used in practice. It does so while
reproducing MPB's eigenvalues to within $tilde 10^(-4)$ (its own convergence
tolerance) with single and double precision giving indistinguishable results.
Its cost is also predictable: solve time grows smoothly with resolution and
stays flat across dielectric contrast, in both of which MPB is uneven.

None of this is accidental. It follows from starting on a modern foundation: a
memory-aware, mixed-precision LOBPCG core written in Rust, parallelized across
whole solves rather than inside them, and shipped as an installable Python
package. Performance, scalability, and simplicity were design goals from the
start.

The most consequential advantage does not appear in any timing plot. Because
Blaze is a plane-wave solver that exposes its internals, it returns far more
than frequencies: the Bloch functions themselves, and the operator matrix
elements assembled from them, such as Berry connections and effective-mass
tensors @berry1984 @wilczek1984 @lowdin1951. Standard band solvers like MPB do
not surface this data. This allows the user to look inside the operator instead
of only at its spectrum, and to do so for thousands of crystal configurations in
a single sweep. For routine band-structure work the speed is the headline. For
the physics that follows, the ability to extract this interior structure at
scale is what turns a fast solver into a foundation to build on.
