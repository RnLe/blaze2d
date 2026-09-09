= Deviation Analysis <sec_deviation>

The recorded median relative frequency difference from MPB is about $1 times 10^(-4)$. Maxima are approximately $4 times 10^(-4)$ for TM and $3 times 10^(-4)$ for TE. These are differences between numerical runs, not certified discretization-error bounds.

The `f32` and `f64` Blaze runs differ by a median of about $1 times 10^(-7)$
and a maximum below $5 times 10^(-5)$. Their agreement in these fixtures is
consistent with mixed-precision LOBPCG studies using double-precision critical
reductions @woo2023. It does not establish equal accuracy for other spectra,
operator derivatives, or discretizations. Timing and memory ratios come from
separate experiments and must be interpreted with their respective settings.
