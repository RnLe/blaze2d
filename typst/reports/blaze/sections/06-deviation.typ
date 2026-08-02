= Deviation Analysis <sec_deviation>

Across the full path, both polarizations agree with the MPB reference to within
Blaze's convergence tolerance (@fig_deviation). The relative deviation has a
median of about $1 times 10^(-4)$, with a worst case of $4 times 10^(-4)$ for TM
and $3 times 10^(-4)$ for TE. The agreement does not degrade with band index; it
is flat from the lowest band to the tenth.

The comparison between the `f32` and `f64` runs of Blaze is just as informative:
they differ by a median of about $1 times 10^(-7)$ and a maximum below
$5 times 10^(-5)$, more than an order of magnitude smaller than either run's
deviation from MPB. The eigenvalues from mixed precision and full precision are,
in effect, identical. That outcome is consistent with prior mixed-precision
LOBPCG results in electronic-structure calculations, where single-precision
storage with double-precision critical reductions preserved eigenvalue accuracy
while accelerating solves @woo2023. Here, the higher noise floor of
single-precision field storage remains below the convergence floor of the runs,
so the 3× speedup and the 95% memory reduction come at no measurable cost in
accuracy.
