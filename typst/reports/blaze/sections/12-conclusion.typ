= Conclusion <sec_conclusion>

The recorded experiments show lower Blaze wall-clock times and peak process memory for the tested configurations. The ratios depend on the stated numerical settings. A matched release comparison requires explicit hardware, solver revisions, retained outputs, and residual checks.

Blaze schedules independent configurations and uses mixed-precision storage with f64 critical reductions. It also exports projected operator and derivative data, including velocity matrices, effective masses, and formulation-specific geometric terms @berry1984 @wilczek1984 @lowdin1951.

MPB exposes fields, eigenvectors, and group velocities. Blaze focuses on integrating parameter studies and projected extraction in one configuration and result contract.
