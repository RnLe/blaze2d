# Performance measurements

Use a release wheel and the shared calculation examples:

```bash
python run.py ../examples/calculations/square-rods.toml --repeats 10 --warmups 2 --output bands.json
python run.py ../examples/calculations/operator-point.toml --repeats 10 --warmups 2 --output operators.json
```

The report records all resolved numerical settings, retained outputs, source revision,
backend, processor description, individual timings, median time, and peak process memory.
Warmups occur in the same process. Run measurements sequentially on an idle machine.
Use the same Python and NumPy environment when comparing two solver builds.

Peak memory includes the interpreter and libraries. It is a process high-water mark,
not a measurement of solver workspace alone. Match retained arrays, band windows,
thread counts, and precision when comparing memory or solve time.

The pre-0.7 benchmark scripts and data-processing tools are in `archives/benchmarks`.
They document the historical website datasets and require the historical solver
interface. Do not use their old TOML dialect with release 0.7. Existing recorded data
is retained; release comparisons require fresh measurements and convergence checks.
