"""Post-run validation workflows.

Validation code is deliberately separate from the numerical cores.  In
particular, importing this package must not initialize CUDA or construct a
spherical-harmonic backend.
"""

