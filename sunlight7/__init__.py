"""Sunlight-7 reproducibility package."""
import os
# Limit numerical-library thread pools for deterministic, lightweight execution.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
__version__ = "1.0.0"
