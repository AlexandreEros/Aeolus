"""Compatibility shim: the BVE physics moved to ``physics/barotropic.py``.

The class and dataclass are re-exported unchanged so every historical import
(``planetary_sandbox.run.bve.barotropic_vorticity``) keeps working. New code
should import from :mod:`planetary_sandbox.physics.barotropic`.
"""
from planetary_sandbox.physics.barotropic import (  # noqa: F401
    BarotropicState, BarotropicVorticity)
