"""Barotropic vorticity equation run package.

Re-exports are lazy (PEP 562) so that configuration-only imports
(``planetary_sandbox.run.bve.config``, ``planetary_sandbox.run.bve.io``)
do not pull in CuPy. ``from planetary_sandbox.run.bve import run_bve``
keeps working exactly as before.
"""
import importlib

_LAZY_EXPORTS = {
    "BarotropicVorticity": ".barotropic_vorticity",
    "BarotropicState": ".barotropic_vorticity",
    "run_bve": ".runner",
    "rk4_step": ".runner",
}

# Wildcard exports match the historical eager __init__ so
# ``from planetary_sandbox.run.bve import *`` keeps its previous surface.
__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
