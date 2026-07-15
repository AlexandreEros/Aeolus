"""Clear the CuPy kernel cache and verify kernel compilation.

``aeolus recompile`` is the canonical interface; ``psx-recompile`` is a
compatibility alias. Output is plain ASCII so legacy Windows code pages
(CP1252) can print it, and ``--help`` shows help instead of touching the
cache. CuPy is imported only when the command actually runs.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Only clear the on-disk kernel cache and CuPy memory pools; "
             "skip the kernel recompilation check.")


def build_parser(prog: str = "psx-recompile") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Clear CuPy's kernel cache and verify that the "
                    "spherical-harmonics kernel recompiles.")
    add_arguments(parser)
    return parser


def clear_cupy_cache(cp) -> None:
    """Clear CuPy's on-disk kernel cache and in-memory pools."""
    cache_dir = Path.home() / ".cupy" / "kernel_cache"

    if cache_dir.exists():
        print(f"Clearing CuPy cache at: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("[ok] Cache cleared")
    else:
        print(f"Cache directory not found: {cache_dir}")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("[ok] Memory pools cleared")


def verify_kernel_source(cp):
    """Recompile the sh_matrix kernel and sanity-check its output."""
    from planetary_sandbox.numerics.cuda.cuda_utils import raw_module_from_cuda

    print("\nRecompiling sh_matrix kernel...")
    module = raw_module_from_cuda("sh_matrix")
    kernel = module.get_function("generate_sph_harm_basis")
    print("[ok] Kernel compiled successfully")

    lat = cp.array([0.0, cp.pi / 4, cp.pi / 2])
    lon = cp.array([0.0, cp.pi / 2, cp.pi])
    Y = cp.zeros((3, 3), dtype=cp.complex128)
    kernel((1,), (3,), (lat, lon, Y, 3, 1))

    print("\nKernel output sample (first 3 points, l_max=1):")
    print("Shape:", Y.shape)
    imag_max = float(cp.abs(Y.imag).max())
    print("Imaginary max:", imag_max)
    if imag_max > 1e-10:
        print("[warn] Imaginary parts are nonzero; the compiled kernel may be stale.")
    else:
        print("[ok] Imaginary parts are zero (real kernel in use)")
    return Y


def run(args: argparse.Namespace) -> int:
    try:
        import cupy as cp
    except Exception as err:
        print(f"error: CuPy is unavailable ({err}); nothing to clear.")
        return 1

    clear_cupy_cache(cp)
    if not args.skip_verify:
        verify_kernel_source(cp)
    return 0


def main() -> int:
    """psx-recompile == aeolus recompile."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
