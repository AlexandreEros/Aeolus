#!/usr/bin/env python3
"""
Clear CuPy cache and verify kernel compilation.
"""
import shutil
from pathlib import Path
import cupy as cp

def clear_cupy_cache():
    """Clear CuPy's kernel cache."""
    cache_dir = Path.home() / '.cupy' / 'kernel_cache'
    
    if cache_dir.exists():
        print(f"Clearing CuPy cache at: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("✓ Cache cleared")
    else:
        print(f"Cache directory not found: {cache_dir}")
    
    # Also clear any in-memory cache
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("✓ Memory pools cleared")


def verify_kernel_source():
    """Check what kernel source is being used."""
    from planetary_sandbox.numerics.cuda.cuda_utils import raw_module_from_cuda
    
    # This will recompile with cleared cache
    print("\nRecompiling sh_matrix kernel...")
    module = raw_module_from_cuda("sh_matrix")
    kernel = module.get_function('generate_sph_harm_basis')
    print("✓ Kernel compiled successfully")
    
    # Test that it produces the right output
    lat = cp.array([0.0, cp.pi/4, cp.pi/2])
    lon = cp.array([0.0, cp.pi/2, cp.pi])
    Y = cp.zeros((3, 3), dtype=cp.complex128)
    
    kernel((1,), (3,), (lat, lon, Y, 3, 1))
    
    print("\nKernel output sample (first 3 points, l_max=1):")
    print("Shape:", Y.shape)
    print("Imaginary max:", cp.abs(Y.imag).max())
    
    if cp.abs(Y.imag).max() > 1e-10:
        print("⚠ WARNING: Imaginary parts are large - still using COMPLEX kernel")
        print("Check that sh_matrix.cu was actually replaced!")
    else:
        print("✓ Imaginary parts are zero - using REAL kernel")
    
    return Y

def main():
    print("="*70)
    print("CuPy Cache Management")
    print("="*70)
    
    clear_cupy_cache()
    Y = verify_kernel_source()
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. If imaginary parts are still large, check:")
    print("   - Did you actually replace sh_matrix.cu?")
    print("   - Run: cp sh_matrix_fixed.cu planetary_sandbox/numerics/cuda/sh_matrix.cu")
    print("2. Run test_orthogonality.py again")
    print("="*70)

if __name__ == "__main__":
    main()