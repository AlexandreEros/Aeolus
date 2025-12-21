import importlib.resources as resources
import cupy as cp

def load_cuda_source(name: str) -> str:
    """
    Load a .cu file from the package 'planetary_sandbox.numerics.cuda'.

    Example:
        code = load_cuda_source("legendre")
    """
    package = "planetary_sandbox.numerics.cuda"
    with resources.files(package).joinpath(f"{name}.cu").open("r") as f:
        return f.read()

def raw_module_from_cuda(name: str, **kwargs) -> cp.RawModule:
    """
    Convenience wrapper: load <name>.cu and create a RawModule.
    """
    code = load_cuda_source(name)
    return cp.RawModule(code=code, **kwargs)

def get_kernel_from_cuda(name: str, kernel_name: str, **kwargs) -> cp.RawKernel:
    """
    Convenience wrapper: load <name>.cu, create a RawModule, and get a kernel.
    """
    module = raw_module_from_cuda(name, **kwargs)
    return module.get_kernel(kernel_name)