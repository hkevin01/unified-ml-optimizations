# =============================================================================
# FILE: src/models/custom_cuda/binding.py
# PURPOSE: Build and load the custom CUDA extension; provide Python interface.
# =============================================================================

"""
ID: BIND-001
Requirement: Compile kernel.cu via torch.utils.cpp_extension.load() on import
             and expose fused_relu_quant_dequant() as a Python callable.
Purpose: Allow the CUDA kernel to be used from Python without a separate build step.
Inputs: None (compilation triggered on module import if not cached).
Outputs: Module-level 'cuda_ext' variable (the loaded extension) and
         'fused_relu_quant_dequant' function.
Preconditions: CUDA toolkit installed; nvcc on PATH; PyTorch with CUDA support.
Postconditions: cuda_ext is a loaded C++ extension module if CUDA available;
                None otherwise.
Assumptions: kernel.cu located in same directory as this file.
Side Effects: Compiles .cu -> .so on first import; cached in torch's build dir.
Failure Modes: Missing nvcc -> ImportError; non-CUDA PyTorch -> RuntimeError.
Error Handling: Catches exceptions and sets cuda_ext = None for graceful fallback.
Constraints: Compilation may take 30-60 seconds on first run.
Verification: Call fused_relu_quant_dequant on a small CUDA tensor and
              compare vs reference (torch.relu + per-channel quant).
References: PyTorch JIT compilation docs.
"""

import os
import torch
from typing import Optional

cuda_ext = None

def _try_load_extension():
    """
    ID: BIND-LOAD-001
    Requirement: Attempt JIT compilation of kernel.cu into a Python extension.
    Purpose: Deferred loading avoids import-time failure on CPU-only machines.
    Inputs: None.
    Outputs: Loaded extension module or None.
    Preconditions: CUDA available; nvcc on PATH.
    Postconditions: cuda_ext set at module level.
    Assumptions: kernel.cu is in the same directory as this file.
    Side Effects: Compiles CUDA code; writes .so to torch build cache.
    Failure Modes: nvcc not found; CUDA toolkit missing.
    Error Handling: Catches all exceptions; returns None on failure.
    Constraints: None.
    Verification: Check cuda_ext is not None on a CUDA machine.
    References: torch.utils.cpp_extension.load documentation.
    """
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ext = load(
            name="custom_cuda_ops",
            sources=[os.path.join(kernel_dir, "kernel.cu")],
            verbose=False,
        )
        return ext
    except Exception as exc:
        print(f"[custom_cuda] Failed to load CUDA extension: {exc}")
        return None


cuda_ext = _try_load_extension()


def fused_relu_quant_dequant(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    ID: BIND-FN-001
    Requirement: Apply fused ReLU + INT8 quant + dequant via CUDA kernel or
                 a Python fallback when the kernel is unavailable.
    Purpose: Unified Python interface regardless of CUDA availability.
    Inputs:
      - x: (N, C) float32 CUDA tensor.
      - scale: (C,) float32 CUDA per-channel scale tensor (values > 0).
    Outputs: (N, C) float32 tensor with fused operation applied.
    Preconditions: x.shape[1] == scale.shape[0]; scale values > 0.
    Postconditions: Output semantically equals:
                    q = clamp(round(relu(x) / scale), -127, 127)
                    return q * scale
    Assumptions: Scales computed from abs-max of channel; guaranteed > 0.
    Side Effects: None.
    Failure Modes: scale == 0 -> division by zero in kernel (undefined behavior).
    Error Handling: Clamp scale to 1e-8 minimum before calling.
    Constraints: None.
    Verification: MSE < 0.01 vs Python reference for uniform random input.
    References: None.
    """
    # Guard scale against zero
    scale = scale.clamp(min=1e-8)

    if cuda_ext is not None and x.is_cuda:
        return cuda_ext.fused_relu_quant_dequant(x.float(), scale.float())

    # Python fallback: functionally identical, no kernel launch
    val = torch.relu(x.float())
    q = (val / scale).round().clamp(-127, 127)
    return (q * scale).to(x.dtype)
