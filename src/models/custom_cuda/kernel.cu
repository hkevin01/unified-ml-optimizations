/*
 * =============================================================================
 * FILE: src/models/custom_cuda/kernel.cu
 * PURPOSE: Custom CUDA C++ kernel for fused ReLU quantization + dequantization.
 *          Category: Systems-Level & Kernel Engineering (Requirement B).
 *
 * ID: CUDA-QDQ-001
 * Requirement: Implement INT8 per-channel fused quantize/ReLU/dequantize in a
 *              single CUDA kernel to minimize HBM round-trips.
 * Purpose: Demonstrate hand-written CUDA kernel for memory-layout optimization.
 * Inputs:
 *   - input:  float32 tensor of shape (N, C) - N samples, C channels.
 *   - scale:  float32 per-channel scale tensor of shape (C,).
 *   - output: float32 output tensor of shape (N, C); written in-place.
 *   - N: number of rows.
 *   - C: number of channels (columns).
 * Outputs: output[n,c] = dequant(clamp(round(ReLU(input[n,c]) / scale[c]), -127, 127))
 * Preconditions: input, scale, output allocated on CUDA device; N,C >= 1.
 * Postconditions: output contains float32 dequantized activations.
 * Assumptions: float32 arithmetic; per-channel scales > 0.
 * Side Effects: Writes to output buffer.
 * Failure Modes: C not multiple of BLOCK_SIZE -> padding handled by guard.
 * Error Handling: CUDA runtime errors propagate to PyTorch extension.
 * Constraints: Requires sm_60+ (Pascal or newer) for FP16 atomics if used.
 * Verification: Python test compares against reference CPU implementation.
 * References: CUDA Programming Guide; PTX ISA reference.
 * =============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Number of threads per block for the 1-D quantization kernel
#define BLOCK_SIZE 256

/*
 * ID: CUDA-QDQ-KERNEL-001
 * Requirement: Fused ReLU + INT8 quant + dequant in a single kernel.
 * Purpose: Avoid intermediate tensor writes between ReLU and quantization.
 * Inputs: See file header.
 * Outputs: See file header.
 * Preconditions: threadIdx.x + blockIdx.x*blockDim.x < N*C.
 * Postconditions: output[idx] set for all valid indices.
 * Assumptions: scale[c] > 0 for all c; no NaN in input.
 * Side Effects: Writes output.
 * Failure Modes: scale == 0 -> division by zero; caller must guard.
 * Error Handling: No-op for OOB threads (guard idx < N*C).
 * Constraints: None.
 * Verification: Unit test in binding.py.
 * References: None.
 */
__global__ void fused_relu_quant_dequant_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float*       __restrict__ output,
    int N,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int c = idx % C;

    // Step 1: ReLU
    float val = input[idx];
    val = val > 0.0f ? val : 0.0f;

    // Step 2: Quantize to INT8 range [-127, 127] using per-channel scale
    float s = scale[c];
    float q = rintf(val / s);
    q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);

    // Step 3: Dequantize back to float32
    output[idx] = q * s;
}

/*
 * ID: CUDA-QDQ-LAUNCH-001
 * Requirement: Python-callable launcher that checks tensor properties and
 *              dispatches the kernel with correct grid/block dimensions.
 * Purpose: Bridge between PyTorch tensor API and raw CUDA kernel.
 * Inputs:
 *   - input: torch::Tensor of shape (N, C), float32, CUDA.
 *   - scale: torch::Tensor of shape (C,), float32, CUDA.
 * Outputs: torch::Tensor of shape (N, C), float32, CUDA.
 * Preconditions: input.is_cuda(); scale.is_cuda(); input.dtype() == float32.
 * Postconditions: Returned tensor contains fused ReLU-quant-dequant values.
 * Assumptions: Tensors are contiguous.
 * Side Effects: Allocates output tensor; launches CUDA kernel.
 * Failure Modes: Non-contiguous input -> incorrect indexing.
 * Error Handling: .contiguous() called on inputs.
 * Constraints: None.
 * Verification: Compare output vs Python reference in binding.py tests.
 * References: PyTorch C++ extension API.
 */
torch::Tensor fused_relu_quant_dequant(
    torch::Tensor input,
    torch::Tensor scale
) {
    TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(),  "scale must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale must be float32");

    input = input.contiguous();
    scale = scale.contiguous();

    int N = input.size(0);
    int C = input.size(1);

    TORCH_CHECK(scale.numel() == C, "scale must have C elements");

    auto output = torch::empty_like(input);

    int total = N * C;
    int grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_relu_quant_dequant_kernel<<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C
    );

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_relu_quant_dequant",
        &fused_relu_quant_dequant,
        "Fused ReLU + INT8 quantize/dequantize (CUDA)"
    );
}
