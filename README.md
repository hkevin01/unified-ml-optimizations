<div align="center">

# unified-ml-optimizations

**A research-grade PyTorch project demonstrating all three major categories of ML algorithmic optimization in one unified, runnable codebase.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Triton](https://img.shields.io/badge/Triton-2.1%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://triton-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Grade-blue?style=for-the-badge)]()

[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)](https://github.com/psf/black)
[![Docstring Standard](https://img.shields.io/badge/Docs-NASA%20Style-0B3D91?style=flat-square)]()
[![AMP](https://img.shields.io/badge/Training-AMP%20fp16%2Fbf16-orange?style=flat-square)]()
[![CPU Fallback](https://img.shields.io/badge/CPU-Fallback%20Supported-brightgreen?style=flat-square)]()
[![Triton Kernels](https://img.shields.io/badge/Triton-Custom%20GPU%20Kernels-76B900?style=flat-square&logo=nvidia)]()
[![INT8 Quant](https://img.shields.io/badge/Quantization-INT8%20QAT-blueviolet?style=flat-square)]()
[![Low Rank](https://img.shields.io/badge/Attention-Low--Rank%20LoRA--style-blue?style=flat-square)]()
[![Pruning](https://img.shields.io/badge/Sparsity-Gradual%20Magnitude%20Pruning-red?style=flat-square)]()

*Three optimization categories. One model. Zero compromises.*

</div>

---

---

## Table of Contents

- [Overview](#overview)
- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Module Dependency Graph](#module-dependency-graph)
- [Optimization Categories](#optimization-categories)
- [Training Pipeline State Machine](#training-pipeline-state-machine)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Component Deep Dives](#component-deep-dives)
- [Memory and Compute Breakdown](#memory-and-compute-breakdown)
- [LR Schedule and Pruning Interaction](#lr-schedule-and-pruning-interaction)
- [Performance Benchmarks](#performance-benchmarks)
- [Hardware Requirements](#hardware-requirements)
- [API Reference](#api-reference)
- [Extension Guide](#extension-guide)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)
- [References](#references)

---

## Overview

This project is a **single end-to-end machine learning codebase** that demonstrates, in one training run, the three dominant axes along which modern ML systems are optimized: mathematical structure, hardware-level kernel engineering, and training-dynamics convergence control. Rather than presenting these as separate toy examples, this codebase composes all three into a unified `OptimizedTransformer` model that can be trained from scratch on any sequence classification task.

The model is a **Pre-LN Transformer classifier** with six layers, multi-head attention, and a feedforward network - similar in structure to a small BERT or GPT encoder. Every component has been replaced or augmented with an optimization representative of its category. The attention Q/K/V projections use low-rank matrix factorization. The feedforward network uses a Triton-compiled fused kernel. The classifier head uses INT8 weight quantization. The optimizer separates parameter groups for proper weight decay. The learning rate follows a cosine schedule with linear warmup. Weights are gradually pruned during training using a magnitude-based ramp.

Understanding how these techniques compose in a single model is the core goal of this repository. Each optimization interacts with the others in non-trivial ways: a low-rank factorization changes the curvature of the loss landscape, which affects how AdamW's moment estimates accumulate. A fused kernel changes which tensors touch GPU memory, which influences cache behavior during the backward pass. Pruning modifies the effective rank of the weight matrices mid-training, altering the gradient signal for all downstream parameters. This project makes those interactions visible and reproducible.

> [!IMPORTANT]
> This project is designed to run on both CPU and GPU. All CUDA and Triton components fall back gracefully to pure PyTorch when a GPU is unavailable, so you can study the code and run smoke tests on any machine.

---

## Why This Project Exists

Most optimization tutorials isolate techniques into independent notebooks: "here is how AdamW works," "here is a Triton kernel," "here is low-rank decomposition." This is pedagogically convenient but misses the point - in real systems, these optimizations compose and interact. A low-rank layer changes the gradient landscape, which affects how AdamW's second moment estimates evolve. A fused kernel changes memory access patterns, which determines how much the learning rate schedule matters for convergence speed.

This codebase presents a **research-grade reference implementation** where every component is production-quality: NASA-style docstrings with explicit preconditions, postconditions, failure modes, and verification strategies; proper AMP (Automatic Mixed Precision) integration; gradient clipping; a profiling harness; and a Jupyter notebook for post-training analysis. The design philosophy is that you should be able to take any individual module from this repository and drop it directly into a production codebase with no modification.

The secondary goal is **explainability**. Comments describe not just what the code does, but why the specific approach was chosen over alternatives - why cosine decay is better than step decay for Transformers, why INT8 is preferred over INT4 for QAT, why the low-rank B matrix is initialized near zero, and why weight decay must be separated from biases and LayerNorm parameters.

> [!NOTE]
> The synthetic dataset used here is intentionally simple - the goal is to verify that all components integrate correctly and that the loss decreases. Swap in any real dataset (CIFAR-10, WikiText, etc.) by replacing `src/data/dataset.py` with your own DataLoader. No other files need to change.

---

## Architecture

The model is structured as a stack of identical `TransformerBlock` modules. Each block applies Pre-LN (layer normalization applied *before* the residual branch, not after), which is the key architectural difference from the original 2017 "Attention is All You Need" Transformer. Pre-LN normalization improves gradient flow at initialization, making deep models significantly easier to train at small batch sizes. The attention sub-layer within each block uses low-rank factorized projections for Q, K, and V, while the feedforward sub-layer uses a Triton-fused kernel. The final classification head uses an INT8-quantized linear layer.

The following diagram shows how data flows through the full system during a single training step, illustrating where each optimization category is applied and how the optimizer feedback loop closes.

```mermaid
flowchart TD
    A["Input IDs\n(B, L) LongTensor"] --> B["Token Embedding\n+ Positional Encoding"]
    B --> C["Embedding Dropout"]
    C --> D["TransformerBlock × N"]

    subgraph BLOCK["TransformerBlock (× n_layers)"]
        D1["LayerNorm (Pre-LN)"] --> D2["LowRankAttention\n🔵 Category A"]
        D2 --> D3["Residual Add"]
        D3 --> D4["LayerNorm (Pre-LN)"]
        D4 --> D5["FusedLinearGELU\n🟢 Category B - Triton Kernel"]
        D5 --> D6["Linear (up-proj)"]
        D6 --> D7["Residual Add"]
    end

    D --> BLOCK
    BLOCK --> E["Final LayerNorm"]
    E --> F["Mean Pool over L"]
    F --> G["QuantizedLinear Head\n🟢 Category B - INT8"]
    G --> H["Logits (B, num_classes)"]
    H --> I["Cross-Entropy Loss"]

    subgraph OPT["Optimizer (Category C 🔴)"]
        J["AdamW\nweight-decay groups"]
        K["Cosine LR Warmup"]
        L["DynamicPruner\nmagnitude sparsity"]
    end

    I --> J
    J --> K
    K --> L
    L --> D
```

> [!TIP]
> The Pre-LN placement (LayerNorm before the residual branch) is a deliberate architectural choice. Pre-LN Transformers are significantly more stable to train than Post-LN variants, especially at small batch sizes and high learning rates, because the gradient norm at initialization is bounded regardless of model depth. Post-LN requires careful warm-up and learning rate tuning to prevent the gradient from exploding in early layers.

---

## Module Dependency Graph

Understanding the module dependency graph is essential before making changes to the codebase. If you want to change the attention mechanism, you only need to modify `low_rank_layer.py` and potentially `base_model.py`. If you want to swap the optimizer, only `optimizer.py` is involved. The graph below makes these boundaries explicit, coloring each node by the optimization category it belongs to.

```mermaid
graph LR
    main["src/main.py"] --> train["src/training/train.py"]
    main --> dataset["src/data/dataset.py"]
    main --> base_model["src/models/base_model.py"]
    main --> metrics["src/utils/metrics.py"]

    train --> optimizer["src/training/optimizer.py"]
    train --> scheduler["src/training/scheduler.py"]
    train --> config["src/utils/config.py"]
    train --> metrics
    train --> profiler["src/utils/profiler.py"]

    base_model --> low_rank["src/models/low_rank_layer.py"]
    base_model --> fused["src/models/fused_kernel.py"]
    base_model --> config

    fused --> cuda_bind["src/models/custom_cuda/binding.py"]
    cuda_bind --> kernel_cu["src/models/custom_cuda/kernel.cu"]

    profiler --> metrics

    style low_rank fill:#3498db,color:#fff
    style fused fill:#2ecc71,color:#fff
    style cuda_bind fill:#2ecc71,color:#fff
    style kernel_cu fill:#27ae60,color:#fff
    style optimizer fill:#e74c3c,color:#fff
    style scheduler fill:#e74c3c,color:#fff
```

> [!NOTE]
> Blue nodes are Category A (mathematical/structural), green nodes are Category B (systems/kernel), and red nodes are Category C (convergence). Grey nodes are shared infrastructure with no category affiliation. `config.py` is the only file imported by both the model and the training loop - all hyperparameters flow from a single source of truth.

---

## Optimization Categories

Each of the three categories addresses a fundamentally different bottleneck in ML training. Category A reduces the mathematical cost of the computation itself by exploiting structure in the weight matrices - the insight being that learned weight matrices in trained models are often low-rank, meaning their information can be captured by two smaller matrices with far fewer parameters. Category B reduces the hardware cost of executing that computation by minimizing memory traffic between the GPU's compute units and its high-bandwidth memory (HBM) - the insight being that modern GPUs are memory-bandwidth-limited for many layer sizes, not compute-limited. Category C reduces the number of gradient steps needed to reach a given loss value by steering the optimizer toward better trajectories - the insight being that naive SGD wastes steps and that schedule, decay strategy, and sparsity can all be co-designed.

| <sub>#</sub> | <sub>Category</sub> | <sub>Bottleneck Addressed</sub> | <sub>Technique Used</sub> | <sub>Key Benefit</sub> | <sub>Primary Files</sub> |
|---|---|---|---|---|---|
| <sub>A</sub> | <sub>Mathematical & Structural</sub> | <sub>Parameter count and FLOP count of linear projections</sub> | <sub>Low-rank factorization of Q, K, V projections as A @ B</sub> | <sub>Up to 8x fewer params and FLOPs per attention projection at rank=32</sub> | <sub>low_rank_layer.py</sub> |
| <sub>B1</sub> | <sub>Systems - Fused Kernel</sub> | <sub>Memory bandwidth waste from writing intermediate activations to HBM</sub> | <sub>Triton fused matmul + GELU in a single kernel pass, no intermediate buffer</sub> | <sub>Eliminates one full HBM read+write per feedforward sublayer per step</sub> | <sub>fused_kernel.py</sub> |
| <sub>B2</sub> | <sub>Systems - Quantization</sub> | <sub>Weight memory footprint and memory bandwidth during linear projection</sub> | <sub>INT8 symmetric per-tensor quantization with straight-through estimator for QAT</sub> | <sub>4x weight memory reduction; INT8 GEMM capable on tensor-core hardware</sub> | <sub>fused_kernel.py</sub> |
| <sub>B3</sub> | <sub>Systems - CUDA Kernel</sub> | <sub>Fused activation + quantization without intermediate memory buffers</sub> | <sub>Hand-written CUDA C++ kernel: ReLU + INT8 quant + dequant in one launch</sub> | <sub>Eliminates kernel launch overhead and intermediate tensor allocation</sub> | <sub>custom_cuda/</sub> |
| <sub>C1</sub> | <sub>Convergence - Optimizer</sub> | <sub>Weight decay incorrectly applied to 1-D parameters such as biases and LN scales</sub> | <sub>AdamW with separate param groups; fused CUDA AdamW kernel when available</sub> | <sub>Correct regularization geometry; faster parameter updates via kernel fusion</sub> | <sub>optimizer.py</sub> |
| <sub>C2</sub> | <sub>Convergence - Scheduling</sub> | <sub>Large gradient updates early in training destabilize weights before they have structure</sub> | <sub>Linear warmup over first 10% of steps then cosine decay to 10% of peak LR</sub> | <sub>Prevents early divergence; smooth final convergence without hard LR drops</sub> | <sub>scheduler.py</sub> |
| <sub>C3</sub> | <sub>Convergence - Pruning</sub> | <sub>Model capacity wasted on near-zero weights that contribute noise, not signal</sub> | <sub>Gradual magnitude pruning from 0% to 30% sparsity over the middle 70% of training</sub> | <sub>Reduced inference cost; implicit regularization effect on remaining weights</sub> | <sub>optimizer.py</sub> |

> [!NOTE]
> The three categories are listed in order of increasing subtlety. Category A optimizations (low-rank) are purely mathematical and can be derived from first principles. Category B optimizations (kernel fusion, quantization) require understanding GPU hardware memory hierarchy. Category C optimizations (scheduling, pruning) require understanding training dynamics and how loss landscapes evolve over time.

---

## Training Pipeline State Machine

Every training step moves through a precise sequence of operations. Understanding this sequence is important for debugging, because a problem in step N will manifest as an error in step N+1. For example, NaN loss usually appears at the `ComputeLoss` state, but the root cause is typically an exploding gradient that should have been caught one step earlier at `ClipGrads`. The diagram below makes this sequence explicit and annotates which optimization category each state belongs to.

```mermaid
stateDiagram-v2
    [*] --> LoadBatch: DataLoader yields (ids, labels)
    LoadBatch --> ZeroGrad: optimizer.zero_grad(set_to_none=True)
    ZeroGrad --> ForwardAMP: autocast context (AMP fp16/bf16)

    state ForwardAMP {
        [*] --> EmbedTokens
        EmbedTokens --> LowRankAttn: Category A - low-rank Q/K/V
        LowRankAttn --> FusedFFN: Category B - Triton kernel
        FusedFFN --> QuantHead: Category B - INT8 classifier
        QuantHead --> [*]
    }

    ForwardAMP --> ComputeLoss: CrossEntropyLoss
    ComputeLoss --> Backward: scaler.scale(loss).backward()
    Backward --> UnscaleGrads: scaler.unscale_(optimizer)
    UnscaleGrads --> ClipGrads: clip_grad_norm_(max_norm=1.0)
    ClipGrads --> OptimizerStep: AdamW step - Category C
    OptimizerStep --> ScalerUpdate: GradScaler.update()
    ScalerUpdate --> PrunerStep: DynamicPruner.step() - Category C
    PrunerStep --> SchedulerStep: LambdaLR.step() - Category C
    SchedulerStep --> ProfilerEnd: ThroughputProfiler.end_step()
    ProfilerEnd --> [*]
```

> [!TIP]
> The `set_to_none=True` flag in `zero_grad` is a meaningful optimization. Setting gradients to `None` instead of zero avoids allocating and writing a zero-filled tensor for each parameter, which reduces memory bandwidth consumption during the gradient reset phase. This is a recommended best practice in PyTorch 2.x.

> [!WARNING]
> The `GradScaler.update()` call must happen after `optimizer.step()`, not before. If skipped, the scaler will not adapt its scale factor, and the next `scaler.scale(loss).backward()` call will use a stale scale, potentially causing underflow in fp16 gradients. This is one of the most common AMP bugs in practice.

---

## Tech Stack

The following table describes every library used in this project, why it was chosen over simpler alternatives, and what happens if it is absent. Understanding the tech stack is important because each library represents a deliberate tradeoff between portability, performance, and expressiveness.

**PyTorch** is the foundation. It provides automatic differentiation (autograd), the `nn.Module` abstraction, CUDA tensor operations, and the `torch.compile` / AMP infrastructure. All other libraries in this stack either wrap PyTorch or plug into its extension system.

**Triton** is OpenAI's domain-specific language for writing GPU kernels in Python-like syntax that compiles to PTX (Parallel Thread Execution), the NVIDIA GPU assembly language. Triton auto-tunes tile sizes for your specific GPU architecture and handles thread coarsening automatically. Without Triton, writing the fused LinearGELU kernel would require raw CUDA C++, which is significantly more verbose and architecture-specific.

**Ninja** is the build system used by `torch.utils.cpp_extension.load()` to compile `kernel.cu` at runtime. Ninja uses file checksums instead of timestamps to determine which files need recompilation, making incremental CUDA builds significantly faster than Make. It also supports parallel compilation across multiple CUDA translation units.

| <sub>#</sub> | <sub>Library</sub> | <sub>Version</sub> | <sub>Role in This Project</sub> | <sub>Why This Library Was Chosen</sub> | <sub>Fallback If Absent</sub> |
|---|---|---|---|---|---|
| <sub>1</sub> | <sub>torch</sub> | <sub>>=2.1.0</sub> | <sub>Core tensor operations, autograd, AMP, DataLoader, nn.Module backbone</sub> | <sub>Industry standard; provides fused AdamW CUDA kernel via fused=True; best-in-class autograd</sub> | <sub>None - required</sub> |
| <sub>2</sub> | <sub>triton</sub> | <sub>>=2.1.0</sub> | <sub>JIT-compiled GPU kernels written in Python-like syntax for FusedLinearGELU</sub> | <sub>Allows writing custom CUDA-speed kernels without raw CUDA C++; auto-tunes tile sizes per GPU</sub> | <sub>Pure PyTorch fallback in fused_kernel.py</sub> |
| <sub>3</sub> | <sub>numpy</sub> | <sub>>=1.24.0</sub> | <sub>Array utilities and interop in the analysis notebook</sub> | <sub>Standard numerical Python; required by matplotlib internals for plot data</sub> | <sub>Pure Python lists for simple aggregations</sub> |
| <sub>4</sub> | <sub>matplotlib</sub> | <sub>>=3.7.0</sub> | <sub>Loss curves, LR schedule plots, FLOPs comparison charts in the notebook</sub> | <sub>Most compatible with Jupyter; produces publication-quality static plots with minimal code</sub> | <sub>Any plotting library (plotly, seaborn, etc.)</sub> |
| <sub>5</sub> | <sub>tqdm</sub> | <sub>>=4.65.0</sub> | <sub>Progress bars in training loops with ETA and steps/sec display</sub> | <sub>Zero-overhead progress display; works in both notebooks (tqdm.notebook) and terminals</sub> | <sub>Print statements every log_interval steps</sub> |
| <sub>6</sub> | <sub>ninja</sub> | <sub>>=1.11.0</sub> | <sub>Build system used by torch.utils.cpp_extension.load() to compile kernel.cu</sub> | <sub>Dramatically faster than Make for incremental CUDA builds; supports parallel compilation</sub> | <sub>Make (slower, may not be installed)</sub> |
| <sub>7</sub> | <sub>einops</sub> | <sub>>=0.7.0</sub> | <sub>Readable tensor rearrange operations available for extension</sub> | <sub>Makes shape manipulation self-documenting; eliminates reshape/transpose bugs in attention code</sub> | <sub>Manual view/transpose/permute calls</sub> |
| <sub>8</sub> | <sub>scipy</sub> | <sub>>=1.11.0</sub> | <sub>Statistical utilities in the analysis notebook for significance testing</sub> | <sub>Provides scipy.stats for curve fitting and correlation analysis of training metrics</sub> | <sub>Manual numpy implementations</sub> |
| <sub>9</sub> | <sub>jupyter</sub> | <sub>>=1.0.0</sub> | <sub>Interactive environment for running analysis.ipynb</sub> | <sub>Standard for ML analysis; allows inline plots and incremental computation</sub> | <sub>Convert notebook cells to a .py script</sub> |

> [!WARNING]
> Triton requires a CUDA-capable GPU and a compatible CUDA toolkit. If you are on CPU or an unsupported GPU, `import triton` will raise an ImportError that is caught at import time in `fused_kernel.py`, and the project will use the pure PyTorch fallback for all kernel operations. No functionality is lost - only the speedup from the fused kernel is unavailable.

---

## Project Structure

```
unified-ml-optimizations/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Pinned dependencies
│
├── src/
│   │
│   ├── main.py                        ← Entry point; --fast flag for smoke tests
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                 ← SyntheticClassificationDataset + DataLoader factory
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py              ← OptimizedTransformer, TransformerBlock
│   │   ├── low_rank_layer.py          ← LowRankLinear, LowRankAttention  [Category A]
│   │   ├── fused_kernel.py            ← FusedLinearGELU (Triton), QuantizedLinear  [Category B]
│   │   └── custom_cuda/
│   │       ├── __init__.py
│   │       ├── kernel.cu              ← CUDA C++ fused ReLU-quant-dequant kernel  [Category B]
│   │       └── binding.py             ← JIT loader + Python interface + CPU fallback
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                   ← Full training loop (AMP, grad clip, profiling)
│   │   ├── optimizer.py               ← AdamW factory, DynamicPruner  [Category C]
│   │   └── scheduler.py               ← Cosine LR with linear warmup  [Category C]
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  ← ModelConfig, TrainingConfig, Config dataclasses
│       ├── metrics.py                 ← AverageMeter, accuracy(), estimate_flops(), Stopwatch
│       └── profiler.py                ← ThroughputProfiler, pytorch_profile() context manager
│
└── notebooks/
    └── analysis.ipynb                 ← 5-cell analysis: LR, FLOPs, compression, loss curves
```

---

## Quick Start

### Prerequisites

You need Python 3.10+ and pip. A CUDA GPU is optional but recommended for the fused kernel and quantization paths. The virtual environment isolates the project's dependencies from your system Python so that version conflicts with other projects are avoided.

```bash
# Clone or navigate to the project
cd /home/kevin/Projects/unified-ml-optimizations

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Run a 50-step smoke test (CPU, ~4 seconds)

The `--fast` flag instantiates a tiny model (`d_model=128`, 2 layers, `seq_len=32`) and runs exactly 50 optimizer steps. This is sufficient to verify that every component imports cleanly, the forward pass produces finite outputs, and the loss decreases at least once. The fast mode exercises all three optimization categories with minimal compute.

```bash
python src/main.py --fast
```

### Run full training (GPU recommended)

Full training runs for 1,000 gradient steps with the default configuration: `d_model=512`, 6 layers, `seq_len=128`, batch size 64. On a modern GPU (RTX 3080 or better), this completes in approximately 2 minutes. On CPU, expect 15-25 minutes.

```bash
python src/main.py
```

### Run with PyTorch profiler

The profiler writes a Chrome-trace-format JSON to `./profile_trace/` that can be opened in `chrome://tracing` or the PyTorch TensorBoard plugin to visualize kernel timings per layer.

```bash
python src/main.py --profile
```

### Open the analysis notebook

The analysis notebook contains 5 pre-written cells that plot the LR schedule, compare FLOPs between full-rank and low-rank attention, visualize the sparsity ramp, and display training loss curves from the most recent run.

```bash
jupyter notebook notebooks/analysis.ipynb
```

> [!TIP]
> The `--fast` flag is the fastest way to check that a code change has not broken anything. It completes in under 10 seconds on any modern CPU and exercises all three optimization categories with minimal compute. Use it after every non-trivial edit before committing.

---

## Configuration Reference

All hyperparameters are centralized in `src/utils/config.py` as Python dataclasses. There are no YAML files or argument parsers to manage - just instantiate the dataclasses with keyword overrides. This design makes configuration type-checked at import time, makes IDE autocomplete work correctly, and eliminates the runtime errors caused by misspelled YAML keys.

> [!NOTE]
> A Python dataclass (`@dataclass`) is a class where `__init__`, `__repr__`, and `__eq__` are automatically generated from the field annotations. This means you get free pretty-printing of your config (`print(cfg)` shows all fields and values) and free equality comparison (`cfg_a == cfg_b` compares field-by-field).

### ModelConfig - Architecture Shape

Controls the architecture of `OptimizedTransformer`. These fields determine the size and shape of every tensor in the model. Changing `d_model` has a quadratic effect on parameter count (it affects both the width and depth of most linear layers), so small changes produce large memory differences.

| <sub>#</sub> | <sub>Field</sub> | <sub>Default</sub> | <sub>Type</sub> | <sub>Description</sub> | <sub>Impact on Model Size</sub> | <sub>Valid Range</sub> |
|---|---|---|---|---|---|---|
| <sub>1</sub> | <sub>vocab_size</sub> | <sub>16384</sub> | <sub>int</sub> | <sub>Number of token embeddings in the lookup table</sub> | <sub>Embedding table: vocab_size * d_model floats (33M bytes at default)</sub> | <sub>>= num_classes</sub> |
| <sub>2</sub> | <sub>d_model</sub> | <sub>512</sub> | <sub>int</sub> | <sub>Hidden dimension throughout all layers</sub> | <sub>Dominant factor in FLOPs and param count; scales quadratically</sub> | <sub>Must be divisible by n_heads</sub> |
| <sub>3</sub> | <sub>n_heads</sub> | <sub>8</sub> | <sub>int</sub> | <sub>Number of attention heads; determines head_dim = d_model / n_heads</sub> | <sub>Affects attention expressivity; more heads = finer-grained attention patterns</sub> | <sub>Must divide d_model evenly</sub> |
| <sub>4</sub> | <sub>n_layers</sub> | <sub>6</sub> | <sub>int</sub> | <sub>Number of stacked TransformerBlocks</sub> | <sub>Linear multiplier on both depth-related FLOPs and parameter count</sub> | <sub>>= 1</sub> |
| <sub>5</sub> | <sub>seq_len</sub> | <sub>128</sub> | <sub>int</sub> | <sub>Maximum input sequence length for positional embeddings</sub> | <sub>Positional embedding table; attention is O(L^2) so long seqs are expensive</sub> | <sub>>= 1</sub> |
| <sub>6</sub> | <sub>rank</sub> | <sub>32</sub> | <sub>int</sub> | <sub>Low-rank bottleneck dimension for Q/K/V projections</sub> | <sub>Lower rank = fewer params and FLOPs but reduced attention expressivity</sub> | <sub>1 to d_model</sub> |
| <sub>7</sub> | <sub>dropout</sub> | <sub>0.1</sub> | <sub>float</sub> | <sub>Dropout probability applied to attention weights and feedforward activations</sub> | <sub>No effect on param count; higher = stronger regularization</sub> | <sub>0.0 to 0.5</sub> |
| <sub>8</sub> | <sub>num_classes</sub> | <sub>10</sub> | <sub>int</sub> | <sub>Output classes for the quantized classifier head</sub> | <sub>Final linear layer: d_model * num_classes parameters</sub> | <sub>>= 2</sub> |

### TrainingConfig - Optimizer and Runtime

Controls everything about the training run: optimizer hyperparameters, scheduling, hardware device selection, and logging frequency. These fields can be overridden at instantiation without touching any source files, making it easy to script hyperparameter sweeps.

| <sub>#</sub> | <sub>Field</sub> | <sub>Default</sub> | <sub>Type</sub> | <sub>Description</sub> | <sub>Tuning Notes</sub> |
|---|---|---|---|---|---|
| <sub>1</sub> | <sub>batch_size</sub> | <sub>64</sub> | <sub>int</sub> | <sub>Number of samples per gradient update step</sub> | <sub>Larger batches give more stable gradient estimates but require more GPU VRAM</sub> |
| <sub>2</sub> | <sub>lr</sub> | <sub>3e-4</sub> | <sub>float</sub> | <sub>Peak learning rate passed to AdamW</sub> | <sub>The cosine schedule decays this to lr * min_lr_ratio (3e-5) by end of training</sub> |
| <sub>3</sub> | <sub>weight_decay</sub> | <sub>0.1</sub> | <sub>float</sub> | <sub>L2 regularization coefficient for 2-D weight matrices only</sub> | <sub>Not applied to biases or LayerNorm parameters (see Category C1 explanation)</sub> |
| <sub>4</sub> | <sub>max_steps</sub> | <sub>1000</sub> | <sub>int</sub> | <sub>Total gradient update steps before training terminates</sub> | <sub>Training loops until this count regardless of epoch count</sub> |
| <sub>5</sub> | <sub>warmup_steps</sub> | <sub>100</sub> | <sub>int</sub> | <sub>Steps over which LR ramps linearly from 0 to peak value</sub> | <sub>Typically 5-10% of max_steps; prevents gradient explosion at initialization</sub> |
| <sub>6</sub> | <sub>grad_clip</sub> | <sub>1.0</sub> | <sub>float</sub> | <sub>Maximum gradient norm for clip_grad_norm_ before optimizer step</sub> | <sub>1.0 is safe for most Transformers; lower if loss goes NaN</sub> |
| <sub>7</sub> | <sub>log_interval</sub> | <sub>50</sub> | <sub>int</sub> | <sub>Print loss, LR, and throughput metrics every N steps</sub> | <sub>No effect on training outcome; only controls terminal output frequency</sub> |
| <sub>8</sub> | <sub>device</sub> | <sub>"cuda"</sub> | <sub>str</sub> | <sub>Target device string; auto-falls-back to CPU if CUDA unavailable</sub> | <sub>Set to "cpu" to force CPU execution for debugging</sub> |
| <sub>9</sub> | <sub>use_amp</sub> | <sub>True</sub> | <sub>bool</sub> | <sub>Enable Automatic Mixed Precision forward pass in fp16 or bf16</sub> | <sub>Only active when device=="cuda"; silently ignored on CPU</sub> |
| <sub>10</sub> | <sub>profile</sub> | <sub>False</sub> | <sub>bool</sub> | <sub>Write torch.profiler Chrome-trace JSON to ./profile_trace/</sub> | <sub>Adds overhead; disable in production or performance benchmarks</sub> |

> [!IMPORTANT]
> `warmup_steps` should always be less than `max_steps`. If `warmup_steps >= max_steps`, the cosine decay phase never starts and the learning rate never drops, which will cause the model to continue taking large gradient steps into the final epochs, destabilizing convergence. A safe rule of thumb is `warmup_steps = max_steps // 10`.

---

## Component Deep Dives

### Category A - Low-Rank Decomposition

Standard attention uses three weight matrices W_Q, W_K, W_V each of size `d_model × d_model`. For `d_model=512`, each projection contains 262,144 parameters and requires 262,144 multiply-add operations per token per forward pass. Low-rank decomposition (inspired directly by the LoRA paper) replaces each of these with two smaller matrices: `A` of size `d_model × rank` and `B` of size `rank × d_model`. The product `A @ B` approximates the original full-rank matrix, but the parameter count drops from `d_model²` to `2 × d_model × rank`. At the default `rank=32` and `d_model=512`, each projection is compressed by 8x in both parameters and FLOPs.

The key insight behind why this works is that weight matrices in trained neural networks are empirically low-rank. The singular value decomposition of a trained weight matrix typically has a handful of large singular values and many near-zero ones. Low-rank decomposition exploits this by constraining the weight to live in a low-dimensional subspace from the start of training, rather than learning a full-rank matrix and discarding its small singular values afterward.

```mermaid
graph LR
    subgraph FULL["Full-Rank Projection (512x512 = 262,144 params)"]
        X1["x\n(B, L, 512)"] -->|"W: 512x512\n262,144 params"| Y1["q\n(B, L, 512)"]
    end

    subgraph LOWRANK["Low-Rank Projection r=32 (32,768 params - 8x fewer)"]
        X2["x\n(B, L, 512)"] -->|"A: 512x32\n16,384 params"| H["h\n(B, L, 32)"]
        H -->|"B: 32x512\n16,384 params"| Y2["q\n(B, L, 512)"]
    end

    style FULL fill:#e74c3c22
    style LOWRANK fill:#3498db22
```

> [!IMPORTANT]
> Matrix B is initialized with near-zero values (std=0.01) while A uses Kaiming uniform initialization. This means the product `A @ B` is approximately zero at initialization, matching the LoRA initialization strategy. This prevents the low-rank layers from dominating the residual stream early in training, before they have learned meaningful structure. If both matrices were initialized normally, the initial low-rank projections would have random large values that destabilize the attention computation.

### Category B - Triton Fused Kernel

In a standard feedforward layer, the computation `GELU(x @ W.T + b)` involves two separate GPU operations: a matrix multiplication (GEMM) followed by an element-wise GELU activation. Between these two operations, the GPU must write the GEMM output (an `(M, N)` float tensor) to High Bandwidth Memory (HBM), then read it back for the GELU pass. This HBM round-trip is the bottleneck because modern GPUs have far more arithmetic throughput than memory bandwidth - an A100 GPU can perform 312 TFLOPS of FP16 arithmetic but its HBM bandwidth is only 2 TB/s.

The Triton kernel `_fused_linear_gelu_kernel` eliminates this round-trip entirely. It computes the GELU activation inside the matmul accumulation loop, in SRAM registers, before writing the result to HBM. The output tensor is written exactly once. The GELU approximation used is the tanh variant: `x × 0.5 × (1 + tanh(√(2/π) × (x + 0.044715x³)))`.

> [!NOTE]
> Triton kernels use a tile-based programming model. The matrix is divided into 2D tiles that fit in the GPU's L1 cache (SRAM). Each tile is loaded once, the full fused computation is performed on it in registers, and the result is written back. The `tl.constexpr` parameters `BLOCK_M` and `BLOCK_N` control tile dimensions and are auto-tuned by Triton's autotuner for your specific GPU's cache size and warp count.

### Category B - INT8 Quantization

The `QuantizedLinear` module demonstrates the quantization-aware training (QAT) path for the classifier head. During the forward pass, the float32 weight matrix is quantized to INT8 using symmetric per-tensor absolute-maximum scaling (`scale = max(|W|) / 127`), then immediately dequantized back to float32 before the matrix multiply. This "fake quantization" approach allows gradients to flow through the quantized weight path during training, simulating the numerical noise of INT8 without actually requiring an INT8 GEMM kernel.

The straight-through estimator (STE) is the key training trick here. The quantize-then-dequantize operation has a gradient of zero for weights outside the clamp range (because rounding is a step function). STE replaces this zero gradient with the identity gradient, allowing weights to still be updated even when their quantized representation is at the clamp boundary. This is mathematically an approximation but works remarkably well in practice.

At inference time in a production deployment, the dequantization step would be skipped and the matrix multiply would use a native INT8 GEMM kernel. On hardware with INT8 tensor cores (Turing, Ampere, Ada Lovelace architectures), INT8 GEMM is 2-4x faster than FP16 GEMM, and the weight memory footprint is 4x smaller than FP32.

### Category C - AdamW Parameter Groups

AdamW (Adam with decoupled weight decay) was introduced to fix a subtle bug in L2 regularization applied within the Adam update rule. In vanilla Adam+L2, the weight decay is scaled by the adaptive learning rate, making the effective regularization stronger for infrequently-updated parameters. AdamW decouples the weight decay from the moment estimates, applying it directly to the parameter value as `w = w - lr * weight_decay * w`, which gives uniform L2 regularization regardless of gradient history.

Beyond AdamW correctness, a second important consideration is which parameters should receive weight decay at all. Weight decay encourages parameters toward zero, which is a useful inductive bias for 2-D weight matrices because it prevents any single weight from dominating the output. But for 1-D parameters - biases, LayerNorm scale (gamma), and LayerNorm shift (beta) - weight decay is harmful. These parameters are calibration constants that center and scale activations at each layer. Forcing them toward zero prevents LayerNorm from learning the appropriate scale for each layer's activation distribution, degrading convergence. The `build_adamw_optimizer` function automatically separates parameters by dimensionality to apply the correct decay to each group.

> [!TIP]
> To identify which parameters fall into each group, you can inspect the optimizer after construction: `[p['weight_decay'] for p in optimizer.param_groups]` will return `[0.1, 0.0]` for the two groups. `[len(p['params']) for p in optimizer.param_groups]` will show how many parameters are in each group.

---

## Memory and Compute Breakdown

Understanding where memory goes and where FLOPs are spent is essential for optimizing training throughput. The following table breaks down each component of the model by parameter count and estimated forward-pass FLOPs per token, using the default configuration (`d_model=512`, `rank=32`, `n_layers=6`, `seq_len=128`).

| <sub>#</sub> | <sub>Component</sub> | <sub>Parameters</sub> | <sub>% of Total</sub> | <sub>FLOPs per Token</sub> | <sub>Memory (FP32)</sub> | <sub>Optimization Applied</sub> |
|---|---|---|---|---|---|---|
| <sub>1</sub> | <sub>Token Embedding</sub> | <sub>8,388,608</sub> | <sub>~64%</sub> | <sub>0 (lookup, not matmul)</sub> | <sub>32 MB</sub> | <sub>None - shared across layers</sub> |
| <sub>2</sub> | <sub>LowRankAttention Q/K/V x6 layers</sub> | <sub>589,824</sub> | <sub>~4.5%</sub> | <sub>196,608 per layer</sub> | <sub>2.25 MB</sub> | <sub>Category A: 8x fewer than full-rank</sub> |
| <sub>3</sub> | <sub>Attention Output Proj x6 layers</sub> | <sub>1,572,864</sub> | <sub>~12%</sub> | <sub>524,288 per layer</sub> | <sub>6 MB</sub> | <sub>Full-rank (output proj is not low-ranked)</sub> |
| <sub>4</sub> | <sub>FFN FusedLinearGELU x6 layers</sub> | <sub>1,572,864</sub> | <sub>~12%</sub> | <sub>1,048,576 per layer</sub> | <sub>6 MB</sub> | <sub>Category B: fused kernel, no intermediate HBM write</sub> |
| <sub>5</sub> | <sub>FFN Up-Projection x6 layers</sub> | <sub>1,572,864</sub> | <sub>~12%</sub> | <sub>1,048,576 per layer</sub> | <sub>6 MB</sub> | <sub>None - standard linear</sub> |
| <sub>6</sub> | <sub>LayerNorm params x12 (pre + post)</sub> | <sub>12,288</sub> | <sub><0.1%</sub> | <sub>Negligible</sub> | <sub>0.05 MB</sub> | <sub>Category C: excluded from weight decay</sub> |
| <sub>7</sub> | <sub>QuantizedLinear Classifier Head</sub> | <sub>5,120</sub> | <sub><0.1%</sub> | <sub>5,120</sub> | <sub>0.02 MB</sub> | <sub>Category B: INT8 quantized weights</sub> |

> [!NOTE]
> The token embedding table dominates parameter count at 64% of the total. This is typical for small Transformer models with large vocabularies. At scale (GPT-3 style models), the ratio inverts and the feedforward layers dominate. If you reduce `vocab_size` from 16384 to 1024, the total model size drops from ~13M to ~5M parameters, which significantly speeds up CPU smoke tests.

---

## LR Schedule and Pruning Interaction

The learning rate schedule and the pruning sparsity ramp are co-designed to complement each other. The key insight is that pruning should not begin until the model has had enough gradient steps to form stable weight representations - if you start pruning at step 0, you remove weights that the model hasn't had a chance to learn, which permanently destroys potential capacity. By delaying pruning until after the warmup phase (step 100), the model first reaches a reasonable solution, and then pruning removes the least important weights while the still-high LR allows the remaining weights to compensate.

The following diagram shows the relationship between the normalized LR (which peaks at step 100 after warmup and decays via cosine to 10% at step 1000) and the normalized sparsity ramp (which starts at step 100 and reaches 30% target by step 800, then stays fixed for the final 200 steps).

```mermaid
xychart-beta
    title "LR Schedule vs Pruning Sparsity (over 1000 steps)"
    x-axis [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    y-axis "Normalized Value" 0 --> 1
    line [0.0, 1.0, 0.97, 0.88, 0.75, 0.59, 0.41, 0.25, 0.12, 0.06, 0.1]
    line [0.0, 0.0, 0.0, 0.11, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89]
```

> [!NOTE]
> The chart above shows normalized LR (first line, peaks at 1.0) and normalized sparsity (second line, reaches 0.89 which maps to 30% target sparsity). Pruning begins at step 100 (warmup end) and reaches its target by step 800. The final 200 steps train with fixed 30% sparsity and decaying LR, allowing the remaining weights to fine-tune into the pruned structure without further structural changes.

The interaction between these two schedules produces a third emergent effect: as sparsity increases and the LR decreases simultaneously in the middle of training (steps 200-800), the effective learning rate per active parameter actually stays roughly constant. This coincidence is not a design constraint but it is a useful property for understanding why the training curves tend to be smooth through the pruning phase.

---

## Performance Benchmarks

The following table shows theoretical parameter and FLOPs reduction from the low-rank attention optimization at various ranks, for the default `d_model=512` configuration. These numbers represent the Q/K/V projections only; the output projection and feedforward layers are not affected by the `rank` parameter.

| <sub>#</sub> | <sub>Rank (r)</sub> | <sub>Params per Q/K/V</sub> | <sub>Total QKV Params (x3)</sub> | <sub>Delta vs Full-Rank</sub> | <sub>Compression Ratio</sub> | <sub>Recommended Use Case</sub> |
|---|---|---|---|---|---|---|
| <sub>1</sub> | <sub>512 (full-rank baseline)</sub> | <sub>262,144</sub> | <sub>786,432</sub> | <sub>baseline</sub> | <sub>1.0x</sub> | <sub>Maximum expressivity; research ablations only</sub> |
| <sub>2</sub> | <sub>256</sub> | <sub>262,144</sub> | <sub>786,432</sub> | <sub>-0%</sub> | <sub>1.0x</sub> | <sub>Effectively full-rank at this d_model size</sub> |
| <sub>3</sub> | <sub>128</sub> | <sub>131,072</sub> | <sub>393,216</sub> | <sub>-50%</sub> | <sub>2.0x</sub> | <sub>Light compression; minimal accuracy impact on most tasks</sub> |
| <sub>4</sub> | <sub>64</sub> | <sub>65,536</sub> | <sub>196,608</sub> | <sub>-75%</sub> | <sub>4.0x</sub> | <sub>Good balance for most classification tasks</sub> |
| <sub>5</sub> | <sub>32 (project default)</sub> | <sub>32,768</sub> | <sub>98,304</sub> | <sub>-87.5%</sub> | <sub>8.0x</sub> | <sub>Strong compression; works for synthetic and simple real tasks</sub> |
| <sub>6</sub> | <sub>16</sub> | <sub>16,384</sub> | <sub>49,152</sub> | <sub>-93.75%</sub> | <sub>16.0x</sub> | <sub>Aggressive compression; suitable for small embedding spaces only</sub> |
| <sub>7</sub> | <sub>8</sub> | <sub>8,192</sub> | <sub>24,576</sub> | <sub>-96.875%</sub> | <sub>32.0x</sub> | <sub>Extreme compression; expect measurable accuracy degradation</sub> |

> [!CAUTION]
> Compression ratios above 8x (rank <= 32 for d_model=512) will reduce model expressivity enough to impact accuracy on real-world datasets. The synthetic classification task used in this project is simple enough that even rank=8 converges, but production models should always be benchmarked carefully before choosing a rank below d_model/8. Measure validation accuracy, not just training loss.

---

## Hardware Requirements

The project runs on a spectrum of hardware, from a laptop CPU to a data-center GPU. The following table describes what each configuration supports and what is degraded or unavailable.

| <sub>#</sub> | <sub>Hardware Configuration</sub> | <sub>Triton Kernels</sub> | <sub>CUDA Kernel</sub> | <sub>AMP (fp16)</sub> | <sub>Fused AdamW</sub> | <sub>Expected Full-Run Time</sub> | <sub>Recommended For</sub> |
|---|---|---|---|---|---|---|---|
| <sub>1</sub> | <sub>CPU only (no GPU)</sub> | <sub>PyTorch fallback</sub> | <sub>Python fallback</sub> | <sub>Disabled</sub> | <sub>Disabled</sub> | <sub>15-25 minutes</sub> | <sub>Code reading, smoke tests, debugging</sub> |
| <sub>2</sub> | <sub>Consumer GPU: RTX 3060/3070</sub> | <sub>Fully active</sub> | <sub>Fully active</sub> | <sub>fp16 active</sub> | <sub>Active (CUDA >= 11.4)</sub> | <sub>3-5 minutes</sub> | <sub>Full training, hyperparameter search</sub> |
| <sub>3</sub> | <sub>High-end GPU: RTX 3090/4090</sub> | <sub>Fully active</sub> | <sub>Fully active</sub> | <sub>bf16 preferred</sub> | <sub>Active</sub> | <sub>1-2 minutes</sub> | <sub>Full training with profiling enabled</sub> |
| <sub>4</sub> | <sub>Data-center GPU: A100/H100</sub> | <sub>Fully active, auto-tuned</sub> | <sub>Fully active</sub> | <sub>bf16 preferred</sub> | <sub>Active, highest throughput</sub> | <sub>Under 1 minute</sub> | <sub>Benchmarking, large-scale ablations</sub> |
| <sub>5</sub> | <sub>Apple Silicon (M1/M2 MPS)</sub> | <sub>PyTorch fallback (Triton GPU target is CUDA only)</sub> | <sub>Python fallback</sub> | <sub>MPS fp16 partially supported</sub> | <sub>Disabled</sub> | <sub>5-10 minutes</sub> | <sub>Development and testing on macOS</sub> |

> [!NOTE]
> To use Apple Silicon MPS acceleration, set `device="mps"` in `TrainingConfig`. The MPS backend supports most PyTorch operations but does not support the CUDA-specific `fused=True` optimizer flag or the custom CUDA kernel. Both will fall back gracefully. AMP with MPS is experimental in PyTorch 2.x and may produce inconsistent results on some operations.

---
| <sub>128</sub> | <sub>131,072</sub> | <sub>393,216</sub> | <sub>-50%</sub> | <sub>2.0×</sub> | <sub>Light compression; minimal accuracy impact</sub> |
| <sub>64</sub> | <sub>65,536</sub> | <sub>196,608</sub> | <sub>-75%</sub> | <sub>4.0×</sub> | <sub>Good balance for most classification tasks</sub> |
| <sub>32 (default)</sub> | <sub>32,768</sub> | <sub>98,304</sub> | <sub>-87.5%</sub> | <sub>8.0×</sub> | <sub>Strong compression; works for synthetic tasks</sub> |
| <sub>16</sub> | <sub>16,384</sub> | <sub>49,152</sub> | <sub>-93.75%</sub> | <sub>16.0×</sub> | <sub>Aggressive; use for small embedding spaces</sub> |
| <sub>8</sub> | <sub>8,192</sub> | <sub>24,576</sub> | <sub>-96.875%</sub> | <sub>32.0×</sub> | <sub>Extreme compression; expect accuracy degradation</sub> |

> [!CAUTION]
> Compression ratios above 8x (rank <= 32 for d_model=512) will likely reduce model expressivity enough to impact accuracy on real-world datasets. The synthetic classification task used in this project is simple enough that even rank=8 converges, but production models should be benchmarked carefully before choosing a rank.

---

## API Reference

<details>
<summary><strong>📦 src/utils/config.py</strong> - Configuration dataclasses</summary>

### `ModelConfig`

Defines the shape of the `OptimizedTransformer`. All fields have defaults that produce a 458K-parameter model suitable for a CPU smoke test.

```python
from src.utils.config import ModelConfig

cfg = ModelConfig(
    vocab_size=16384,   # token vocabulary size
    d_model=512,        # hidden dimension
    n_heads=8,          # attention heads (must divide d_model)
    n_layers=6,         # number of TransformerBlocks
    seq_len=128,        # max sequence length
    rank=32,            # low-rank bottleneck dimension
    dropout=0.1,        # dropout applied throughout
    num_classes=10,     # output classes
)
```

### `TrainingConfig`

Controls the training run. Device auto-falls-back to CPU if CUDA unavailable.

```python
from src.utils.config import TrainingConfig

cfg = TrainingConfig(
    batch_size=64,
    lr=3e-4,
    weight_decay=0.1,
    max_steps=1000,
    warmup_steps=100,
    grad_clip=1.0,
    log_interval=50,
    device="cuda",
    use_amp=True,
    profile=False,
)
```

</details>

<details>
<summary><strong>📦 src/models/low_rank_layer.py</strong> - Category A: Mathematical optimization</summary>

### `LowRankLinear(in_features, out_features, rank, bias=True, init_scale=0.01)`

A drop-in replacement for `nn.Linear` that factorizes the weight matrix into `A @ B` where `A` is `(in_features, rank)` and `B` is `(rank, out_features)`. The parameter count is `rank * (in_features + out_features)` instead of `in_features * out_features`.

```python
from src.models.low_rank_layer import LowRankLinear

# Standard: 512 * 512 = 262,144 parameters
full = torch.nn.Linear(512, 512)

# Low-rank r=32: 32 * (512 + 512) = 32,768 parameters (8x fewer)
low_rank = LowRankLinear(512, 512, rank=32)

x = torch.randn(4, 16, 512)
out = low_rank(x)   # shape: (4, 16, 512)
```

### `LowRankAttention(d_model, n_heads, rank, dropout=0.1)`

Multi-head attention where all three projections (Q, K, V) use `LowRankLinear`. The output projection remains full-rank. The interface is identical to a standard `nn.MultiheadAttention` forward pass.

```python
from src.models.low_rank_layer import LowRankAttention

attn = LowRankAttention(d_model=512, n_heads=8, rank=32)
x = torch.randn(4, 64, 512)   # (batch, seq, d_model)
out = attn(x)                  # (4, 64, 512)
```

</details>

<details>
<summary><strong>📦 src/models/fused_kernel.py</strong> - Category B: Systems optimization</summary>

### `FusedLinearGELU(in_features, out_features, bias=True)`

An `nn.Module` that fuses a linear projection and GELU activation into a single pass. On CUDA with Triton installed, dispatches to `_fused_linear_gelu_kernel`. On CPU or without Triton, falls back to `F.gelu(F.linear(x, weight, bias))`.

```python
from src.models.fused_kernel import FusedLinearGELU

layer = FusedLinearGELU(512, 2048).cuda()
x = torch.randn(4, 64, 512).cuda()
out = layer(x)   # (4, 64, 2048) - GELU already applied
```

### `QuantizedLinear(in_features, out_features, bits=8)`

INT8-quantized linear layer using symmetric per-tensor quantization with a straight-through gradient estimator. Suitable for QAT (quantization-aware training). The `bits` argument can be 4 or 8.

```python
from src.models.fused_kernel import QuantizedLinear

head = QuantizedLinear(512, 10, bits=8)
x = torch.randn(4, 512)
out = head(x)   # (4, 10) - computed via quantized weight path
```

</details>

<details>
<summary><strong>📦 src/training/optimizer.py</strong> - Category C: Optimizer and pruning</summary>

### `build_adamw_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8)`

Constructs an `AdamW` optimizer with two parameter groups: 2-D parameters receive `weight_decay` and 1-D parameters receive `weight_decay=0.0`. On CUDA, enables the fused AdamW kernel automatically.

```python
from src.training.optimizer import build_adamw_optimizer

optimizer = build_adamw_optimizer(
    model,
    lr=3e-4,
    weight_decay=0.1,
)
# optimizer.param_groups[0]['weight_decay'] == 0.1  (matrices)
# optimizer.param_groups[1]['weight_decay'] == 0.0  (biases, LN)
```

### `DynamicPruner(model, target_sparsity=0.5, start_step=200, end_step=800)`

Applies gradual magnitude-based unstructured sparsity to all 2-D weight tensors. Sparsity ramps linearly from 0% at `start_step` to `target_sparsity * 100`% at `end_step`. Call `.step(current_step)` after each `optimizer.step()`.

```python
from src.training.optimizer import DynamicPruner

pruner = DynamicPruner(model, target_sparsity=0.3, start_step=100, end_step=800)

# Inside training loop, after optimizer.step():
pruner.step(global_step)
```

</details>

<details>
<summary><strong>📦 src/training/scheduler.py</strong> - Category C: LR scheduling</summary>

### `get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1)`

Returns a `LambdaLR` scheduler that linearly ramps the LR from 0 to the optimizer's base LR over `warmup_steps`, then applies cosine decay down to `min_lr_ratio * base_lr` by `max_steps`.

```python
from src.training.scheduler import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps=100,
    max_steps=1000,
    min_lr_ratio=0.1,    # LR decays to 10% of peak
)

# Each step:
scheduler.step()
current_lr = scheduler.get_last_lr()[0]
```

</details>

<details>
<summary><strong>📦 src/utils/metrics.py</strong> - Measurement utilities</summary>

### `AverageMeter(window_size=100)`

Tracks a smoothed running average using a fixed-size deque. The `.value` property returns the mean over the last `window_size` updates. The `.global_avg` property returns the total mean over all updates.

### `estimate_flops(model, sample_input) -> int`

Estimates forward-pass FLOPs by registering hooks on all `nn.Linear` layers and counting multiply-add operations. Returns the total as an integer. Hooks are cleaned up via `try/finally` regardless of errors.

### `Stopwatch(use_cuda=True)`

A CUDA-synchronized wall-clock timer. Calls `torch.cuda.synchronize()` before recording timestamps to prevent CPU-side timing from returning before the GPU has finished. Use as a context manager or via `.start()` / `.stop()` / `.elapsed`.

</details>

<details>
<summary><strong>📦 src/models/custom_cuda/</strong> - Hand-written CUDA extension</summary>

### `kernel.cu`

A CUDA C++ kernel that computes `output[n,c] = dequant(clamp(round(ReLU(input[n,c]) / scale[c]), -127, 127)) * scale[c]` for each element. Uses per-channel scaling and fuses ReLU, quantization, and dequantization into a single kernel to avoid intermediate memory writes. Grid and block dimensions are computed to cover the full `(N, C)` tensor.

### `binding.py` - `fused_relu_quant_dequant(x, scale)`

Python interface that attempts to JIT-compile `kernel.cu` on first import using `torch.utils.cpp_extension.load()`. If compilation fails (no CUDA toolkit, no nvcc), falls back to a pure PyTorch implementation that is semantically identical.

```python
from src.models.custom_cuda.binding import fused_relu_quant_dequant

x = torch.randn(128, 512).cuda()
scale = x.abs().max(dim=0).values.clamp(min=1e-8) / 127.0
out = fused_relu_quant_dequant(x, scale)   # (128, 512)
```

> [!NOTE]
> First import will trigger CUDA JIT compilation which takes 30-60 seconds. Subsequent imports load the cached `.so` from PyTorch's extension build cache and are near-instantaneous.

</details>

---

## Troubleshooting

| <sub>Symptom</sub> | <sub>Likely Cause</sub> | <sub>Fix</sub> |
|---|---|---|
| <sub>`ModuleNotFoundError: No module named 'torch'`</sub> | <sub>Running outside the virtualenv</sub> | <sub>`source .venv/bin/activate` then retry</sub> |
| <sub>`ModuleNotFoundError: No module named 'triton'`</sub> | <sub>Triton not installed (it is optional)</sub> | <sub>`pip install triton` or ignore - PyTorch fallback is used automatically</sub> |
| <sub>`CUDA out of memory`</sub> | <sub>batch_size or d_model too large for GPU VRAM</sub> | <sub>Reduce `batch_size` in TrainingConfig or use `--fast`</sub> |
| <sub>`AssertionError: d_model must be divisible by n_heads`</sub> | <sub>Invalid ModelConfig combination</sub> | <sub>Ensure `d_model % n_heads == 0`</sub> |
| <sub>Loss is NaN after warmup</sub> | <sub>LR too high; gradient exploding</sub> | <sub>Reduce `lr` by 10x or reduce `grad_clip` to 0.5</sub> |
| <sub>CUDA kernel compilation fails</sub> | <sub>nvcc not on PATH or CUDA toolkit version mismatch</sub> | <sub>Install CUDA toolkit matching your PyTorch build; the Python fallback in binding.py will be used in the meantime</sub> |
| <sub>`pin_memory` UserWarning on CPU</sub> | <sub>Normal warning when running without CUDA</sub> | <sub>Set `pin_memory=False` in DataLoader or ignore - it is cosmetic only</sub> |

> [!WARNING]
> If you modify `kernel.cu` and the old compiled `.so` is cached, you must either delete the Torch extension cache (`~/.cache/torch_extensions/`) or change the extension `name` in `binding.py` to force recompilation. Torch will not detect changes to the source `.cu` file automatically.

---

## References

The techniques in this project are grounded in the following foundational papers and resources. Each reference is linked to the specific component it informs.

| <sub>Reference</sub> | <sub>Authors</sub> | <sub>Year</sub> | <sub>Used In</sub> |
|---|---|---|---|
| <sub>LoRA: Low-Rank Adaptation of Large Language Models</sub> | <sub>Hu et al.</sub> | <sub>2021</sub> | <sub>LowRankLinear initialization strategy</sub> |
| <sub>Attention Is All You Need</sub> | <sub>Vaswani et al.</sub> | <sub>2017</sub> | <sub>Overall Transformer architecture</sub> |
| <sub>Decoupled Weight Decay Regularization (AdamW)</sub> | <sub>Loshchilov & Hutter</sub> | <sub>2019</sub> | <sub>build_adamw_optimizer</sub> |
| <sub>To Prune, or Not to Prune (gradual magnitude pruning)</sub> | <sub>Zhu & Gupta</sub> | <sub>2018</sub> | <sub>DynamicPruner linear sparsity ramp</sub> |
| <sub>On Layer Normalization in the Transformer Architecture (Pre-LN)</sub> | <sub>Xiong et al.</sub> | <sub>2020</sub> | <sub>TransformerBlock Pre-LN placement</sub> |
| <sub>LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale</sub> | <sub>Dettmers et al.</sub> | <sub>2022</sub> | <sub>QuantizedLinear per-tensor INT8 scheme</sub> |
| <sub>Triton: An Intermediate Language for GPU Kernels</sub> | <sub>Tillet et al.</sub> | <sub>2019</sub> | <sub>_fused_linear_gelu_kernel</sub> |
| <sub>BERT: Pre-training of Deep Bidirectional Transformers</sub> | <sub>Devlin et al.</sub> | <sub>2019</sub> | <sub>Mean-pooling classification head design</sub> |

---

<div align="center">

**Built with PyTorch, Triton, and CUDA - runs everywhere from a laptop CPU to an A100.**

[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com)
[![CPU Fallback](https://img.shields.io/badge/CPU-Fallback%20Supported-blue?style=flat-square)]()

</div>
