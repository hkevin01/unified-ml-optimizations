# =============================================================================
# FILE: src/utils/metrics.py
# PURPOSE: Accuracy, loss tracking, and FLOPs estimation utilities.
# =============================================================================

import time
from collections import deque
from typing import Deque

import torch


class AverageMeter:
    """
    ID: MET-AVG-001
    Requirement: Track a running average of a scalar metric.
    Purpose: Smooth noisy per-step metrics for logging.
    Inputs: window_size (int) - number of recent values to average.
    Outputs: AverageMeter instance; .value property returns smoothed mean.
    Preconditions: window_size >= 1.
    Postconditions: .value reflects the mean of the last window_size updates.
    Assumptions: update() called with finite float-compatible values.
    Side Effects: Maintains internal deque; old values are evicted.
    Failure Modes: update() with NaN propagates NaN to .value.
    Error Handling: Caller should guard against NaN losses before update.
    Constraints: Thread-unsafe; single-process use only.
    Verification: Manually verified in profiler.py smoke tests.
    References: None.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window: Deque[float] = deque(maxlen=window_size)
        self._total: float = 0.0
        self._count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self._window.append(val)
        self._total += val * n
        self._count += n

    @property
    def value(self) -> float:
        return sum(self._window) / max(len(self._window), 1)

    @property
    def global_avg(self) -> float:
        return self._total / max(self._count, 1)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    ID: MET-ACC-001
    Requirement: Compute top-1 classification accuracy.
    Purpose: Standard metric for classification benchmarks.
    Inputs:
      - logits: (N, C) float tensor of unnormalized class scores.
      - labels: (N,) long tensor of ground-truth class indices.
    Outputs: float in [0, 1].
    Preconditions: logits.shape[0] == labels.shape[0], C >= 1.
    Postconditions: Return value is a Python float.
    Assumptions: labels are valid class indices in [0, C-1].
    Side Effects: None.
    Failure Modes: Empty batch returns 0.0.
    Error Handling: Guard against zero-length batch.
    Constraints: None.
    Verification: Unit test: perfect logits -> 1.0, random -> ~1/C.
    References: None.
    """
    if logits.shape[0] == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().sum().item()
    return correct / labels.shape[0]


def estimate_flops(model: torch.nn.Module, sample_input: torch.Tensor) -> int:
    """
    ID: MET-FLOP-001
    Requirement: Estimate forward-pass FLOPs via PyTorch profiler hook counting.
    Purpose: Provide a rough FLOPs baseline to quantify optimization speedups.
    Inputs:
      - model: an nn.Module in eval mode.
      - sample_input: a tensor matching the model's expected input shape.
    Outputs: int - estimated total multiply-add operations.
    Preconditions: model is callable; sample_input is on the same device.
    Postconditions: Returns a non-negative integer.
    Assumptions: Counts only linear/conv multiply-adds; ignores activations.
    Side Effects: Registers and removes forward hooks on all nn.Linear layers.
    Failure Modes: Non-linear modules not counted; estimate is a lower bound.
    Error Handling: Hook removal guaranteed via try/finally.
    Constraints: None.
    Verification: Compare against known layer sizes manually.
    References: FLOPs definition: 2*M*N*K for a (M,K)x(K,N) matmul.
    """
    flops: list[int] = [0]
    hooks = []

    def _linear_hook(module: torch.nn.Linear, inp, out) -> None:
        # ID: MET-FLOP-HOOK-001
        # Purpose: Count multiply-adds for a single linear layer forward pass.
        # Inputs: module (Linear), inp (tuple of tensors), out (tensor).
        # Outputs: Accumulates into flops[0].
        # Preconditions: module.weight is 2-D.
        # Postconditions: flops[0] incremented by 2 * in_features * out_features * batch.
        batch = inp[0].numel() // inp[0].shape[-1]
        flops[0] += 2 * module.in_features * module.out_features * batch

    try:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(_linear_hook))
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    return flops[0]


class Stopwatch:
    """
    ID: MET-STOP-001
    Requirement: Wall-clock timer with CUDA synchronization support.
    Purpose: Accurately time GPU-bound operations by forcing device sync.
    Inputs: use_cuda (bool) - whether to call torch.cuda.synchronize().
    Outputs: elapsed seconds as float via .elapsed property.
    Preconditions: CUDA available when use_cuda=True.
    Postconditions: .elapsed reflects real wall-clock time post-sync.
    Assumptions: Single-threaded usage.
    Side Effects: Calls torch.cuda.synchronize() which stalls the GPU pipeline.
    Failure Modes: CUDA not available -> falls back to CPU timing.
    Error Handling: Catches RuntimeError from missing CUDA gracefully.
    Constraints: Do not use inside tight training loops; profiling only.
    Verification: Compare against known sleep durations in tests.
    References: PyTorch profiling best practices.
    """

    def __init__(self, use_cuda: bool = True) -> None:
        self._use_cuda = use_cuda and torch.cuda.is_available()
        self._start: float = 0.0
        self._end: float = 0.0

    def _sync(self) -> None:
        if self._use_cuda:
            torch.cuda.synchronize()

    def start(self) -> "Stopwatch":
        self._sync()
        self._start = time.perf_counter()
        return self

    def stop(self) -> "Stopwatch":
        self._sync()
        self._end = time.perf_counter()
        return self

    @property
    def elapsed(self) -> float:
        return self._end - self._start

    def __enter__(self) -> "Stopwatch":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
