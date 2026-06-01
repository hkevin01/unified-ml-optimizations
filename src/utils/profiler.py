# =============================================================================
# FILE: src/utils/profiler.py
# PURPOSE: Lightweight profiling wrapper around PyTorch profiler and Stopwatch.
# =============================================================================

import contextlib
from typing import Optional

import torch
import torch.profiler

from src.utils.metrics import Stopwatch


class ThroughputProfiler:
    """
    ID: PROF-TP-001
    Requirement: Measure samples-per-second throughput for a training step.
    Purpose: Quantify hardware utilization and optimization gains.
    Inputs:
      - batch_size (int): number of samples per step.
      - device (str): 'cuda' or 'cpu'.
    Outputs: ThroughputProfiler instance; .report() prints stats.
    Preconditions: batch_size >= 1.
    Postconditions: Internal state tracks cumulative steps and elapsed time.
    Assumptions: start_step/end_step called in matching pairs.
    Side Effects: Calls CUDA synchronize on GPU device.
    Failure Modes: Mismatched start/end calls yield incorrect timing.
    Error Handling: Stopwatch handles missing CUDA gracefully.
    Constraints: Do not call report() before at least one step.
    Verification: Smoke test in main.py with profile=True.
    References: None.
    """

    def __init__(self, batch_size: int, device: str = "cuda") -> None:
        self.batch_size = batch_size
        self._watch = Stopwatch(use_cuda=(device == "cuda"))
        self._total_elapsed: float = 0.0
        self._steps: int = 0

    def start_step(self) -> None:
        self._watch.start()

    def end_step(self) -> None:
        self._watch.stop()
        self._total_elapsed += self._watch.elapsed
        self._steps += 1

    def report(self) -> dict:
        """
        ID: PROF-TP-REPORT-001
        Requirement: Return throughput statistics as a dict.
        Purpose: Structured output for logging and notebook analysis.
        Inputs: None (uses internal state).
        Outputs: dict with keys 'steps', 'total_time_s', 'throughput_samples_s'.
        Preconditions: At least one step recorded.
        Postconditions: All values are finite floats.
        Assumptions: _steps > 0.
        Side Effects: None.
        Failure Modes: Division by zero if _steps == 0; returns zeros.
        Error Handling: Guards against zero steps.
        Constraints: None.
        Verification: Assert throughput > 0 in tests.
        References: None.
        """
        if self._steps == 0:
            return {"steps": 0, "total_time_s": 0.0, "throughput_samples_s": 0.0}
        throughput = (self._steps * self.batch_size) / self._total_elapsed
        return {
            "steps": self._steps,
            "total_time_s": self._total_elapsed,
            "throughput_samples_s": throughput,
        }

    def print_report(self) -> None:
        stats = self.report()
        print(
            f"[Profiler] Steps: {stats['steps']} | "
            f"Total time: {stats['total_time_s']:.3f}s | "
            f"Throughput: {stats['throughput_samples_s']:.1f} samples/s"
        )


@contextlib.contextmanager
def pytorch_profile(output_dir: str = "./profile_trace", use_cuda: bool = True):
    """
    ID: PROF-PT-001
    Requirement: Context manager wrapping torch.profiler for trace export.
    Purpose: Capture detailed GPU/CPU activity for visualization in TensorBoard.
    Inputs:
      - output_dir (str): directory to write chrome trace JSON.
      - use_cuda (bool): enable CUDA activity recording.
    Outputs: Yields a torch.profiler.profile instance.
    Preconditions: output_dir is writable.
    Postconditions: Trace file written to output_dir on context exit.
    Assumptions: CUDA available when use_cuda=True.
    Side Effects: Writes files to disk; may increase step latency.
    Failure Modes: Disk full -> OSError; CUDA unavailable -> CPU-only trace.
    Error Handling: Caller handles OSError.
    Constraints: Use for diagnosis only; disable in production training.
    Verification: Open trace in chrome://tracing after a profiled run.
    References: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        yield prof
