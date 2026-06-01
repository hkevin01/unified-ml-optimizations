# =============================================================================
# FILE: src/utils/config.py
# PURPOSE: Centralized configuration for the unified-ml-optimizations project.
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    ID: CFG-MODEL-001
    Requirement: Define hyperparameters for the base model and low-rank layers.
    Purpose: Single source of truth for architectural dimensions.
    Inputs: None (instantiated with defaults or keyword overrides).
    Outputs: ModelConfig instance.
    Preconditions: None.
    Postconditions: All fields are set to valid positive integers or floats.
    Assumptions: vocab_size > 0, d_model > 0, rank > 0 and rank <= d_model.
    Side Effects: None.
    Failure Modes: Incorrect field types will raise TypeError at runtime.
    Error Handling: Relies on Python dataclass type hints for documentation;
                    validation is performed in train.py.
    Constraints: rank should be << d_model for meaningful low-rank compression.
    Verification: Unit tests in tests/test_config.py.
    References: Low-rank decomposition literature (LoRA, SVD-based methods).
    """
    vocab_size: int = 16384
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    seq_len: int = 128
    rank: int = 32          # low-rank bottleneck dimension
    dropout: float = 0.1
    num_classes: int = 10


@dataclass
class TrainingConfig:
    """
    ID: CFG-TRAIN-001
    Requirement: Define all training hyperparameters.
    Purpose: Separates training concerns from model architecture.
    Inputs: None.
    Outputs: TrainingConfig instance.
    Preconditions: None.
    Postconditions: All numeric fields hold valid positive values.
    Assumptions: lr and weight_decay are compatible with AdamW stability.
    Side Effects: None.
    Failure Modes: Negative lr or weight_decay cause divergence.
    Error Handling: Validated in train.py before optimizer construction.
    Constraints: max_steps >= warmup_steps.
    Verification: Integration test via main.py dry run.
    References: AdamW paper (Loshchilov & Hutter, 2019).
    """
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_interval: int = 50
    device: str = "cuda"
    use_amp: bool = True        # automatic mixed precision
    profile: bool = False


@dataclass
class Config:
    """
    ID: CFG-ROOT-001
    Requirement: Aggregate all sub-configs into one root config object.
    Purpose: Pass a single config object through the training pipeline.
    Inputs: Optional sub-config instances.
    Outputs: Config instance.
    Preconditions: None.
    Postconditions: model and training fields are fully initialized.
    Assumptions: Default sub-configs are valid for a smoke-test run.
    Side Effects: None.
    Failure Modes: Sub-config validation failures propagate here.
    Error Handling: Delegates to sub-config validation.
    Constraints: None.
    Verification: main.py entry point.
    References: None.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
