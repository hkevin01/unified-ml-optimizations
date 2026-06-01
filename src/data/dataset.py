# =============================================================================
# FILE: src/data/dataset.py
# PURPOSE: Synthetic and CIFAR-10 dataset loaders for the training pipeline.
# =============================================================================

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class SyntheticClassificationDataset(Dataset):
    """
    ID: DATA-SYN-001
    Requirement: Generate synthetic token-sequence classification samples.
    Purpose: Allow pipeline smoke-tests without downloading external data.
    Inputs:
      - num_samples (int): total dataset size.
      - seq_len (int): sequence length (token count).
      - vocab_size (int): upper bound for token id sampling.
      - num_classes (int): number of output classes.
    Outputs: Dataset yielding (input_ids: LongTensor[seq_len], label: long).
    Preconditions: All ints > 0.
    Postconditions: __len__ returns num_samples; __getitem__ yields valid types.
    Assumptions: Labels are assigned as sample_index % num_classes.
    Side Effects: None (data generated lazily in __getitem__).
    Failure Modes: vocab_size=0 causes randint domain error.
    Error Handling: Input validation in __init__.
    Constraints: None.
    Verification: Instantiate and iterate one epoch in tests.
    References: None.
    """

    def __init__(
        self,
        num_samples: int = 8192,
        seq_len: int = 128,
        vocab_size: int = 16384,
        num_classes: int = 10,
        seed: int = 42,
    ) -> None:
        assert num_samples > 0 and seq_len > 0 and vocab_size > 0 and num_classes > 0
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        # Pre-generate to ensure reproducibility
        gen = torch.Generator()
        gen.manual_seed(seed)
        self._data = torch.randint(0, vocab_size, (num_samples, seq_len), generator=gen)
        self._labels = torch.arange(num_samples) % num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._data[idx], self._labels[idx]


def get_dataloaders(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    num_train: int = 8192,
    num_val: int = 1024,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    ID: DATA-LOAD-001
    Requirement: Return train and validation DataLoaders backed by synthetic data.
    Purpose: Decouple data loading from training logic.
    Inputs:
      - batch_size: samples per mini-batch.
      - seq_len: token sequence length.
      - vocab_size: token vocabulary size.
      - num_classes: number of output categories.
      - num_train: training set size.
      - num_val: validation set size.
      - num_workers: DataLoader worker processes.
    Outputs: (train_loader, val_loader) tuple.
    Preconditions: All int arguments > 0.
    Postconditions: Loaders yield (LongTensor[B,L], LongTensor[B]) batches.
    Assumptions: Sufficient RAM for pre-generated data.
    Side Effects: None.
    Failure Modes: num_workers > 0 on platforms without fork support may error.
    Error Handling: Set num_workers=0 on Windows if fork fails.
    Constraints: None.
    Verification: Iterate one batch and check shapes in tests.
    References: None.
    """
    train_ds = SyntheticClassificationDataset(num_train, seq_len, vocab_size, num_classes)
    val_ds = SyntheticClassificationDataset(num_val, seq_len, vocab_size, num_classes, seed=99)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
