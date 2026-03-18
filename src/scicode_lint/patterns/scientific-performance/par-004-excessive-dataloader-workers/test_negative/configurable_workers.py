import os

import torch
from torch.utils.data import DataLoader, Dataset


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, max_len=512):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = [ord(c) - ord("A") for c in seq[: self.max_len]]
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)


def create_training_loader(sequences, batch_size=64):
    dataset = ProteinSequenceDataset(sequences)
    num_workers = int(os.environ.get("DATALOADER_WORKERS", "4"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
