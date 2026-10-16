"""Deep Knowledge Tracing (Piech et al., 2015) with a small GRU. Optional: needs `.[deep]`.

Each answer becomes one token (word, right/wrong). The GRU reads a learner's tokens in order and,
at every step, outputs P(correct) for every word; the loss uses the word actually asked next.
Step t only sees tokens before t (the inputs are shifted by one), so there is no leakage.

DKT has no notion of clock time here, only order, so it cannot see forgetting between days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .bkt import to_sequences


class _Net(nn.Module):
    def __init__(self, n_tokens: int, n_items: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_tokens, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_items)

    def forward(self, tokens: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.embed(tokens))
        return self.out(states).gather(-1, items.unsqueeze(-1)).squeeze(-1)


class DKT:
    name = "DKT (GRU)"

    def __init__(self, hidden: int = 32, epochs: int = 8, lr: float = 5e-3, seed: int = 0) -> None:
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.index: dict[str, int] = {}
        self.net: _Net | None = None

    def _tensors(self, frame: pd.DataFrame) -> tuple[torch.Tensor, ...]:
        """Padded (tokens, items, labels, mask, rows). Item 0 means "not seen in training"."""
        seq = to_sequences(frame)
        rows = np.maximum(seq.rows, 0)
        items = frame["item_id"].map(self.index).fillna(0).to_numpy(dtype=np.int64)[rows]
        labels = seq.obs.astype(np.int64)
        tokens = 1 + 2 * items + labels  # token 0 = start of sequence
        shifted = np.concatenate([np.zeros((len(tokens), 1), np.int64), tokens[:, :-1]], axis=1)
        return (
            torch.as_tensor(shifted * seq.mask),
            torch.as_tensor(items),
            torch.as_tensor(seq.obs, dtype=torch.float32),
            torch.as_tensor(seq.mask),
            torch.as_tensor(seq.rows),
        )

    def fit(self, frame: pd.DataFrame) -> DKT:
        torch.manual_seed(self.seed)
        self.index = {item: k + 1 for k, item in enumerate(sorted(frame["item_id"].unique()))}
        n_items = len(self.index) + 1
        self.net = _Net(1 + 2 * n_items, n_items, self.hidden)
        tokens, items, labels, mask, _ = self._tensors(frame)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for batch in torch.randperm(len(tokens), generator=generator).split(64):
                logits = self.net(tokens[batch], items[batch])
                loss = loss_fn(logits[mask[batch]], labels[batch][mask[batch]])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("fit the model first")
        tokens, items, _, mask, rows = self._tensors(frame)
        with torch.no_grad():
            p = torch.sigmoid(self.net(tokens, items))
        out = np.empty(len(frame))
        out[rows[mask].numpy()] = p[mask].numpy()
        return out
