"""
Train the Transformer sequential recommendation model.

Each training example is a sliding window of length MAX_SEQ_LEN drawn from a
user's watch history.  Given positions [0..T-1], the model predicts [1..T]
(shifted by one).  Loss is cross-entropy averaged over non-padding positions.

Hyperparameters match the plan:
  - embed_dim=128, 2 layers, 2 attention heads, ffn_dim=256
  - sliding window length 50
  - 5 epochs, Adam, print loss every 1000 batches

Saved to artifacts/transformer_model.pt.

Run:
  python scripts/train_transformer.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

# ── hyperparameters ────────────────────────────────────────────────────────────
EMBED_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
FFN_DIM = 256
MAX_SEQ_LEN = 10  # sliding window length
DROPOUT = 0.1
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 5
LOG_EVERY = 1000  # print running loss every N batches


# ── dataset ────────────────────────────────────────────────────────────────────


class SequenceDataset(Dataset):
    """
    Sliding-window dataset for next-item prediction.

    For a user sequence of length L we generate windows:
      [0:W], [1:W+1], …, [L-W:L]
    where W = MAX_SEQ_LEN.  Each window of length W produces W-1 training
    examples (the model predicts tokens 1..W given tokens 0..W-1).

    Short sequences (length < 2) are skipped.  Sequences shorter than W are
    left-padded with pad_idx so every window tensor has the same shape.
    """

    def __init__(
        self,
        sequences_train: dict[int, list[int]],
        max_seq_len: int,
        pad_idx: int,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.windows: list[list[int]] = []

        for seq in sequences_train.values():
            if len(seq) < 2:
                continue
            # Generate all windows of length max_seq_len+1 (+1 because the
            # last token is the target for the last input position)
            window_len = max_seq_len + 1
            if len(seq) <= window_len:
                # Pad and treat the whole sequence as one window
                padded = [pad_idx] * (window_len - len(seq)) + seq
                self.windows.append(padded)
            else:
                # Stride-1 sliding windows
                for start in range(len(seq) - window_len + 1):
                    self.windows.append(seq[start : start + window_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            src:     (max_seq_len,)  — input tokens  [t_0 … t_{T-1}]
            tgt:     (max_seq_len,)  — target tokens [t_1 … t_T]
            pad_mask:(max_seq_len,)  — True where src is padding (for attention mask)
        """
        window = self.windows[idx]
        src = torch.tensor(window[:-1], dtype=torch.long)  # all but last
        tgt = torch.tensor(window[1:], dtype=torch.long)  # all but first
        pad_mask = src == self.pad_idx  # True = ignore in attention
        return src, tgt, pad_mask


# ── training loop ──────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
) -> list[float]:
    """
    Train for EPOCHS epochs.  Loss is cross-entropy, ignoring padding positions.

    Returns per-epoch average loss for plotting later.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Ignore padding index in the loss so it doesn't contaminate gradients
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epoch_losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        batch_loss = 0.0  # running loss for LOG_EVERY window
        t0 = time.time()

        for batch_idx, (src, tgt, pad_mask) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        ):
            src = src.to(device)  # (batch, seq_len)
            tgt = tgt.to(device)  # (batch, seq_len)
            pad_mask = pad_mask.to(device)  # (batch, seq_len)

            optimizer.zero_grad()

            # Forward: (batch, seq_len, vocab_size)
            logits = model(src, src_key_padding_mask=pad_mask)

            # CrossEntropyLoss expects (batch, classes, seq_len) and (batch, seq_len)
            loss = criterion(logits.permute(0, 2, 1), tgt)

            loss.backward()
            # Gradient clipping — prevents exploding gradients with small batches
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_loss += loss.item()

            if (batch_idx + 1) % LOG_EVERY == 0:
                avg_batch_loss = batch_loss / LOG_EVERY
                print(
                    f"  Epoch {epoch} | batch {batch_idx + 1:>6} "
                    f"| running loss {avg_batch_loss:.4f}"
                )
                batch_loss = 0.0

        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  ({elapsed:.1f}s)")

    return epoch_losses


def main() -> None:
    # ── load processed data ──────────────────────────────────────────────────
    print("Loading processed sequences …")
    with open(PROCESSED_DIR / "sequences_train.pkl", "rb") as f:
        sequences_train: dict[int, list[int]] = pickle.load(f)
    with open(PROCESSED_DIR / "movie2idx.pkl", "rb") as f:
        movie2idx: dict[int, int] = pickle.load(f)

    vocab_size = len(movie2idx)
    pad_idx = vocab_size  # one past the last real index

    print(f"  Vocab size: {vocab_size:,}  |  Pad idx: {pad_idx}")

    dataset = SequenceDataset(sequences_train, max_seq_len=MAX_SEQ_LEN, pad_idx=pad_idx)
    print(f"  Windows: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ── device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── model ────────────────────────────────────────────────────────────────
    from models.transformer_rec import TransformerRecommender

    model = TransformerRecommender(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pad_idx=pad_idx,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── train ────────────────────────────────────────────────────────────────
    epoch_losses = train(model, loader, device, pad_idx)

    # ── save ─────────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "transformer_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ffn_dim": FFN_DIM,
            "max_seq_len": MAX_SEQ_LEN,
            "dropout": DROPOUT,
            "pad_idx": pad_idx,
            "epoch_losses": epoch_losses,
        },
        model_path,
    )
    print(f"\nSaved model → {model_path}")

    loss_path = ARTIFACTS_DIR / "transformer_loss_curve.json"
    with open(loss_path, "w") as f:
        json.dump({"epoch_losses": epoch_losses}, f)
    print(f"Saved loss curve → {loss_path}")


if __name__ == "__main__":
    main()
