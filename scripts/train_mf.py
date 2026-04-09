"""
Train the Matrix Factorization baseline model.

Training treats every (user, movie) pair in the training sequences as a
positive example with the actual star rating as the target.  We minimise
MSE between predicted and actual ratings.

The trained model is saved to artifacts/mf_model.pt.

Run:
  python scripts/train_mf.py
"""

import json
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Resolve paths relative to repo root regardless of where the script is called from
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

# ── hyperparameters ────────────────────────────────────────────────────────────
EMBED_DIM = 64
BATCH_SIZE = 4096  # large batch fits comfortably in memory; speeds up training
LR = 1e-3
EPOCHS = 5
WEIGHT_DECAY = 1e-5  # mild L2 regularisation to prevent embedding collapse


# ── dataset ────────────────────────────────────────────────────────────────────


class RatingsDataset(Dataset):
    """
    Flat dataset of (user_idx, movie_idx, rating) triples derived from the
    per-user training sequences.

    We don't have per-rating labels in the sequences (timestamps only), so we
    assign a synthetic rating of 1.0 for every interaction — the model learns
    which user-movie pairs co-occur, which is sufficient for ranking.

    If the raw ratings file is available and we wanted true star ratings, we
    could join back to ratings.csv here.  For simplicity we use implicit feedback.
    """

    def __init__(self, sequences_train: dict[int, list[int]]) -> None:
        # Build a flat list of (user_idx, movie_idx) pairs
        # We use the user's position in the sorted dict as the user_idx
        self.user_ids: list[int] = []
        self.movie_ids: list[int] = []

        # Map userId (arbitrary int) → contiguous index for the embedding table
        user_list = sorted(sequences_train.keys())
        self.user2idx = {uid: i for i, uid in enumerate(user_list)}

        for uid, seq in sequences_train.items():
            user_idx = self.user2idx[uid]
            for movie_idx in seq:
                self.user_ids.append(user_idx)
                self.movie_ids.append(movie_idx)

        self.user_ids = torch.tensor(self.user_ids, dtype=torch.long)
        self.movie_ids = torch.tensor(self.movie_ids, dtype=torch.long)
        # Implicit rating: 1.0 for every observed interaction
        self.ratings = torch.ones(len(self.user_ids), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


# ── training loop ──────────────────────────────────────────────────────────────


def train(model: nn.Module, loader: DataLoader, device: torch.device) -> list[float]:
    """Run the full training loop, return per-epoch average losses."""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    epoch_losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        t0 = time.time()

        for user_ids, movie_ids, ratings in tqdm(
            loader, desc=f"Epoch {epoch}/{EPOCHS}"
        ):
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            preds = model(user_ids, movie_ids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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

    n_movies = len(movie2idx)

    dataset = RatingsDataset(sequences_train)
    n_users = len(dataset.user2idx)
    print(
        f"  {len(dataset):,} training interactions | {n_users:,} users | {n_movies:,} movies"
    )

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # ── device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU — faster than CPU for PyTorch on M-series Macs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── model ────────────────────────────────────────────────────────────────
    # Import here (not at top) so the script can be run from the repo root
    import sys

    sys.path.insert(0, str(ROOT))
    from models.matrix_factorization import MatrixFactorization

    model = MatrixFactorization(n_users=n_users, n_movies=n_movies, embed_dim=EMBED_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── train ────────────────────────────────────────────────────────────────
    epoch_losses = train(model, loader, device)

    # ── save ─────────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "mf_model.pt"
    # Save weights + the user2idx mapping (needed at inference time)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user2idx": dataset.user2idx,
            "n_users": n_users,
            "n_movies": n_movies,
            "embed_dim": EMBED_DIM,
            "epoch_losses": epoch_losses,
        },
        model_path,
    )
    print(f"\nSaved model → {model_path}")

    # Also persist losses alongside the transformer curve for the comparison page
    loss_path = ARTIFACTS_DIR / "mf_loss_curve.json"
    with open(loss_path, "w") as f:
        json.dump({"epoch_losses": epoch_losses}, f)
    print(f"Saved loss curve → {loss_path}")


if __name__ == "__main__":
    main()
