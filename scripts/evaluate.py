"""
Evaluate both models on the held-out test set and write metrics to artifacts/.

Metrics computed per model:
  Hit@10  — fraction of test (user, item) pairs where the item appears in top-10
  NDCG@10 — normalised discounted cumulative gain; rewards higher-ranked hits

For each user we have 3 test items.  Each test item is evaluated independently:
  - The model ranks all movies given the user's training history as context.
  - We check whether the held-out item lands in the top-10.

Results are written to artifacts/metrics.json for the Streamlit comparison page.

Run:
  python scripts/evaluate.py
"""

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

TOP_K = 10


# ── metric helpers ─────────────────────────────────────────────────────────────


def hit_at_k(ranked_list: list[int], target: int, k: int) -> float:
    """1.0 if target appears in the top-k ranked items, else 0.0."""
    return 1.0 if target in ranked_list[:k] else 0.0


def ndcg_at_k(ranked_list: list[int], target: int, k: int) -> float:
    """
    NDCG@k for a single relevant item.

    Since there is exactly one relevant item per query, the ideal DCG is 1.0
    (the item is at rank 1).  The actual DCG is 1/log2(rank+1) if the item
    is found within the top-k, else 0.
    """
    if target in ranked_list[:k]:
        rank = ranked_list[:k].index(target) + 1  # 1-based
        return 1.0 / math.log2(rank + 1)
    return 0.0


# ── model-specific rankers ─────────────────────────────────────────────────────


def build_mf_ranker(device: torch.device) -> tuple[Callable, dict]:
    """
    Load the trained MF model and return a ranking function.

    The ranker takes (user_id_raw, history_idxs) and returns a sorted list
    of movie indices (best first), excluding already-seen movies.
    """
    from models.matrix_factorization import MatrixFactorization

    checkpoint = torch.load(
        ARTIFACTS_DIR / "mf_model.pt", map_location=device, weights_only=False
    )
    model = MatrixFactorization(
        n_users=checkpoint["n_users"],
        n_movies=checkpoint["n_movies"],
        embed_dim=checkpoint["embed_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    user2idx: dict[int, int] = checkpoint["user2idx"]

    def rank_mf(user_id_raw: int, history_idxs: list[int]) -> list[int]:
        """Return all movies ranked by MF score, seen items excluded."""
        if user_id_raw not in user2idx:
            # Unknown user — fall back to random ordering (rare edge case)
            return list(range(checkpoint["n_movies"]))
        uid = user2idx[user_id_raw]
        ranked = model.recommend(
            user_id=uid,
            seen_movie_ids=history_idxs,
            top_k=TOP_K,
            device=device,
        )
        return [idx for idx, _ in ranked]

    return rank_mf, user2idx


def build_transformer_ranker(device: torch.device) -> Callable:
    """
    Load the trained Transformer model and return a ranking function.

    The ranker takes (history_idxs,) and returns the top-K movie indices.
    """
    from models.transformer_rec import TransformerRecommender

    checkpoint = torch.load(
        ARTIFACTS_DIR / "transformer_model.pt", map_location=device, weights_only=False
    )
    model = TransformerRecommender(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        ffn_dim=checkpoint["ffn_dim"],
        max_seq_len=checkpoint["max_seq_len"],
        dropout=0.0,  # disable dropout at inference time
        pad_idx=checkpoint["pad_idx"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    def rank_transformer(history_idxs: list[int]) -> list[int]:
        """Return top-K movie indices from the Transformer, seen items excluded."""
        ranked = model.recommend(
            history=history_idxs,
            top_k=TOP_K,
            seen_movie_ids=history_idxs,
            device=device,
        )
        return [idx for idx, _ in ranked]

    return rank_transformer


# ── evaluation loop ────────────────────────────────────────────────────────────


def evaluate_model(
    model_name: str,
    rank_fn: Callable,
    sequences_train: dict[int, list[int]],
    sequences_test: dict[int, list[int]],
    is_transformer: bool,
    user2idx: dict | None = None,
) -> dict[str, float]:
    """
    Compute Hit@K and NDCG@K for a model over all users and their test items.

    For each user:
      - History = training sequence (context given to the model)
      - Test items = the 3 held-out movies
      - Each test item is evaluated independently as a single query

    Args:
        model_name:    display name for progress bar
        rank_fn:       callable returning top-K movie indices
        sequences_train: {user_id: [train_movie_idxs]}
        sequences_test:  {user_id: [test_movie_idxs]}
        is_transformer:  if True, call rank_fn(history); else rank_fn(uid, history)
        user2idx:        MF-specific mapping of raw user_id → embedding index

    Returns:
        dict with hit@10 and ndcg@10
    """
    hits, ndcgs = [], []
    users = list(sequences_test.keys())

    for uid in tqdm(users, desc=f"Evaluating {model_name}"):
        history = sequences_train.get(uid, [])
        test_items = sequences_test[uid]

        if not history:
            # No training history — model has nothing to condition on; skip
            continue

        if is_transformer:
            top_k_indices = rank_fn(history)
        else:
            top_k_indices = rank_fn(uid, history)

        # Each of the 3 test movies is a separate evaluation query
        for target in test_items:
            hits.append(hit_at_k(top_k_indices, target, TOP_K))
            ndcgs.append(ndcg_at_k(top_k_indices, target, TOP_K))

    return {
        f"hit@{TOP_K}": round(sum(hits) / len(hits), 4) if hits else 0.0,
        f"ndcg@{TOP_K}": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "n_queries": len(hits),
    }


def main() -> None:
    # ── device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # ── load test data ───────────────────────────────────────────────────────
    print("Loading sequences …")
    with open(PROCESSED_DIR / "sequences_train.pkl", "rb") as f:
        sequences_train: dict[int, list[int]] = pickle.load(f)
    with open(PROCESSED_DIR / "sequences_test.pkl", "rb") as f:
        sequences_test: dict[int, list[int]] = pickle.load(f)
    print(f"  {len(sequences_test):,} users to evaluate\n")

    metrics: dict[str, dict] = {}

    # ── evaluate MF ──────────────────────────────────────────────────────────
    print("Loading MF model …")
    rank_mf, user2idx = build_mf_ranker(device)
    mf_metrics = evaluate_model(
        "MF",
        rank_mf,
        sequences_train,
        sequences_test,
        is_transformer=False,
        user2idx=user2idx,
    )
    metrics["matrix_factorization"] = mf_metrics
    print(
        f"MF    → Hit@{TOP_K}: {mf_metrics[f'hit@{TOP_K}']:.4f}  |  NDCG@{TOP_K}: {mf_metrics[f'ndcg@{TOP_K}']:.4f}\n"
    )

    # ── evaluate Transformer ─────────────────────────────────────────────────
    print("Loading Transformer model …")
    rank_transformer = build_transformer_ranker(device)
    tr_metrics = evaluate_model(
        "Transformer",
        rank_transformer,
        sequences_train,
        sequences_test,
        is_transformer=True,
    )
    metrics["transformer"] = tr_metrics
    print(
        f"Transformer → Hit@{TOP_K}: {tr_metrics[f'hit@{TOP_K}']:.4f}  |  NDCG@{TOP_K}: {tr_metrics[f'ndcg@{TOP_K}']:.4f}\n"
    )

    # ── save ─────────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    # Pretty-print comparison table
    print("\n── Results ─────────────────────────────────────────")
    print(f"{'Model':<25} {'Hit@10':>8}  {'NDCG@10':>8}")
    print("-" * 45)
    for name, m in metrics.items():
        print(f"{name:<25} {m[f'hit@{TOP_K}']:>8.4f}  {m[f'ndcg@{TOP_K}']:>8.4f}")


if __name__ == "__main__":
    main()
