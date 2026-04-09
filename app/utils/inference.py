"""
Model loading and inference utilities.

All resources (models + data mappings) are loaded once via st.cache_resource
and shared across all pages for the lifetime of the Streamlit server process.
"""

import pickle
import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RAW_DIR = ROOT / "data" / "raw" / "ml-25m"


# ── resource loading ───────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_pickle(name: str):
    path = PROCESSED_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_genres(idx2movie: dict[int, int]) -> dict[int, str]:
    """Map movie index → human-readable genre string from movies.csv."""
    movies_path = RAW_DIR / "movies.csv"
    if not movies_path.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(movies_path)
    mid_to_genre = dict(zip(df["movieId"], df["genres"]))
    return {
        idx: mid_to_genre.get(mid, "Unknown").replace("|", " · ")
        for idx, mid in idx2movie.items()
    }


def _load_mf_model(device: torch.device):
    """Reconstruct the MF model from its checkpoint."""
    path = ARTIFACTS_DIR / "mf_model.pt"
    if not path.exists():
        return None, None
    from models.matrix_factorization import MatrixFactorization

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MatrixFactorization(
        n_users=ckpt["n_users"],
        n_movies=ckpt["n_movies"],
        embed_dim=ckpt["embed_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["user2idx"]


def _load_transformer_model(device: torch.device):
    """Reconstruct the Transformer model from its checkpoint."""
    path = ARTIFACTS_DIR / "transformer_model.pt"
    if not path.exists():
        return None
    from models.transformer_rec import TransformerRecommender

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TransformerRecommender(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        num_heads=ckpt["num_heads"],
        num_layers=ckpt["num_layers"],
        ffn_dim=ckpt["ffn_dim"],
        max_seq_len=ckpt["max_seq_len"],
        dropout=0.0,  # inference — no regularisation needed
        pad_idx=ckpt["pad_idx"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@st.cache_resource(show_spinner="Loading models and data…")
def load_resources() -> dict:
    """
    Load all models and processed data into a shared dict.

    Decorated with st.cache_resource so this runs exactly once per server
    process, no matter how many pages or users trigger it.

    Returns a dict with keys:
      device, movie2idx, idx2movie, idx2title, idx2genre,
      sequences_train, sequences_test,
      mf_model, mf_user2idx,  (None if not trained yet)
      transformer_model       (None if not trained yet)
    """
    device = _get_device()
    res: dict = {"device": device, "ready": False}

    # ── processed data ────────────────────────────────────────────────────────
    for name in [
        "movie2idx",
        "idx2movie",
        "idx2title",
        "sequences_train",
        "sequences_test",
    ]:
        res[name] = _load_pickle(name)

    if res["idx2movie"] is None:
        # Preprocessing hasn't been run yet — app will show setup instructions
        return res

    res["idx2genre"] = _load_genres(res["idx2movie"])

    # ── models ────────────────────────────────────────────────────────────────
    res["mf_model"], res["mf_user2idx"] = _load_mf_model(device)
    res["transformer_model"] = _load_transformer_model(device)

    res["ready"] = True
    return res


# ── search ─────────────────────────────────────────────────────────────────────


def search_movies(
    query: str,
    idx2title: dict[int, str],
    n: int = 25,
) -> list[tuple[int, str]]:
    """
    Case-insensitive substring search over all movie titles.

    Returns up to n results sorted by title length (shorter = more likely
    to be the exact film the user wants, not a sequel or collection).
    """
    if not query.strip():
        return []
    q = query.lower()
    matches = [(idx, title) for idx, title in idx2title.items() if q in title.lower()]
    matches.sort(key=lambda x: len(x[1]))
    return matches[:n]


# ── inference ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def recommend_transformer(
    model,
    history_idxs: list[int],
    top_k: int = 10,
    device: torch.device | None = None,
) -> list[tuple[int, float]]:
    """Run Transformer next-item prediction given an ordered watch history."""
    if device is None:
        device = next(model.parameters()).device
    return model.recommend(
        history=history_idxs,
        top_k=top_k,
        seen_movie_ids=history_idxs,
        device=device,
    )


@torch.no_grad()
def recommend_mf(
    model,
    user_id: int,
    history_idxs: list[int],
    top_k: int = 10,
    device: torch.device | None = None,
) -> list[tuple[int, float]]:
    """Run MF recommendation for a known training-set user."""
    if device is None:
        device = next(model.parameters()).device
    return model.recommend(
        user_id=user_id,
        seen_movie_ids=history_idxs,
        top_k=top_k,
        device=device,
    )


def find_closest_training_user(
    query_history: list[int],
    sequences_train: dict[int, list[int]] | None,
    mf_user2idx: dict[int, int],
) -> tuple[int, int]:
    """
    Find the training user whose watch history has the most overlap with
    query_history.  Used to proxy MF recommendations for arbitrary inputs.

    sequences_train may be None on Streamlit Cloud (the pkl is too large to
    commit; only the model weights and small mappings are in the repo).
    In that case we fall back to user index 0.

    Returns (raw_user_id, mf_user_idx).
    """
    if not sequences_train:
        # sequences_train not available (Streamlit Cloud deployment) — use first user
        first_uid = next(iter(mf_user2idx))
        return first_uid, mf_user2idx[first_uid]

    query_set = set(query_history)
    best_uid, best_overlap = -1, -1
    for uid, seq in sequences_train.items():
        if uid not in mf_user2idx:
            continue
        overlap = len(query_set & set(seq))
        if overlap > best_overlap:
            best_overlap = overlap
            best_uid = uid
    return best_uid, mf_user2idx.get(best_uid, 0)


# ── formatting ─────────────────────────────────────────────────────────────────


def format_recommendations(
    ranked: list[tuple[int, float]],
    idx2title: dict[int, str],
    idx2genre: dict[int, str],
) -> list[dict]:
    """Convert (movie_idx, score) pairs to display-ready dicts."""
    return [
        {
            "rank": i + 1,
            "movie_idx": idx,
            "title": idx2title.get(idx, f"Movie {idx}"),
            "genres": idx2genre.get(idx, ""),
            "score": float(score),
        }
        for i, (idx, score) in enumerate(ranked)
    ]
