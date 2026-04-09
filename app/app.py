"""
Streamlit entry point — Overview page.

Run from the repo root:
  streamlit run app/app.py

This page shows:
  - Hero banner with project summary
  - Dataset statistics (loaded from data/processed/dataset_stats.json)
  - Full pipeline architecture diagram (Mermaid)
  - MF vs Transformer comparison table
  - Setup instructions if data hasn't been processed yet
"""

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.inference import load_resources
from app.utils.visualizations import (
    PAGE_CSS,
    PIPELINE_DIAGRAM,
    TRANSFORMER_DIAGRAM,
    mermaid_html,
)

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── load shared resources ──────────────────────────────────────────────────────

res = load_resources()

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown(
        "Sequence-based recommendation using Transformers vs Matrix Factorization."
    )
    st.divider()

    device_label = str(res.get("device", "cpu")).upper()
    status = "✅ Models loaded" if res.get("ready") else "⚠️ Run setup first"
    st.markdown(f"**Status:** {status}  \n**Device:** `{device_label}`")

    st.divider()
    st.markdown(
        "**Pages**\n"
        "- Overview ← you are here\n"
        "- Demo\n"
        "- Model Comparison\n"
        "- How It Works"
    )

# ── hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>🎬 Sequence-Based Movie Recommender</h1>
  <p>
    Comparing <strong>Matrix Factorization</strong> (order-agnostic baseline) with a
    <strong>Transformer encoder</strong> that models the temporal dynamics of a user's
    watch history — trained on 25 million ratings from MovieLens.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── setup warning if data not ready ───────────────────────────────────────────

if not res.get("ready"):
    st.warning("Data or models not found. Run the setup steps below to get started.")
    st.code(
        "# 1. Download dataset\n"
        "python scripts/download_data.py\n\n"
        "# 2. Preprocess sequences\n"
        "python scripts/preprocess.py\n\n"
        "# 3. Train both models\n"
        "python scripts/train_mf.py\n"
        "python scripts/train_transformer.py\n\n"
        "# 4. Evaluate\n"
        "python scripts/evaluate.py",
        language="bash",
    )
    st.stop()

# ── dataset stats ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)

stats_path = ROOT / "data" / "processed" / "dataset_stats.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

c1, c2, c3, c4, c5 = st.columns(5)
stat_items = [
    (c1, stats.get("n_ratings_total", "—"), "Total Ratings"),
    (c2, stats.get("n_users", "—"), "Users (≥20 ratings)"),
    (c3, stats.get("n_movies", "—"), "Unique Movies"),
    (c4, stats.get("avg_sequence_length", "—"), "Avg Sequence Length"),
    (c5, stats.get("test_items_per_user", "—"), "Test Items / User"),
]
for col, value, label in stat_items:
    with col:
        val_fmt = f"{value:,}" if isinstance(value, int) else str(value)
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{val_fmt}</div>'
            f'<div class="stat-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ── pipeline diagram ───────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True
)
components.html(mermaid_html(PIPELINE_DIAGRAM, height=260), height=260)

# ── model architecture ─────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Transformer Architecture</div>', unsafe_allow_html=True
)
components.html(mermaid_html(TRANSFORMER_DIAGRAM, height=220), height=220)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**What the Transformer learns**")
    st.markdown("""
    Each position in the sequence attends to all previous positions via
    **scaled dot-product attention**.  The model learns that a user who
    recently watched several thrillers should be recommended another thriller —
    even if their all-time favourite genre is comedy.
    """)

with col_right:
    st.markdown("**Key design choices**")
    st.markdown("""
    | Component | Value | Why |
    |---|---|---|
    | Embedding dim | 128 | Enough capacity, not overparameterised |
    | Layers | 2 | Captures 2nd-order context |
    | Heads | 2 | Two independent attention patterns |
    | Window | 50 | Covers typical recent watch history |
    | Causal mask | ✓ | Prevents attending to future items |
    """)

# ── model comparison summary ───────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Model Comparison</div>', unsafe_allow_html=True
)

col_mf, col_tr = st.columns(2)

with col_mf:
    st.markdown("""
    ### 🔵 Matrix Factorization
    **The baseline.**  Each user and movie gets a 64-dimensional vector.
    Recommendation = dot product of user × movie embeddings.

    - ✅ Fast to train
    - ✅ Works well for users with many ratings
    - ❌ Treats all past ratings equally — ignores *when* each movie was watched
    - ❌ No notion of "current mood" or evolving taste
    """)

with col_tr:
    st.markdown("""
    ### 🟢 Transformer Encoder
    **The sequential model.**  Given an ordered watch history, the model
    attends over past movies and predicts what comes next.

    - ✅ Captures temporal patterns ("just finished a trilogy")
    - ✅ Long-range context via multi-head attention
    - ✅ No user ID needed — works for any input sequence
    - ❌ More parameters, slower to train
    """)

st.markdown(
    '<div class="callout">See the <strong>Model Comparison</strong> page for '
    "Hit@10 and NDCG@10 metrics side-by-side.</div>",
    unsafe_allow_html=True,
)
