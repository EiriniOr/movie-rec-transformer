"""
Model Comparison page.

Shows side-by-side Hit@10 and NDCG@10 metrics for both models, plus training
loss curves.  All data is read from artifacts/ — no model inference here.

If metrics.json doesn't exist yet, the page shows instructions for running evaluate.py.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.visualizations import (
    PAGE_CSS,
    COLORS,
    plot_loss_curves,
    plot_metrics_comparison,
)

ARTIFACTS_DIR = ROOT / "artifacts"

# ── page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Comparison · Movie Recommender", page_icon="📊", layout="wide"
)
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>📊 Model Comparison</h1>
  <p>
    Hit@10 and NDCG@10 evaluated on the held-out last-3-movies per user.
    Both models rank the full movie vocabulary; the test items must appear in the top 10 to count.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── load metrics ───────────────────────────────────────────────────────────────

metrics_path = ARTIFACTS_DIR / "metrics.json"

if not metrics_path.exists():
    st.warning("**metrics.json not found.**  Run evaluation first:")
    st.code("python scripts/evaluate.py", language="bash")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

mf = metrics.get("matrix_factorization", {})
tr = metrics.get("transformer", {})

# ── metric headline cards ──────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Evaluation Results</div>', unsafe_allow_html=True
)

# Four cards: Hit@10 for each model, NDCG@10 for each model
c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "🔵 MF  Hit@10", mf.get("hit@10", 0), COLORS["mf"]),
    (c2, "🟢 Transformer  Hit@10", tr.get("hit@10", 0), COLORS["transformer"]),
    (c3, "🔵 MF  NDCG@10", mf.get("ndcg@10", 0), COLORS["mf"]),
    (c4, "🟢 Transformer  NDCG@10", tr.get("ndcg@10", 0), COLORS["transformer"]),
]
for col, label, value, color in cards:
    with col:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value" style="color:{color}">{value:.4f}</div>'
            f'<div class="stat-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── bar chart ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Side-by-Side Comparison</div>', unsafe_allow_html=True
)

fig = plot_metrics_comparison(metrics)
st.plotly_chart(fig, use_container_width=True)

# ── interpretation ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    **Hit@10** answers: *"Did we recommend the right movie somewhere in our top 10?"*

    A Hit@10 of 0.15 means 15% of test (user, movie) pairs had the held-out movie
    appear in the model's top-10 ranked list.  Given the vocabulary is ~60,000 movies,
    this is significantly better than random chance (~0.017%).
    """)

with col_right:
    st.markdown("""
    **NDCG@10** (Normalised Discounted Cumulative Gain) additionally rewards *rank quality*.
    A recommendation at position 1 scores higher than the same recommendation at position 10.

    A higher NDCG means the model doesn't just find the right movie — it finds it faster.
    """)

# Winning model callout
if tr.get("hit@10", 0) >= mf.get("hit@10", 0):
    winner = "Transformer"
    delta_hit = tr["hit@10"] - mf["hit@10"]
    delta_ndcg = tr["ndcg@10"] - mf["ndcg@10"]
    msg = (
        f"The **Transformer** outperforms Matrix Factorization by "
        f"**+{delta_hit:.4f} Hit@10** and **+{delta_ndcg:.4f} NDCG@10**, "
        f"showing that modelling the order of watch history matters."
    )
else:
    winner = "Matrix Factorization"
    delta_hit = mf["hit@10"] - tr["hit@10"]
    delta_ndcg = mf["ndcg@10"] - tr["ndcg@10"]
    msg = (
        f"**Matrix Factorization** edges out the Transformer by "
        f"**+{delta_hit:.4f} Hit@10** and **+{delta_ndcg:.4f} NDCG@10**. "
        f"This can happen if the dataset lacks strong sequential patterns or if "
        f"the Transformer needs more training epochs / larger embedding dim."
    )

st.markdown(f'<div class="callout">{msg}</div>', unsafe_allow_html=True)

# ── training loss curves ───────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Training Loss Curves</div>', unsafe_allow_html=True
)

loss_fig = plot_loss_curves()
if loss_fig:
    st.plotly_chart(loss_fig, use_container_width=True)
    st.caption(
        "MF uses MSE loss on implicit-feedback (user, movie) pairs. "
        "Transformer uses cross-entropy on next-item prediction from sliding windows of length 10."
    )
else:
    st.info("Loss curves not found in artifacts/. Train both models to generate them.")

# ── raw numbers ───────────────────────────────────────────────────────────────

with st.expander("Raw numbers"):
    col_mf_raw, col_tr_raw = st.columns(2)
    with col_mf_raw:
        st.markdown("**Matrix Factorization**")
        st.json(mf)
    with col_tr_raw:
        st.markdown("**Transformer**")
        st.json(tr)
