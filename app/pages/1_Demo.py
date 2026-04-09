"""
Live Demo page.

Visitors pick movies from a curated pill grid (or use the search fallback)
to build an ordered watch history, then hit "Get Recommendations" to see the
Transformer's top-10 predictions.  An optional expander compares with MF.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.inference import (
    find_closest_training_user,
    format_recommendations,
    load_resources,
    recommend_mf,
    recommend_transformer,
    search_movies,
)
from app.utils.visualizations import PAGE_CSS, history_item_html, rec_card_html

# ── page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Demo · Movie Recommender", page_icon="🎬", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Extra CSS: pill-style buttons for the curated grid
st.markdown(
    """
<style>
/* Make buttons in the pill grid look like rounded tags */
div.pill-row div[data-testid="column"] button {
    border-radius: 999px !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.78rem !important;
    background: rgba(45,43,107,0.45) !important;
    border: 1px solid rgba(123,140,222,0.3) !important;
    color: #c0c8f0 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
div.pill-row div[data-testid="column"] button:hover {
    background: rgba(86,207,178,0.15) !important;
    border-color: #56CFB2 !important;
    color: #56CFB2 !important;
}
/* Selected pill (movie already in history) */
div.pill-row div[data-testid="column"] button[kind="primary"] {
    background: rgba(86,207,178,0.2) !important;
    border-color: #56CFB2 !important;
    color: #56CFB2 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

res = load_resources()

if not res.get("ready"):
    st.error("Models not loaded. Run the setup scripts and refresh.")
    st.stop()

if res.get("transformer_model") is None:
    st.error("Transformer model not found. Run `python scripts/train_transformer.py`.")
    st.stop()

idx2title: dict = res["idx2title"]
idx2genre: dict = res["idx2genre"]

# ── curated movie list ─────────────────────────────────────────────────────────

# Well-known titles used as search queries against the real MovieLens vocabulary.
# Resolved once at load time to actual (movie_idx, full_title) pairs.
CURATED_QUERIES = [
    "Toy Story",
    "The Dark Knight",
    "Inception",
    "Interstellar",
    "The Matrix",
    "Pulp Fiction",
    "The Shawshank Redemption",
    "Forrest Gump",
    "Jurassic Park",
    "Goodfellas",
    "Fight Club",
    "The Lion King",
    "Gladiator",
    "The Godfather",
    "Se7en",
    "Titanic",
    "The Silence of the Lambs",
    "Schindler's List",
    "Saving Private Ryan",
    "Eternal Sunshine",
]


@st.cache_data
def resolve_curated(_idx2title: dict) -> list[tuple[int, str]]:
    """
    Map each curated query to the best-matching (idx, full_title) in the dataset.
    Cached so the search only runs once per server process.
    """
    resolved = []
    for query in CURATED_QUERIES:
        hits = search_movies(query, _idx2title, n=1)
        if hits:
            resolved.append(hits[0])  # (idx, title)
    return resolved


curated_movies: list[tuple[int, str]] = resolve_curated(idx2title)

# ── session state ──────────────────────────────────────────────────────────────

if "watch_history" not in st.session_state:
    st.session_state.watch_history: list[tuple[int, str]] = []

# ── header ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>🎬 Live Demo</h1>
  <p>
    Pick a few movies you've enjoyed — in roughly the order you watched them.
    The Transformer predicts your top-10 next watch based on the
    <em>sequence</em>, not just the individual titles.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── layout ─────────────────────────────────────────────────────────────────────

col_input, col_output = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — movie selection
# ══════════════════════════════════════════════════════════════════════════════

with col_input:
    st.markdown(
        '<div class="section-header">1 · Pick movies</div>', unsafe_allow_html=True
    )
    st.caption("Click to add · order matters · add them newest last")

    # ── pill grid ─────────────────────────────────────────────────────────────
    history_idx_set = {idx for idx, _ in st.session_state.watch_history}

    # Wrap in a named div so our CSS selector targets only these buttons
    st.markdown('<div class="pill-row">', unsafe_allow_html=True)
    COLS = 4
    rows = [curated_movies[i : i + COLS] for i in range(0, len(curated_movies), COLS)]
    for row in rows:
        cols = st.columns(COLS)
        for col, (idx, title) in zip(cols, row):
            in_history = idx in history_idx_set
            # Trim long titles so they fit in a narrow pill
            label = ("✓ " if in_history else "") + (
                title[:22] + "…" if len(title) > 24 else title
            )
            # kind="primary" triggers the teal selected style via our CSS
            kind = "primary" if in_history else "secondary"
            if col.button(
                label, key=f"pill_{idx}", type=kind, use_container_width=True
            ):
                if not in_history:
                    st.session_state.watch_history.append((idx, title))
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── search fallback ───────────────────────────────────────────────────────
    with st.expander("Search for a different movie"):
        query = st.text_input(
            "Search movies",
            placeholder="e.g. Blade Runner, Parasite, Coco…",
            key="search_query",
            label_visibility="collapsed",
        )
        if query:
            results = search_movies(query, idx2title, n=20)
            if results:
                options = {title: idx for idx, title in results}
                chosen_title = st.selectbox(
                    "Select to add",
                    list(options.keys()),
                    label_visibility="collapsed",
                )
                if st.button("➕ Add", use_container_width=True):
                    chosen_idx = options[chosen_title]
                    if chosen_idx not in history_idx_set:
                        st.session_state.watch_history.append(
                            (chosen_idx, chosen_title)
                        )
                        st.rerun()
            else:
                st.info("No matches. Try a shorter title.")

    st.markdown("---")

    # ── watch history display ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">2 · Your watch history</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.watch_history:
        history_html = "".join(
            history_item_html(i + 1, title)
            for i, (_, title) in enumerate(st.session_state.watch_history)
        )
        st.markdown(history_html, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🗑 Remove last", use_container_width=True):
                st.session_state.watch_history.pop()
                st.rerun()
        with btn_col2:
            if st.button("✖ Clear all", use_container_width=True):
                st.session_state.watch_history = []
                st.rerun()

        st.markdown("---")

        run_recs = st.button(
            "🎯 Get Recommendations",
            type="primary",
            use_container_width=True,
            disabled=len(st.session_state.watch_history) < 2,
        )
        if len(st.session_state.watch_history) < 2:
            st.caption("Add at least 2 movies.")
    else:
        st.markdown(
            '<div class="callout">Click any pill above to start building your history.</div>',
            unsafe_allow_html=True,
        )
        run_recs = False

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — recommendations
# ══════════════════════════════════════════════════════════════════════════════

with col_output:
    st.markdown(
        '<div class="section-header">3 · Top-10 predictions</div>',
        unsafe_allow_html=True,
    )

    history_idxs = [idx for idx, _ in st.session_state.watch_history]

    if run_recs and history_idxs:
        # ── Transformer ───────────────────────────────────────────────────────
        with st.spinner("Running Transformer inference…"):
            tr_ranked = recommend_transformer(
                model=res["transformer_model"],
                history_idxs=history_idxs,
                top_k=10,
                device=res["device"],
            )
        tr_recs = format_recommendations(tr_ranked, idx2title, idx2genre)

        st.markdown("#### 🟢 Transformer")
        st.markdown(
            "".join(
                rec_card_html(r["rank"], r["title"], r["genres"], r["score"])
                for r in tr_recs
            ),
            unsafe_allow_html=True,
        )

        # ── MF comparison (optional) ──────────────────────────────────────────
        if res.get("mf_model") is not None:
            with st.expander("Compare with Matrix Factorization"):
                st.caption(
                    "MF is tied to training-set user profiles.  We find the user "
                    "whose history most overlaps with yours and borrow their embedding."
                )
                sequences_train = res.get(
                    "sequences_train"
                )  # may be None on Streamlit Cloud
                with st.spinner("Finding nearest training user…"):
                    raw_uid, mf_uid = find_closest_training_user(
                        history_idxs, sequences_train, res["mf_user2idx"]
                    )
                # Use the found user's full history as MF context, or fall back to input
                mf_history = (
                    sequences_train.get(raw_uid, history_idxs)
                    if sequences_train
                    else history_idxs
                )
                with st.spinner("Running MF inference…"):
                    mf_ranked = recommend_mf(
                        model=res["mf_model"],
                        user_id=mf_uid,
                        history_idxs=mf_history,
                        top_k=10,
                        device=res["device"],
                    )
                mf_recs = format_recommendations(mf_ranked, idx2title, idx2genre)

                st.markdown("#### 🔵 Matrix Factorization")
                st.markdown(
                    "".join(
                        rec_card_html(r["rank"], r["title"], r["genres"], r["score"])
                        for r in mf_recs
                    ),
                    unsafe_allow_html=True,
                )

                overlap = len(
                    {r["movie_idx"] for r in tr_recs}
                    & {r["movie_idx"] for r in mf_recs}
                )
                st.metric("Titles in common (top-10)", f"{overlap} / 10")

    elif not st.session_state.watch_history:
        st.info("Pick movies on the left, then click **Get Recommendations**.")
    else:
        st.info("Click **Get Recommendations** when you're ready.")
