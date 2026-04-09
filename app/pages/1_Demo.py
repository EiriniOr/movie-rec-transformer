"""
Live Demo page.

Workflow:
  1. User searches for movies by title and adds them to an ordered watch history.
  2. The Transformer predicts the top-10 most likely next movies.
  3. An optional MF comparison is shown by finding the nearest-neighbor training
     user (the training user whose history has the most overlap with the input).

The Transformer is the star here — it works with any arbitrary input sequence,
unlike MF which is tied to known training-set users.
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

res = load_resources()

if not res.get("ready"):
    st.error("Models not loaded. Run the setup scripts and refresh.")
    st.stop()

if res.get("transformer_model") is None:
    st.error(
        "Transformer model not found in artifacts/. Run `python scripts/train_transformer.py`."
    )
    st.stop()

idx2title: dict = res["idx2title"]
idx2genre: dict = res["idx2genre"]
movie2idx: dict = res["movie2idx"]

# ── session state ──────────────────────────────────────────────────────────────

# Watch history is a list of (movie_idx, title) tuples stored in session state
# so it persists across reruns triggered by widget interactions.
if "watch_history" not in st.session_state:
    st.session_state.watch_history: list[tuple[int, str]] = []

# ── header ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>🎬 Live Demo</h1>
  <p>
    Build a watch history by searching for movies below.
    The Transformer will predict your top-10 next movies based on the
    <em>order</em> you watched them — not just which ones you liked.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── layout: two columns ────────────────────────────────────────────────────────

col_input, col_output = st.columns([1, 1.4], gap="large")

# ── LEFT: movie search + watch history ────────────────────────────────────────

with col_input:
    st.markdown(
        '<div class="section-header">Build Your Watch History</div>',
        unsafe_allow_html=True,
    )

    # Search box
    query = st.text_input(
        "Search movies",
        placeholder="e.g. Inception, Toy Story, The Dark Knight…",
        key="search_query",
        label_visibility="collapsed",
    )

    # Show search results as a selectbox
    if query:
        results = search_movies(query, idx2title, n=20)
        if results:
            options = {title: idx for idx, title in results}
            chosen_title = st.selectbox(
                "Select a movie to add",
                list(options.keys()),
                label_visibility="collapsed",
            )
            if st.button("➕ Add to watch history", use_container_width=True):
                chosen_idx = options[chosen_title]
                # Avoid exact duplicates (same movie twice in a row looks odd)
                if (
                    not st.session_state.watch_history
                    or st.session_state.watch_history[-1][0] != chosen_idx
                ):
                    st.session_state.watch_history.append((chosen_idx, chosen_title))
                    st.rerun()
        else:
            st.info("No movies matched. Try a different title.")

    st.markdown("---")

    # Display current watch history
    if st.session_state.watch_history:
        st.markdown(f"**Watch history** ({len(st.session_state.watch_history)} movies)")
        history_html = "".join(
            history_item_html(i + 1, title)
            for i, (_, title) in enumerate(st.session_state.watch_history)
        )
        st.markdown(history_html, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🗑️ Remove last", use_container_width=True):
                st.session_state.watch_history.pop()
                st.rerun()
        with btn_col2:
            if st.button("✖️ Clear all", use_container_width=True):
                st.session_state.watch_history = []
                st.rerun()

        st.markdown("---")

        # Recommend button
        run_recs = st.button(
            "🎯 Get Recommendations",
            type="primary",
            use_container_width=True,
            disabled=len(st.session_state.watch_history) < 2,
        )
        if len(st.session_state.watch_history) < 2:
            st.caption("Add at least 2 movies to get recommendations.")
    else:
        st.markdown(
            '<div class="callout">Search for a movie above and add it to start building '
            "your watch history.  Order matters — add them in the order you watched them.</div>",
            unsafe_allow_html=True,
        )
        run_recs = False

# ── RIGHT: recommendations output ─────────────────────────────────────────────

with col_output:
    st.markdown(
        '<div class="section-header">Top-10 Predictions</div>', unsafe_allow_html=True
    )

    history_idxs = [idx for idx, _ in st.session_state.watch_history]

    if "run_recs" in dir() and run_recs and history_idxs:
        # ── Transformer recommendations ──────────────────────────────────────
        with st.spinner("Running Transformer inference…"):
            tr_ranked = recommend_transformer(
                model=res["transformer_model"],
                history_idxs=history_idxs,
                top_k=10,
                device=res["device"],
            )
        tr_recs = format_recommendations(tr_ranked, idx2title, idx2genre)

        st.markdown("#### 🟢 Transformer")
        cards_html = "".join(
            rec_card_html(r["rank"], r["title"], r["genres"], r["score"])
            for r in tr_recs
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── Optional MF comparison ────────────────────────────────────────────
        if res.get("mf_model") is not None:
            with st.expander("Compare with Matrix Factorization"):
                st.markdown(
                    "MF only knows about users in its training set.  We find the "
                    "training user whose watch history has the most overlap with yours "
                    "and use their learned embedding."
                )
                with st.spinner("Finding nearest training user…"):
                    raw_uid, mf_uid = find_closest_training_user(
                        history_idxs,
                        res["sequences_train"],
                        res["mf_user2idx"],
                    )
                mf_history = res["sequences_train"].get(raw_uid, [])
                with st.spinner("Running MF inference…"):
                    mf_ranked = recommend_mf(
                        model=res["mf_model"],
                        user_id=mf_uid,
                        history_idxs=mf_history,
                        top_k=10,
                        device=res["device"],
                    )
                mf_recs = format_recommendations(mf_ranked, idx2title, idx2genre)

                st.markdown("#### 🔵 Matrix Factorization (nearest training user)")
                mf_cards = "".join(
                    rec_card_html(r["rank"], r["title"], r["genres"], r["score"])
                    for r in mf_recs
                )
                st.markdown(mf_cards, unsafe_allow_html=True)

                # Overlap between the two models' top-10
                tr_set = {r["movie_idx"] for r in tr_recs}
                mf_set = {r["movie_idx"] for r in mf_recs}
                overlap = len(tr_set & mf_set)
                st.metric("Titles in common (top-10)", f"{overlap} / 10")

    elif not st.session_state.watch_history:
        st.info(
            "Add movies to your watch history on the left, then click **Get Recommendations**."
        )
    else:
        st.info("Click **Get Recommendations** when your history is ready.")
