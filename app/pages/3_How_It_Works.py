"""
How It Works page — non-technical explainer.

Three sections:
  1. Embeddings  — what they are, why they work (word/movie analogy)
  2. Attention   — how the model decides which past movies matter most
  3. Why order matters — static vs dynamic preferences

All visualisations are synthetic (illustrative), not derived from live model weights.
The goal is intuition, not precision.
"""

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.visualizations import PAGE_CSS, mermaid_html, plot_attention_heatmap

# ── page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="How It Works · Movie Recommender", page_icon="🧠", layout="wide"
)
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>🧠 How It Works</h1>
  <p>
    A plain-English explanation of the three core ideas behind this system:
    <strong>embeddings</strong>, <strong>attention</strong>, and
    <strong>why the order of movies matters</strong>.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">1 · Embeddings — Teaching a Computer What a Movie "Means"</div>',
    unsafe_allow_html=True,
)

col_text, col_diagram = st.columns([1.1, 1], gap="large")

with col_text:
    st.markdown("""
    A computer can't understand the *idea* of a movie the way a person does.
    What it *can* do is learn to represent each movie as a list of numbers —
    called an **embedding vector**.

    The magic is that similar movies end up with *similar numbers*.
    After training on millions of ratings, the system learns on its own that:

    - **Toy Story**, **Shrek**, and **Finding Nemo** land close together
      (family animations)
    - **The Dark Knight**, **Se7en**, and **Joker** cluster elsewhere
      (dark, psychological)
    - **The Godfather** and **Goodfellas** share a neighbourhood
      (crime dramas)

    Nobody told the model these categories — it discovered them from patterns
    in what real people watch together.

    #### The analogy: latitude and longitude

    Just as GPS coordinates place two cities close together on a map if
    they are physically near, embedding coordinates place two movies close
    together in "taste space" if people who like one tend to like the other.

    In this system, each movie gets a **128-number** coordinate.
    128 dimensions is enough to capture subtle differences — genre, tone,
    era, pace — simultaneously.
    """)

with col_diagram:
    # Simplified 2D scatter showing genre clusters (illustrative)
    import plotly.graph_objects as go

    # Genre cluster centroids — illustrative positions in a fictional 2D embedding space
    clusters = {
        "Family Animation": (
            {
                "Toy Story": (0.8, 0.9),
                "Shrek": (0.75, 0.85),
                "Finding Nemo": (0.85, 0.8),
            },
            "#56CFB2",
        ),
        "Dark Thriller": (
            {
                "Dark Knight": (-0.7, 0.6),
                "Se7en": (-0.75, 0.55),
                "Joker": (-0.65, 0.65),
            },
            "#F6A623",
        ),
        "Crime Drama": (
            {
                "Godfather": (-0.6, -0.7),
                "Goodfellas": (-0.55, -0.75),
                "Scarface": (-0.65, -0.65),
            },
            "#7B8CDE",
        ),
        "Sci-Fi": (
            {
                "Interstellar": (0.5, -0.6),
                "The Matrix": (0.55, -0.55),
                "Arrival": (0.45, -0.65),
            },
            "#E07BE0",
        ),
    }

    fig_embed = go.Figure()
    for genre, (movies, color) in clusters.items():
        xs = [pos[0] for pos in movies.values()]
        ys = [pos[1] for pos in movies.values()]
        labels = list(movies.keys())
        fig_embed.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=genre,
                text=labels,
                textposition="top center",
                marker=dict(size=14, color=color, opacity=0.85),
                textfont=dict(size=10, color=color),
            )
        )

    fig_embed.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(
            title="← Serious · Playful →",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            title="← Dark · Light →",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        legend=dict(orientation="h", y=-0.12),
        height=380,
        margin=dict(l=10, r=10, t=10, b=40),
        title=dict(text="Illustrative 2D Embedding Space", font=dict(size=13), x=0.5),
    )
    st.plotly_chart(fig_embed, use_container_width=True)
    st.caption("Illustrative only — real embeddings have 128 dimensions.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div class="section-header">2 · Attention — Which Past Movies Matter Most?</div>',
    unsafe_allow_html=True,
)

col_text2, col_heatmap = st.columns([1, 1.2], gap="large")

with col_text2:
    st.markdown("""
    Embeddings tell the model *what* each movie is.
    **Attention** tells it *which past movies to pay attention to* when predicting the next one.

    #### How it works

    Imagine your recent watch history is:

    > **Toy Story → Shrek → The Matrix → Inception → ❓**

    To predict the next film, the model asks:
    *"Which of the previous movies is most relevant to what I should recommend now?"*

    In this example, the model might decide:
    - **Inception** (the most recent film) gets the highest weight — it's most predictive of current mood
    - **The Matrix** also gets weight — both are mind-bending sci-fi
    - **Shrek** and **Toy Story** get lower weights — less relevant for a sci-fi streak

    The heatmap on the right visualises this: **brighter = more attention**.
    Each row is a prediction; each column is a movie being attended to.
    The lower-left triangle is empty because the model can't "see the future"
    — it only attends to movies it has already seen.

    #### Why multiple attention heads?

    The model runs this process **twice in parallel** (2 heads).
    Each head can discover a different pattern:
    - Head 1 might focus on *genre similarity*
    - Head 2 might focus on *recency*

    Together, they capture richer context than either alone.
    """)

with col_heatmap:
    # Synthetic causal attention matrix for illustration
    movies_short = ["Toy\nStory", "Shrek", "Matrix", "Inception"]
    n = len(movies_short)

    # Plausible attention weights (causal — upper triangle is 0)
    attn = [
        [1.00, 0.00, 0.00, 0.00],  # predicting after Toy Story — only self
        [0.35, 0.65, 0.00, 0.00],  # predicting after Shrek
        [0.10, 0.15, 0.75, 0.00],  # predicting after Matrix — Matrix dominates
        [0.08, 0.12, 0.35, 0.45],  # predicting after Inception
    ]

    fig_attn = plot_attention_heatmap(movies_short, attn)
    st.plotly_chart(fig_attn, use_container_width=True)
    st.caption(
        "Synthetic example for illustration.  "
        "Brighter = more attention.  The upper-right triangle is masked (future items)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WHY ORDER MATTERS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div class="section-header">3 · Why Order Matters — Shifting Preferences</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("""
    #### The problem with treating all ratings equally

    Matrix Factorization adds up everything you've ever watched and builds a
    single "taste profile" for you.

    If you watched 50 action movies 5 years ago and 10 slow indie films this month,
    your profile still leans heavily **action** — even though your current taste
    has clearly changed.

    The result: recommendations that reflect *who you were*, not *who you are now*.
    """)

    st.markdown("""
    #### How the Transformer handles this

    The Transformer only looks at your **most recent window of up to 10 movies**.
    It doesn't average — it attends, giving more weight to recent history.

    If your last 10 films are all quiet dramas, the model picks up on that
    pattern and recommends another quiet drama — even if your all-time
    favourite genre is something different.

    This is the core idea behind **session-based recommendation**: preferences
    are dynamic and context-dependent.
    """)

with col_b:
    # Timeline diagram showing taste shift
    diagram = """
    timeline
        title A User's Evolving Taste
        section 2019–2020
            Action Phase : Die Hard
                        : Mad Max
                        : John Wick
        section 2021
            Transition  : Parasite
                        : Knives Out
        section 2022–2023
            Quiet Drama  : Marriage Story
                        : Past Lives
                        : Aftersun
    """
    # Mermaid timeline — fallback to a plotly figure if rendering fails
    timeline_html = mermaid_html(diagram, height=340)
    components.html(timeline_html, height=340)

    st.markdown(
        '<div class="callout">'
        "MF would still recommend action movies in 2023 because the 2019–2020 ratings "
        "outnumber the recent drama ratings.  The Transformer recommends dramas because "
        "it focuses on the most recent context window."
        "</div>",
        unsafe_allow_html=True,
    )

# ── summary ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div class="section-header">Putting It All Together</div>', unsafe_allow_html=True
)

steps_diagram = """
flowchart LR
    A["👤 Watch History\\n[ordered sequence]"]
    B["🎬 Movie Embeddings\\n128-dim vectors"]
    C["📍 Position Embeddings\\n'this was 3rd'"]
    D["🔍 Self-Attention\\n'which past matters?'"]
    E["🎯 Top-10 Recs\\n'what comes next?'"]

    A --> B
    B --> C
    C --> D
    D --> E

    style A fill:#1a1a3a,stroke:#7B8CDE
    style B fill:#2D2B6B,stroke:#7B8CDE
    style C fill:#2D2B6B,stroke:#7B8CDE
    style D fill:#1a1a3a,stroke:#56CFB2,stroke-width:2px
    style E fill:#2D2B6B,stroke:#F6A623
"""
components.html(mermaid_html(steps_diagram, height=200), height=200)

st.markdown("""
| Concept | Simple version |
|---|---|
| Embedding | Every movie gets a 128-number "fingerprint" based on who watches it |
| Attention | The model learns which past movies are most useful for predicting the next one |
| Order | Recent movies get more influence than old ones; current mood beats all-time average |
| Training | The model practises on millions of real sequences, predicting the next movie given everything before it |
""")

# ── possible next step ─────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div class="section-header">Possible Next Step</div>', unsafe_allow_html=True
)

st.markdown(
    """
<div class="callout">
<strong>Fine-tuning a pretrained language model (e.g. T5) for recommendation</strong><br><br>

The system described here learns movie representations purely from co-occurrence patterns
in watch histories — it has no knowledge of what a movie <em>is</em>.  It knows that users
who watch <em>Inception</em> often also watch <em>Interstellar</em>, but it doesn't know
<em>why</em>: both are directed by Christopher Nolan, both are cerebral sci-fi that play
with time and perception, and both share a visual and tonal style.<br><br>

A pretrained language model like <strong>T5</strong> or <strong>BERT</strong> already
holds this kind of world knowledge from training on text — reviews, plot summaries, interviews,
Wikipedia articles.  Fine-tuning it for recommendation would let the model combine two signals:
<ul>
  <li><strong>Behavioural signal</strong> — the sequential patterns from watch histories
      (what this system already captures)</li>
  <li><strong>Semantic signal</strong> — the meaning behind the movies: genre, director,
      themes, tone, era, cultural context</li>
</ul>

This means a user who just watched <em>Inception</em> and <em>Interstellar</em> back to back
could receive a recommendation for <em>Memento</em> — not just because other users make that
transition, but because the model understands all three are Nolan films built around fractured
time and unreliable reality.<br><br>

The approach would be to represent each movie as a short text (title + genre + a brief plot
description), encode it with the language model's encoder, and replace the learned movie
embedding table in this Transformer with those richer, pretrained representations.  The rest
of the architecture — positional embeddings, attention layers, next-item prediction head —
stays the same.
</div>
""",
    unsafe_allow_html=True,
)
