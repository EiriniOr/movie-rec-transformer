"""
Reusable chart and diagram helpers for the Streamlit app.

All Plotly figures use a consistent dark theme.  HTML/SVG components (e.g.
the Mermaid architecture diagram) are returned as raw strings for use with
st.components.v1.html().
"""

import json
from pathlib import Path

import plotly.graph_objects as go

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

# ── shared theme ───────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=20, r=20, t=40, b=20),
)

COLORS = {
    "mf": "#7B8CDE",  # muted blue for Matrix Factorization
    "transformer": "#56CFB2",  # teal for Transformer
    "accent": "#F6A623",  # amber for highlights
    "grid": "rgba(255,255,255,0.08)",
}


# ── metrics comparison ─────────────────────────────────────────────────────────


def plot_metrics_comparison(metrics: dict) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing Hit@10 and NDCG@10
    for Matrix Factorization vs Transformer.

    Args:
        metrics: dict loaded from artifacts/metrics.json

    Returns:
        Plotly Figure
    """
    mf = metrics.get("matrix_factorization", {})
    tr = metrics.get("transformer", {})

    metric_names = ["Hit@10", "NDCG@10"]
    mf_values = [mf.get("hit@10", 0), mf.get("ndcg@10", 0)]
    tr_values = [tr.get("hit@10", 0), tr.get("ndcg@10", 0)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Matrix Factorization",
            x=metric_names,
            y=mf_values,
            marker_color=COLORS["mf"],
            text=[f"{v:.4f}" for v in mf_values],
            textposition="outside",
            textfont=dict(size=14, color=COLORS["mf"]),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Transformer",
            x=metric_names,
            y=tr_values,
            marker_color=COLORS["transformer"],
            text=[f"{v:.4f}" for v in tr_values],
            textposition="outside",
            textfont=dict(size=14, color=COLORS["transformer"]),
        )
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        yaxis=dict(
            title="Score",
            gridcolor=COLORS["grid"],
            range=[0, max(max(mf_values), max(tr_values)) * 1.3],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


# ── training loss curves ───────────────────────────────────────────────────────


def plot_loss_curves() -> go.Figure | None:
    """
    Line chart of training loss per epoch for both models.
    Returns None if neither loss file exists yet.
    """
    traces = []

    for filename, name, color in [
        ("mf_loss_curve.json", "Matrix Factorization", COLORS["mf"]),
        ("transformer_loss_curve.json", "Transformer", COLORS["transformer"]),
    ]:
        path = ARTIFACTS_DIR / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            losses = data.get("epoch_losses", [])
            epochs = list(range(1, len(losses) + 1))
            traces.append(
                go.Scatter(
                    x=epochs,
                    y=losses,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=8),
                )
            )

    if not traces:
        return None

    fig = go.Figure(traces)
    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(title="Epoch", dtick=1, gridcolor=COLORS["grid"]),
        yaxis=dict(title="Loss", gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )
    return fig


# ── recommendation score bars ─────────────────────────────────────────────────


def plot_recommendation_scores(recs: list[dict], model_name: str) -> go.Figure:
    """
    Horizontal bar chart of top-K recommendation scores.

    Args:
        recs:       list of format_recommendations() dicts
        model_name: display name for the chart title

    Returns:
        Plotly Figure
    """
    titles = [f"#{r['rank']}  {r['title'][:45]}" for r in recs]
    scores = [r["score"] for r in recs]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=titles,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[[0, COLORS["mf"]], [1, COLORS["transformer"]]],
                showscale=False,
            ),
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
            textfont=dict(size=11),
        )
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        xaxis=dict(title="Score", gridcolor=COLORS["grid"]),
        height=max(300, 40 * len(recs)),
        title=dict(text=model_name, font=dict(size=15)),
    )
    return fig


# ── attention heatmap (explainer) ──────────────────────────────────────────────


def plot_attention_heatmap(
    labels: list[str],
    attention_matrix: list[list[float]],
) -> go.Figure:
    """
    Render a synthetic attention weight heatmap for the explainer page.

    Rows = query positions (what we're predicting).
    Cols = key positions (what we attend to).
    The lower-left triangle represents valid (causal) attention.

    Args:
        labels:           sequence of short movie/genre labels
        attention_matrix: 2-D list of weights (already causal-masked)

    Returns:
        Plotly Figure
    """
    fig = go.Figure(
        go.Heatmap(
            z=attention_matrix,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "rgba(30,30,60,0.0)"],
                [0.3, "#2D2B6B"],
                [0.7, "#7B8CDE"],
                [1.0, "#56CFB2"],
            ],
            showscale=True,
            colorbar=dict(title="Attention", tickfont=dict(color="#e0e0e0")),
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(title="Attending to →", side="bottom", tickangle=-30),
        yaxis=dict(title="Predicting ↓", autorange="reversed"),
        height=420,
    )
    return fig


# ── architecture diagram (HTML/Mermaid) ────────────────────────────────────────


def mermaid_html(diagram: str, height: int = 300) -> str:
    """
    Wrap a Mermaid diagram definition in a self-contained HTML block that
    loads mermaid.js from the CDN and renders the SVG client-side.

    Usage:
        st.components.v1.html(mermaid_html("graph LR ..."), height=300)
    """
    return f"""
    <html>
    <head>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'dark',
          themeVariables: {{
            primaryColor: '#2D2B6B',
            primaryTextColor: '#e0e0e0',
            primaryBorderColor: '#7B8CDE',
            lineColor: '#56CFB2',
            secondaryColor: '#1a1a3a',
            tertiaryColor: '#0e0e1f'
          }}
        }});
      </script>
      <style>
        body {{ margin: 0; background: transparent; }}
        .mermaid {{ display: flex; justify-content: center; }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{diagram}
      </div>
    </body>
    </html>
    """


PIPELINE_DIAGRAM = """
flowchart LR
    A["📦 MovieLens 25M\\n(25M ratings)"] --> B["⚙️ Preprocess\\nSort by timestamp"]
    B --> C["📋 User Sequences\\n(ordered watch history)"]
    C --> D["🔵 Matrix\\nFactorization"]
    C --> E["🟢 Transformer\\nEncoder × 2"]
    D --> F["📊 Evaluate\\nHit@10 · NDCG@10"]
    E --> F
    F --> G["🖥️ Streamlit App"]

    style A fill:#1a1a3a,stroke:#7B8CDE
    style B fill:#1a1a3a,stroke:#7B8CDE
    style C fill:#2D2B6B,stroke:#7B8CDE
    style D fill:#1a1a3a,stroke:#7B8CDE
    style E fill:#1a1a3a,stroke:#56CFB2
    style F fill:#1a1a3a,stroke:#F6A623
    style G fill:#2D2B6B,stroke:#F6A623
"""

TRANSFORMER_DIAGRAM = """
flowchart LR
    A["[Toy Story\\nMatrix\\nShrek]"] --> B["🎬 Movie\\nEmbeddings"]
    B --> C["➕ Positional\\nEmbeddings"]
    C --> D["🔁 Transformer\\nEncoder × 2"]
    D --> E["📐 Linear\\nHead"]
    E --> F["🎯 Top-10\\nPredictions"]

    style A fill:#1a1a3a,stroke:#7B8CDE
    style B fill:#2D2B6B,stroke:#7B8CDE
    style C fill:#2D2B6B,stroke:#7B8CDE
    style D fill:#1a1a3a,stroke:#56CFB2,stroke-width:2px
    style E fill:#1a1a3a,stroke:#56CFB2
    style F fill:#2D2B6B,stroke:#F6A623
"""


# ── shared page CSS ────────────────────────────────────────────────────────────

PAGE_CSS = """
<style>
/* ── global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0e0e2e 0%, #1a1a4a 50%, #0e1e3e 100%);
    border: 1px solid rgba(123,140,222,0.25);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin: 0 0 0.5rem; }
.hero p  { font-size: 1.1rem; color: #a0a8d0; margin: 0; }

/* ── stat cards ── */
.stat-card {
    background: rgba(45,43,107,0.35);
    border: 1px solid rgba(123,140,222,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-card .stat-value { font-size: 2rem; font-weight: 700; color: #56CFB2; }
.stat-card .stat-label { font-size: 0.85rem; color: #8890b0; margin-top: 0.25rem; }

/* ── recommendation cards ── */
.rec-card {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: rgba(20,20,50,0.6);
    border: 1px solid rgba(123,140,222,0.15);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: rgba(86,207,178,0.4); }
.rank-badge {
    min-width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: linear-gradient(135deg, #2D2B6B, #4a487a);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: #c0c8f0;
    flex-shrink: 0;
}
.rec-title  { font-weight: 600; color: #e0e8ff; font-size: 0.95rem; }
.rec-genres { font-size: 0.78rem; color: #6878a8; margin-top: 0.15rem; }
.rec-score  { margin-left: auto; font-size: 0.8rem; color: #56CFB2; white-space: nowrap; }

/* ── genre pills ── */
.genre-pill {
    display: inline-block;
    background: rgba(123,140,222,0.15);
    border: 1px solid rgba(123,140,222,0.3);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font-size: 0.72rem;
    color: #9aa8d8;
    margin: 0.1rem 0.15rem 0.1rem 0;
}

/* ── section headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #c0c8f0;
    border-left: 3px solid #56CFB2;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* ── callout box ── */
.callout {
    background: rgba(86,207,178,0.08);
    border-left: 3px solid #56CFB2;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #a8d8d0;
    font-size: 0.92rem;
}

/* ── history item ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(30,30,70,0.5);
    border: 1px solid rgba(123,140,222,0.2);
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #c0c8f0;
}
.pos-badge {
    background: rgba(86,207,178,0.2);
    color: #56CFB2;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
"""


def rec_card_html(rank: int, title: str, genres: str, score: float) -> str:
    """Render a single recommendation as an HTML card string."""
    genre_pills = "".join(
        f'<span class="genre-pill">{g.strip()}</span>'
        for g in genres.split("·")
        if g.strip()
    )
    score_fmt = f"{score:.4f}" if score < 1.0 else f"{score:.2f}"
    return f"""
    <div class="rec-card">
      <div class="rank-badge">{rank}</div>
      <div>
        <div class="rec-title">{title}</div>
        <div class="rec-genres">{genre_pills}</div>
      </div>
      <div class="rec-score">{score_fmt}</div>
    </div>
    """


def history_item_html(position: int, title: str) -> str:
    """Render a watch-history item as an HTML row string."""
    return f"""
    <div class="history-item">
      <span class="pos-badge">#{position}</span>
      {title}
    </div>
    """
