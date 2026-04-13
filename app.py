"""
Competitive Intelligence RAG — Streamlit UI
Run with: streamlit run app.py
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

st.set_page_config(
    page_title="Competitive Intel RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tighten default Streamlit top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Hero gradient banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 28px 36px 20px 36px;
        margin-bottom: 20px;
        color: white;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 4px 0;
    }
    .hero-sub {
        font-size: 0.95rem;
        opacity: 0.75;
        margin: 0;
    }

    /* Pipeline step pills */
    .pipeline-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 16px;
    }
    .pipeline-step {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.78rem;
        color: white;
        white-space: nowrap;
    }
    .pipeline-arrow {
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        align-self: center;
    }

    /* Stat pills in results */
    .stat-pill {
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .stat-pill .num  { font-size: 2rem; font-weight: 800; line-height: 1; }
    .stat-pill .lbl  { font-size: 0.8rem; margin-top: 4px; opacity: 0.85; }

    /* Answer card */
    .answer-card {
        background: #f8faff;
        border-left: 4px solid #0f3460;
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        font-size: 0.97rem;
        line-height: 1.65;
        color: #1a1a2e;
    }

    /* Review card */
    .review-card {
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .review-meta {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .conf-badge {
        background: #e8e8e8;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.7rem;
        color: #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Lazy module loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_modules():
    from scraper import scrape_reviews
    from embedder import embed_reviews, embed_query
    from retriever import store_chunks, retrieve, collection_count, delete_collection
    from llm import ask_llm
    from sentiment import analyze_sentiment

    return {
        "scrape_reviews": scrape_reviews,
        "embed_reviews": embed_reviews,
        "embed_query": embed_query,
        "store_chunks": store_chunks,
        "retrieve": retrieve,
        "collection_count": collection_count,
        "delete_collection": delete_collection,
        "ask_llm": ask_llm,
        "analyze_sentiment": analyze_sentiment,
    }


# ── Component helpers ──────────────────────────────────────────────────────────

def _hero():
    """Render the hero banner + pipeline steps."""
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-title">🔍 Competitive Intel RAG</p>
            <p class="hero-sub">
                Ask any question about competitor app reviews — scrapes App Store &amp; Google Play,
                embeds locally, answers with Llama 3.
            </p>
            <div class="pipeline-row">
                <span class="pipeline-step">🌐 Scrape reviews</span>
                <span class="pipeline-arrow">›</span>
                <span class="pipeline-step">🧠 Embed (ONNX)</span>
                <span class="pipeline-arrow">›</span>
                <span class="pipeline-step">🗄️ Store in ChromaDB</span>
                <span class="pipeline-arrow">›</span>
                <span class="pipeline-step">🔎 Semantic search</span>
                <span class="pipeline-arrow">›</span>
                <span class="pipeline-step">💬 Llama 3 answer</span>
                <span class="pipeline-arrow">›</span>
                <span class="pipeline-step">📊 Sentiment pie</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sentiment_pie(sentiment: dict):
    """Render a Plotly donut chart for the sentiment breakdown."""
    import plotly.graph_objects as go

    c = sentiment["counts"]
    labels = ["Positive", "Negative", "Neutral", "Uncertain"]
    values = [c["positive"], c["negative"], c["neutral"], c["uncertain"]]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#f39c12"]

    # Drop zero slices so the chart isn't cluttered
    filtered = [(l, v, col) for l, v, col in zip(labels, values, colors) if v > 0]
    if not filtered:
        st.caption("No sentiment data.")
        return
    fl, fv, fc = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels=fl,
        values=fv,
        hole=0.55,
        marker=dict(colors=list(fc), line=dict(color="#ffffff", width=2)),
        textinfo="percent+label",
        textfont=dict(size=13),
        hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{sentiment['total']}</b><br>reviews",
            x=0.5, y=0.5,
            font=dict(size=14, color="#333"),
            showarrow=False,
        )],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _stat_pill(value, label: str, bg: str, text: str = "white"):
    st.markdown(
        f"""<div class="stat-pill" style="background:{bg};color:{text};">
                <div class="num">{value}</div>
                <div class="lbl">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def _review_card(chunk: dict, label: str, confidence: float | None = None):
    meta = chunk.get("metadata", {})
    source = meta.get("source", "unknown").replace("_", " ").title()
    rating = meta.get("rating")
    date = (meta.get("date") or "")[:10]

    bg_map = {
        "positive": "#f0faf4",
        "negative": "#fff5f5",
        "neutral":  "#f5f5f5",
        "uncertain": "#fffbf0",
    }
    border_map = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral":  "#aaa",
        "uncertain": "#f39c12",
    }
    bg = bg_map.get(label, "#fafafa")
    border = border_map.get(label, "#ccc")
    stars = "⭐" * int(rating) if rating else ""
    conf_html = (
        f'<span class="conf-badge">conf {confidence:.2f}</span>'
        if confidence is not None else ""
    )
    date_html = f'<span>{date}</span>' if date else ""

    st.markdown(
        f"""<div class="review-card" style="background:{bg};border-left:3px solid {border};">
            <div class="review-meta">
                <strong>{source}</strong>{date_html}{stars}{conf_html}
            </div>
            <div>{chunk['text']}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar (settings, collapsed by default) ───────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Reviews to retrieve (top-K)", 5, 30, 15)
        reviews_per_store = st.slider(
            "Max reviews per store", 50, 500, 200, step=50,
            help="Scraped once then cached. Higher = richer context, slower first run."
        )
        st.divider()
        st.subheader("🎯 Sentiment Confidence")
        confidence_threshold = st.slider(
            "Confidence threshold", 0.3, 0.9, 0.6, step=0.05,
            help=(
                "Reviews below this confidence are labelled Uncertain. "
                "Rated reviews: 1/5-star = 1.0, 3-star = 0.5. "
                "Text reviews: VADER compound magnitude."
            ),
        )
        st.divider()
        st.subheader("♻️ Cache")
        rescrape = st.checkbox(
            "Force re-scrape",
            value=False,
            help="Clear cached reviews and re-scrape from both stores.",
        )
    return top_k, reviews_per_store, confidence_threshold, rescrape


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error(
            "**OPENROUTER_API_KEY not found.**  \n"
            "Create a `.env` file in this directory:  \n"
            "```\nOPENROUTER_API_KEY=your_key_here\n```\n"
            "Get a free key at https://openrouter.ai"
        )
        st.stop()

    mods = _load_modules()
    top_k, reviews_per_store, confidence_threshold, rescrape = _sidebar()

    # ── Hero + query form (always visible above the fold) ──────────────────────
    _hero()

    with st.form("query_form"):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            app_name = st.text_input(
                "App name *",
                placeholder="e.g. Wise, Revolut, Monzo",
                label_visibility="collapsed",
            )
        with c2:
            feature_area = st.text_input(
                "Feature area",
                placeholder="Feature area (optional) e.g. onboarding",
                label_visibility="collapsed",
            )
        with c3:
            submitted = st.form_submit_button(
                "🚀 Analyse", use_container_width=True, type="primary"
            )

        question = st.text_area(
            "Your question *",
            placeholder="e.g.  What are users complaining about?  |  What do users love?  |  Any onboarding issues?",
            height=68,
            label_visibility="collapsed",
        )

    # ── Landing state — shown before first submission ───────────────────────────
    if not submitted:
        st.markdown("<br>", unsafe_allow_html=True)
        eg1, eg2, eg3 = st.columns(3)
        with eg1:
            st.markdown(
                """<div style='background:#f0f7ff;border-radius:10px;padding:16px;border:1px solid #c8deff'>
                <div style='font-size:1.3rem'>💸</div>
                <div style='font-weight:700;margin:6px 0 4px'>Try: Wise</div>
                <div style='font-size:0.82rem;color:#555'>"What are users complaining about with international transfers?"</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with eg2:
            st.markdown(
                """<div style='background:#f0fff4;border-radius:10px;padding:16px;border:1px solid #b2ecc8'>
                <div style='font-size:1.3rem'>🏦</div>
                <div style='font-weight:700;margin:6px 0 4px'>Try: Revolut</div>
                <div style='font-size:0.82rem;color:#555'>"What do users love about the app?"</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with eg3:
            st.markdown(
                """<div style='background:#fff8f0;border-radius:10px;padding:16px;border:1px solid #ffd8a8'>
                <div style='font-size:1.3rem'>🔔</div>
                <div style='font-weight:700;margin:6px 0 4px'>Try: Monzo</div>
                <div style='font-size:0.82rem;color:#555'>"Are there any complaints about notifications?"</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<p style='text-align:center;color:#aaa;font-size:0.8rem;margin-top:18px'>"
            "Reviews are cached after the first scrape — subsequent queries are instant. "
            "Open the sidebar (top-left) to adjust settings.</p>",
            unsafe_allow_html=True,
        )
        return

    # ── Validation ─────────────────────────────────────────────────────────────
    if not app_name.strip():
        st.error("App name is required.")
        return
    if not question.strip():
        st.error("Please enter a question.")
        return

    app_name = app_name.strip()
    feature_area_val = feature_area.strip() or None
    question = question.strip()

    # ── Pipeline ───────────────────────────────────────────────────────────────
    progress = st.empty()

    with st.spinner(""):
        cached_count = mods["collection_count"](app_name)
        if rescrape and cached_count > 0:
            mods["delete_collection"](app_name)
            cached_count = 0

        if cached_count == 0:
            progress.info(f"Scraping App Store & Play Store for **{app_name}**…  (~30–60 s)")
            reviews = mods["scrape_reviews"](
                app_name=app_name,
                count_per_store=reviews_per_store,
                feature_area=None,
            )
            if not reviews:
                st.error(
                    f"No reviews found for **{app_name}**. "
                    "Check spelling or try a different app name."
                )
                return

            progress.info(f"Embedding {len(reviews)} reviews with local ONNX model…")
            for r in reviews:
                r["app_name"] = app_name
            chunks = mods["embed_reviews"](reviews)

            progress.info(f"Storing {len(chunks)} chunks in ChromaDB…")
            mods["store_chunks"](app_name, chunks)
            progress.empty()
            scrape_msg = f"Scraped & indexed **{len(reviews)}** reviews ({len(chunks)} chunks)."
        else:
            scrape_msg = f"Using **{cached_count}** cached chunks for **{app_name}**."
            progress.empty()

        query_emb = mods["embed_query"](question)
        retrieved = mods["retrieve"](
            app_name=app_name,
            query_embedding=query_emb,
            top_k=top_k,
            feature_area=feature_area_val,
        )

        if not retrieved:
            fa_note = f" mentioning '{feature_area_val}'" if feature_area_val else ""
            st.warning(f"No relevant reviews found{fa_note}. Try a broader question.")
            return

        sentiment = mods["analyze_sentiment"](retrieved, threshold=confidence_threshold)
        answer = mods["ask_llm"](
            question=question,
            app_name=app_name,
            retrieved_chunks=retrieved,
            feature_area=feature_area_val,
            sentiment_summary=sentiment,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS — above-the-fold panel (answer + pie chart + stat pills)
    # ══════════════════════════════════════════════════════════════════════════
    st.success(scrape_msg)

    ans_col, chart_col = st.columns([3, 2], gap="large")

    with ans_col:
        fa_label = f" · feature: *{feature_area_val}*" if feature_area_val else ""
        st.markdown(
            f"<p style='font-size:0.78rem;color:#888;margin-bottom:6px'>"
            f"<b>{app_name}</b>{fa_label} &nbsp;·&nbsp; {len(retrieved)} reviews retrieved</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="answer-card">{answer}</div>',
            unsafe_allow_html=True,
        )

        # Stat pills row
        c = sentiment["counts"]
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            _stat_pill(c["positive"],  "Positive",  "#2ecc71")
        with p2:
            _stat_pill(c["negative"],  "Negative",  "#e74c3c")
        with p3:
            _stat_pill(c["neutral"],   "Neutral",   "#95a5a6")
        with p4:
            _stat_pill(c["uncertain"], "Uncertain", "#f39c12")

    with chart_col:
        st.markdown(
            "<p style='font-size:0.82rem;font-weight:600;color:#444;"
            "text-align:center;margin-bottom:0'>Sentiment Breakdown</p>",
            unsafe_allow_html=True,
        )
        _sentiment_pie(sentiment)
        st.caption(
            f"Confidence threshold: **{sentiment['threshold']:.2f}**  ·  "
            f"{c['uncertain']} review(s) labelled uncertain",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # BELOW THE FOLD — detailed reviews (scroll down to see)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#bbb;font-size:0.8rem'>"
        "▼ &nbsp; Scroll down for detailed review breakdown &nbsp; ▼</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    labels = sentiment["labels"]
    confidences = sentiment["confidences"]

    st.subheader("📝 Supporting Reviews")
    tabs = st.tabs(["All", "Positive", "Negative", "Neutral", "Uncertain"])

    def _show_reviews(items):
        if not items:
            st.caption("No reviews in this category.")
            return
        for chunk, lbl, conf in items:
            _review_card(chunk, lbl, confidence=conf)

    all_items = list(zip(retrieved, labels, confidences))

    with tabs[0]:
        _show_reviews(all_items[:10])
    with tabs[1]:
        _show_reviews([(c, l, cf) for c, l, cf in all_items if l == "positive"][:6])
    with tabs[2]:
        _show_reviews([(c, l, cf) for c, l, cf in all_items if l == "negative"][:6])
    with tabs[3]:
        _show_reviews([(c, l, cf) for c, l, cf in all_items if l == "neutral"][:6])
    with tabs[4]:
        unc = [(c, l, cf) for c, l, cf in all_items if l == "uncertain"]
        if unc:
            st.caption(
                f"{len(unc)} review(s) had confidence below {sentiment['threshold']:.2f} "
                "and were excluded from the pie chart counts."
            )
        _show_reviews(unc[:6])

    with st.expander("🔎 Debug: raw retrieved chunks"):
        for i, (chunk, lbl, conf) in enumerate(all_items, 1):
            st.markdown(
                f"**Chunk {i}** &nbsp;|&nbsp; distance: `{chunk['distance']:.4f}` "
                f"&nbsp;|&nbsp; label: `{lbl}` &nbsp;|&nbsp; confidence: `{conf:.3f}`"
            )
            st.json({**chunk, "embedding": "[hidden]"}, expanded=False)


if __name__ == "__main__":
    main()
