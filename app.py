"""
app.py — RAG Support Ticket Classifier (Hybrid Search)
Run with: streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Retrieval RAG",
    page_icon="🎫",
    layout="wide",
)

# ── Sample complaints ─────────────────────────────────────────────────────────
SAMPLE_COMPLAINTS = {
    "Select a sample...": "",
    "Payment deducted, order not placed": "I paid ₹1,200 via UPI but the order was never confirmed. My bank shows the money was debited.",
    "OTP not received": "I'm trying to checkout but the OTP for payment verification is not coming to my phone.",
    "Account locked": "I tried logging in too many times and now my account is locked. I can't access it at all.",
    "App keeps crashing": "Every time I open the app and go to the cart, it crashes. I'm on Android 13.",
    "Received damaged product": "The laptop I ordered arrived completely broken. The screen is cracked and the box was damaged.",
    "Refund not received": "It's been 10 days since I returned the item but the refund hasn't hit my account yet.",
    "Coupon not working": "The code SAVE20 is not applying at checkout even though it's valid till next month.",
    "Unauthorized transaction": "I see a transaction of ₹5,000 on my account that I never made. Someone may have hacked my account.",
    "Subscription not activated": "I paid for the premium plan yesterday but it still shows free plan.",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.confidence-high   { background:#d4edda; color:#155724; padding:4px 12px; border-radius:12px; font-weight:600; }
.confidence-medium { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:12px; font-weight:600; }
.confidence-low    { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:12px; font-weight:600; }
.score-bar-label   { font-size:0.8rem; color:#666; }
.step-card         { background:#f8f9fa; border-left:4px solid #0d6efd;
                     padding:10px 16px; margin:6px 0; border-radius:0 8px 8px 0; }
.metric-good  { background:#d4edda; color:#155724; padding:8px 16px; border-radius:8px; text-align:center; font-size:1.4rem; font-weight:700; }
.metric-ok    { background:#fff3cd; color:#856404; padding:8px 16px; border-radius:8px; text-align:center; font-size:1.4rem; font-weight:700; }
.metric-bad   { background:#f8d7da; color:#721c24; padding:8px 16px; border-radius:8px; text-align:center; font-size:1.4rem; font-weight:700; }
.metric-label { text-align:center; font-size:0.78rem; color:#555; margin-top:4px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key_input = st.text_input(
        "Gemini API Key",
        value="",
        type="password",
        placeholder="Paste your key here...",
        help="Set GEMINI_API_KEY in your .env file, or paste it here.",
    )
    # Use typed key first, silently fall back to .env — never expose the value in UI
    api_key = api_key_input.strip() or os.getenv("GEMINI_API_KEY", "")
    if api_key and not api_key_input:
        st.caption("✅ API key loaded from .env")

    st.divider()
    st.subheader("Retrieval Settings")

    top_k = st.slider(
        "Top-K (context entries)",
        min_value=1, max_value=10, value=3,
        help="Number of knowledge base entries to retrieve and pass to the LLM.",
    )

    alpha = st.slider(
        "α — Semantic weight",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Score = α × semantic + (1−α) × keyword. Higher α = more semantic, lower = more keyword.",
    )

    st.caption(f"Semantic: **{alpha:.0%}** | Keyword: **{1-alpha:.0%}**")

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This app uses **Hybrid RAG** — combining FAISS semantic search "
        "with TF-IDF keyword search — to classify support complaints and "
        "suggest resolution steps using Gemini 2.5 Flash."
    )


# ── Helper: get or build retriever ───────────────────────────────────────────
KB_PATH = "data/knowledge_base.csv"


def get_retriever(api_key: str):
    from src.retrieval import HybridRetriever
    import hashlib
    cache_key = "retriever_" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    if cache_key not in st.session_state:
        with st.spinner("🔧 Building search indexes (first run only)..."):
            st.session_state[cache_key] = HybridRetriever(KB_PATH, api_key)
    return st.session_state[cache_key]


def metric_color(value: float) -> str:
    if value >= 0.75:
        return "metric-good"
    elif value >= 0.5:
        return "metric-ok"
    return "metric-bad"


# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("🎫 Hybrid Retrieval RAG")
st.markdown("Classify customer complaints and get resolution steps using **Hybrid Search + LLM**.")

tab_classify, tab_eval = st.tabs(["🔍 Classify & Resolve", "📊 RAG Evaluation"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Classify & Resolve
# ════════════════════════════════════════════════════════════════════════════
with tab_classify:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        sample = st.selectbox("📋 Pick a sample complaint", options=list(SAMPLE_COMPLAINTS.keys()))
        complaint = st.text_area(
            "✏️ Customer Complaint",
            value=SAMPLE_COMPLAINTS[sample],
            height=140,
            placeholder="Type or paste a customer complaint here...",
        )
        classify_btn = st.button("🔍 Classify & Resolve", type="primary", use_container_width=True)

    with col2:
        st.markdown("**How scoring works**")
        st.markdown(f"""
| Method | Weight |
|--------|--------|
| 🧠 Semantic (FAISS + Embeddings) | **{alpha:.0%}** |
| 🔑 Keyword (TF-IDF Cosine) | **{1-alpha:.0%}** |

`hybrid_score = {alpha:.1f} × semantic + {1-alpha:.1f} × keyword`
""")

    st.divider()

    if classify_btn:
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar.")
            st.stop()
        if not complaint.strip():
            st.warning("Please enter a complaint to classify.")
            st.stop()

        from src.llm import LLMGenerator

        retriever = get_retriever(api_key)
        generator = LLMGenerator(api_key)

        with st.spinner("🔍 Retrieving relevant context..."):
            try:
                results = retriever.retrieve(complaint, top_k=top_k, alpha=alpha)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        with st.spinner("🤖 Generating classification with Gemini 2.5 Flash..."):
            try:
                output = generator.generate(complaint, results)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

        st.subheader("📊 Classification Result")

        confidence = output.get("confidence", "Low")
        conf_class = {
            "High": "confidence-high",
            "Medium": "confidence-medium",
            "Low": "confidence-low",
        }.get(confidence, "confidence-low")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Category", output.get("category", "—"))
        r2.metric("Subcategory", output.get("subcategory", "—"))
        r3.metric("Journey Stage", output.get("journey_stage", "—"))
        with r4:
            st.markdown("**Confidence**")
            st.markdown(
                f'<span class="{conf_class}">{confidence}</span>',
                unsafe_allow_html=True,
            )

        st.info(f"**Summary:** {output.get('summary', '')}")

        st.subheader("✅ Resolution Steps")
        steps = output.get("resolution_steps", [])
        if isinstance(steps, list):
            for i, step in enumerate(steps, 1):
                st.markdown(
                    f'<div class="step-card"><b>Step {i}.</b> {step}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.write(steps)

        with st.expander("🔎 Retrieved Knowledge Base Context", expanded=False):
            st.markdown(f"*Top {top_k} entries after hybrid scoring (α={alpha})*")
            for i, entry in enumerate(results, 1):
                sem = entry["semantic_score"]
                kw  = entry["keyword_score"]
                hyb = entry["hybrid_score"]

                st.markdown(f"**[{i}] {entry['category']} › {entry['subcategory']}** — *{entry['journey_stage']}*")
                st.markdown(
                    f'<span class="score-bar-label">🧠 Semantic: {sem:.3f} &nbsp;|&nbsp; '
                    f'🔑 Keyword: {kw:.3f} &nbsp;|&nbsp; '
                    f'⚡ Hybrid: {hyb:.3f}</span>',
                    unsafe_allow_html=True,
                )
                col_s, col_k, col_h = st.columns(3)
                col_s.progress(min(sem, 1.0), text="Semantic")
                col_k.progress(min(kw, 1.0),  text="Keyword")
                col_h.progress(min(hyb, 1.0), text="Hybrid")
                st.caption(f"**Issue:** {entry['issue_description']}")
                st.caption(f"**Resolution:** {entry['resolution_steps']}")
                st.divider()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG Evaluation
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("📊 RAG Evaluation Dashboard")
    st.markdown(
        "Runs a **10-question test suite** and scores both retrieval quality "
        "and generation quality using an **LLM-as-judge** approach."
    )

    with st.expander("ℹ️ What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | What it measures | Good score |
|---|---|---|
| **Hit Rate @K** | Did the correct *category* appear in the top-K results? | ≥ 80% |
| **Subcategory Hit Rate** | Did the correct *subcategory* appear in top-K? | ≥ 70% |
| **MRR (Mean Reciprocal Rank)** | How highly ranked was the correct result? (1.0 = always #1) | ≥ 0.70 |
| **Avg Hybrid Score** | Average confidence of retrieved results across all queries | ≥ 0.50 |
| **Faithfulness** | Are the LLM's resolution steps grounded in retrieved context? | ≥ 80% |
| **Answer Relevance** | Does the LLM response address the actual complaint? | ≥ 80% |
        """)

    eval_btn = st.button("▶️ Run Evaluation (10 test cases)", type="primary")

    if eval_btn:
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar.")
            st.stop()

        from src.llm import LLMGenerator
        from src.evaluation import RAGEvaluator

        retriever = get_retriever(api_key)
        generator = LLMGenerator(api_key)
        evaluator = RAGEvaluator(api_key)

        progress = st.progress(0, text="Starting evaluation...")

        progress.progress(10, text="📡 Evaluating retrieval (10 queries)...")
        ret_results = evaluator.eval_retrieval(retriever, top_k=top_k, alpha=alpha)

        progress.progress(50, text="🤖 Judging generation quality with LLM (10 queries)...")
        gen_results = evaluator.eval_generation(retriever, generator, top_k=top_k, alpha=alpha)

        progress.progress(100, text="✅ Evaluation complete!")
        progress.empty()

        # Aggregate
        n = len(ret_results)
        hit_rate      = sum(r.hit for r in ret_results) / n
        sub_hit_rate  = sum(r.sub_hit for r in ret_results) / n
        mrr           = sum(r.reciprocal_rank for r in ret_results) / n
        avg_hybrid    = sum(r.avg_hybrid_score for r in ret_results) / n
        avg_faith     = sum(r.faithfulness for r in gen_results) / n
        avg_relevance = sum(r.answer_relevance for r in gen_results) / n

        # ── Summary metrics ───────────────────────────────────────────────────
        st.subheader("🏆 Summary Scores")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        def render_metric(col, label, value):
            css = metric_color(value)
            col.markdown(
                f'<div class="{css}">{value:.0%}</div>'
                f'<div class="metric-label">{label}</div>',
                unsafe_allow_html=True,
            )

        render_metric(m1, "Hit Rate @K",      hit_rate)
        render_metric(m2, "Subcat Hit Rate",   sub_hit_rate)
        render_metric(m3, "MRR @K",            mrr)
        render_metric(m4, "Avg Hybrid Score",  avg_hybrid)
        render_metric(m5, "Faithfulness",      avg_faith)
        render_metric(m6, "Answer Relevance",  avg_relevance)

        st.divider()

        # ── Retrieval breakdown ───────────────────────────────────────────────
        st.subheader("📡 Retrieval Results — Per Query")
        ret_rows = []
        for r in ret_results:
            ret_rows.append({
                "Complaint": r.complaint[:65] + "...",
                "Expected Category": r.expected_category,
                "Expected Subcategory": r.expected_subcategory,
                "Retrieved Categories": ", ".join(r.retrieved_categories),
                "Category Hit": "✅" if r.hit else "❌",
                "Subcat Hit": "✅" if r.sub_hit else "❌",
                "Reciprocal Rank": f"{r.reciprocal_rank:.2f}",
                "Avg Hybrid Score": f"{r.avg_hybrid_score:.3f}",
            })
        st.dataframe(pd.DataFrame(ret_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Generation breakdown ──────────────────────────────────────────────
        st.subheader("🤖 Generation Quality — LLM-as-Judge")
        gen_rows = []
        for r in gen_results:
            gen_rows.append({
                "Complaint": r.complaint[:55] + "...",
                "LLM Category": r.llm_category,
                "LLM Subcategory": r.llm_subcategory,
                "Faithfulness": f"{r.faithfulness:.2f}",
                "Faith Reason": r.faithfulness_reason,
                "Answer Relevance": f"{r.answer_relevance:.2f}",
                "Relevance Reason": r.relevance_reason,
            })
        st.dataframe(pd.DataFrame(gen_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Improvement tips ──────────────────────────────────────────────────
        st.subheader("💡 Improvement Suggestions")
        tips = []
        if hit_rate < 0.8:
            tips.append("📉 **Hit Rate is low** — try increasing Top-K or adding more KB entries for weak categories.")
        if mrr < 0.7:
            tips.append("📉 **MRR is low** — correct result isn't ranking #1. Try increasing α to weight semantic search more.")
        if avg_faith < 0.8:
            tips.append("📉 **Faithfulness is low** — LLM may be hallucinating. Lower temperature in `src/llm.py` or tighten the prompt.")
        if avg_relevance < 0.8:
            tips.append("📉 **Answer Relevance is low** — improve the system prompt in `src/llm.py` to stay focused on the complaint.")
        if not tips:
            tips.append("🎉 **All metrics look great!** Consider adding more test cases or expanding the knowledge base to challenge the system further.")

        for tip in tips:
            st.info(tip)

        # ── Export ────────────────────────────────────────────────────────────
        with st.expander("💾 Export results as CSV"):
            st.download_button(
                "⬇️ Download Retrieval Results",
                pd.DataFrame(ret_rows).to_csv(index=False),
                file_name="eval_retrieval.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Download Generation Results",
                pd.DataFrame(gen_rows).to_csv(index=False),
                file_name="eval_generation.csv",
                mime="text/csv",
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center><sub>Built with Streamlit · FAISS · TF-IDF · OpenAI Gemini 1.5 Flash</sub></center>",
    unsafe_allow_html=True,
)
