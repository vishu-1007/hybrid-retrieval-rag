# 🎫 RAG Support Ticket Classifier (AI-Powered Customer Support Tool)

Built an end-to-end **Retrieval-Augmented Generation (RAG)** system that classifies customer support complaints and generates structured resolution steps using hybrid search and Gemini 2.5 Flash.

🔗 **[Live Demo](https://your-vercel-link.vercel.app)** &nbsp;|&nbsp; ⭐ **Star this repo if you found it helpful!**

---

## 📸 Demo

> _Add a screenshot of your app here_

---

## 🔄 How It Works

```
Customer Complaint (free text)
        │
        ▼
┌──────────────────────────────────────┐
│         Hybrid Retrieval             │
│  ┌────────────────┐  ┌─────────────┐ │
│  │ Gemini 2.5     │  │ TF-IDF      │ │
│  │ Flash Semantic │  │ Keyword     │ │
│  │ Scoring        │  │ Cosine Sim  │ │
│  └───────┬────────┘  └──────┬──────┘ │
│          └────── α ─────────┘        │
│           Weighted Hybrid Score      │
└──────────────────┬───────────────────┘
                   │ Top-k context entries
                   ▼
        ┌─────────────────────┐
        │  Gemini 2.5 Flash   │
        │  Structured Prompt  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Structured Output  │
        │  - Category         │
        │  - Subcategory      │
        │  - Journey Stage    │
        │  - Confidence       │
        │  - Resolution Steps │
        └─────────────────────┘
```

---

## ✨ Features

- **Hybrid Retrieval** — combines Gemini 2.5 Flash semantic scoring with TF-IDF keyword search using a weighted alpha formula
- **Structured LLM Output** — classifies complaint by category, subcategory, journey stage, confidence, and generates step-by-step resolution
- **RAG Evaluation Dashboard** — measures Hit Rate, MRR, Faithfulness, and Answer Relevance using LLM-as-judge methodology
- **Synthetic KB Generator** — CLI tool to expand the knowledge base using any LLM via litellm
- **Secure API Handling** — API key never exposed in UI, loaded silently from `.env`
- **Deployed on Vercel** — production-ready with live demo

---

## 🗂️ Project Structure

```
hybrid-retrieval-rag/
├── app.py                    # Streamlit application (entry point)
├── generate_kb.py            # Synthetic knowledge base generator (litellm)
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variable file
├── data/
│   └── knowledge_base.csv    # 30 support issue entries across 9 categories
└── src/
    ├── retrieval.py          # HybridRetriever (Gemini semantic + TF-IDF)
    ├── llm.py                # LLM response generator (Gemini 2.5 Flash)
    └── evaluation.py         # RAG evaluation metrics (LLM-as-judge)
```

---

## 🧠 Key Concepts

| Concept | Description |
|---|---|
| **RAG** | Retrieve relevant knowledge, then generate a grounded response |
| **Semantic search** | Gemini 2.5 Flash scores each KB entry's relevance to the query |
| **Keyword search** | TF-IDF cosine similarity — captures exact terms like "OTP", "UPI" |
| **Hybrid scoring** | `score = α × semantic + (1−α) × keyword` |
| **LLM-as-judge** | Gemini evaluates its own outputs for faithfulness and relevance |
| **Hit Rate** | Did the correct category appear in top-K results? |
| **MRR** | Mean Reciprocal Rank — how high the correct result ranked |

---

## 🧱 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Web UI |
| Google Gemini 2.5 Flash | Semantic scoring, classification, evaluation |
| google-genai SDK | Gemini API client |
| Scikit-learn | TF-IDF vectorizer, cosine similarity |
| Pandas / NumPy | Data handling and score normalization |
| litellm | Multi-provider KB generation |
| Vercel | Deployment and hosting |

---

## 📚 Knowledge Base

`data/knowledge_base.csv` contains **30 support ticket templates** across 9 categories:

| Category | Examples |
|---|---|
| Payment | Failed transaction, UPI failure, refund, OTP, duplicate charge |
| Account | Login failure, account locked, KYC, profile update |
| Order | Order not placed, wrong item, delayed, cancellation, return |
| Delivery | Damaged package, wrong address, agent issue |
| Technical | App crash, slow loading, search failure, gateway error |
| Coupon | Code not applied, cashback not credited |
| Subscription | Not activated, auto-renewal dispute |
| Wallet | Balance not updated, payment failed |
| Fraud | Unauthorized transaction, account hacked |

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hybrid-retrieval-rag.git
cd hybrid-retrieval-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key

```bash
cp .env.example .env
# Edit .env and add your key
```

Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

```
GEMINI_API_KEY=your-gemini-api-key-here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧑‍💻 Usage

1. Enter your **Gemini API key** in the sidebar (or set it in `.env`)
2. Adjust **Top-K** (number of context entries) and **α** (semantic vs keyword weight)
3. Type a customer complaint or pick a sample from the dropdown
4. Click **Classify & Resolve**
5. View the structured classification and resolution steps
6. Expand **Retrieved Knowledge Base Context** to see hybrid scores
7. Switch to the **Evaluation** tab to run RAG quality metrics

---

## 🤖 Generating Synthetic Knowledge Base Data

```bash
# Generate 10 entries using default model
python generate_kb.py --count 10

# Generate for specific categories only
python generate_kb.py --categories Payment Order --count 5

# Dry run — print without saving
python generate_kb.py --count 3 --dry-run

# Use a different model
python generate_kb.py --model gemini/gemini-1.5-flash --count 10
```

---

## 📊 RAG Evaluation Metrics

The evaluation tab runs 10 test complaints through the full pipeline and reports:

| Metric | Description | Target |
|---|---|---|
| **Hit Rate @K** | Correct category in top-K results | ≥ 80% |
| **Subcategory Hit Rate** | Correct subcategory in top-K | ≥ 70% |
| **MRR** | Mean Reciprocal Rank | ≥ 0.70 |
| **Faithfulness** | Resolution grounded in context | ≥ 0.80 |
| **Answer Relevance** | Response addresses the complaint | ≥ 0.80 |

---

## 🔧 Extending the Project

- **Add more KB entries** — run `generate_kb.py` or edit `data/knowledge_base.csv`
- **Add reranking** — insert a cross-encoder between retrieval and generation
- **Swap the LLM** — change `LLM_MODEL` in `src/llm.py`
- **Add more categories** — extend the KB and update test cases in `src/evaluation.py`
- **Persist evaluation logs** — export results to CSV for tracking over time

---

## 🙌 Acknowledgements

- [Google Gemini](https://deepmind.google/technologies/gemini/) for the LLM
- [Streamlit](https://streamlit.io) for the UI framework
- [litellm](https://docs.litellm.ai) for multi-provider LLM support

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
