# Competitive Intelligence RAG App

A RAG-powered tool that scrapes App Store and Google Play reviews to answer product questions about any competitor app, built for PMs who spend hours reading reviews manually.

## The Problem

Product managers need to monitor competitor apps constantly, but reading hundreds of App Store reviews manually is slow, inconsistent, and doesn't scale. A PM trying to understand what users hate about a competitor's "send money" flow has to wade through noise, irrelevant reviews, and vague feedback with no structure. This happens every sprint cycle, every time a competitor ships an update, and every time a stakeholder asks "what are users saying about X?"

## The Solution

This tool lets you type any app name and an optional feature area, ask a plain-English question, and get a structured answer grounded in real user reviews with sentiment breakdown included. It scrapes live reviews from both the App Store and Google Play, embeds them locally, stores them in a vector database, and uses a free LLM via OpenRouter to answer your question. No manual reading, no spreadsheets, just ask.

## Demo

**Input:**
- App name: `Revolut`
- Feature area: `bank account` *(optional)*
- Question: `What are customers complaining about?`

**Output:**
- A direct answer grounded in retrieved reviews
- Sentiment breakdown: Positive 20% / Negative 60% / Neutral 0% / Uncertain 20%
- Sample reviews that support the answer

**Screenshot:**

<img width="708" height="379" alt="image" src="https://github.com/user-attachments/assets/4fa67aa2-3bf0-476d-9265-8b23e7367767" />

## How to Use

### Prerequisites

- Python 3.9+
- A free API key from [OpenRouter](https://openrouter.ai) (no credit card required)

### Setup

```bash
git clone https://github.com/SaurabhS-pm/competitive-intel-rag.git
cd competitive-intel-rag
python -m pip install -r requirements.txt
```

### Configure

setup .env with OPENROUTER_API_KEY=sk-or-your-key-here

### Run

```bash
python -m streamlit run app.py
```

### Example

```
Input:  App name = "Revolut", Feature area = "bank account", Question = "What friction points do users mention?"
Output: Users frequently mention confusion during the identity verification step...
        Sentiment: Positive 20% / Negative 60% / Neutral 0% / Uncertain 20%
```

## How It Works

The scraper calls Apple's iTunes search API to find the app ID, then pulls up to 500 reviews across 10 pages while scraping Google Play in parallel. Reviews are deduplicated by text content before anything else happens.

The embedder chunks reviews into roughly 200-character segments using sentence boundaries, then embeds them locally using `BAAI/bge-small-en-v1.5` via ONNX runtime with no paid embedding API needed. Those chunks go into ChromaDB on disk, where cosine similarity search retrieves the most relevant ones for your question. If you specify a feature area, the retriever scopes results to that area before running semantic search.

Sentiment analysis maps star ratings to labels where available and falls back to VADER for text-only reviews. A configurable confidence threshold filters ambiguous signals into an "Uncertain" bucket rather than forcing them into positive or negative. The final answer gets assembled by a free LLM via OpenRouter's API, with automatic retry and exponential backoff built in for rate limit errors.

## Tradeoffs and Decisions
**Why a confidence threshold on sentiment**

The first version counted ambiguous reviews like "it's okay I guess" as positive, which skewed the breakdown. A confidence threshold of 0.6 filters weak signals into an "Uncertain" bucket instead, giving a more honest picture. Users can adjust this in the sidebar depending on how strict they want the filter to be.

**Why OpenRouter over a direct API like Groq or OpenAI**

OpenRouter lets me swap models with a single line change and no SDK refactoring. I started with Llama 3.3 70B on the free tier and can upgrade to Claude or GPT-4 for production without touching any other code. It also keeps the tool completely free for other PMs to clone and use.

**Why ChromaDB over Pinecone**

ChromaDB runs locally with zero signup friction, which matters when the goal is for anyone to clone and run this immediately. The honest tradeoff is that it is not production-ready at scale. It lacks managed hosting, access controls, and reliable performance under concurrent load. Migrating to Pinecone or Qdrant would be the first step before any team deployment.

**Why local embeddings via ONNX over OpenAI embeddings**

Using `BAAI/bge-small-en-v1.5` via ONNX runtime keeps the entire embedding pipeline offline and free. The original plan used `fastembed`, but it had no prebuilt wheel for Python 3.14 on Windows. Switching to direct ONNX inference was a deliberate fix and a better long-term choice since it removes an external dependency entirely. Embedding quality at 384 dimensions is identical.

**What I would do differently**

The iTunes RSS API is unreliable for some regions and apps. I would replace it with a more robust scraping approach or a paid reviews API for production use. I would also add time-based filtering so PMs can scope analysis to the last 30, 60, or 90 days, which is far more useful for tracking reaction to specific releases. On the storage side, I would migrate from ChromaDB to Pinecone or Qdrant early. ChromaDB was the right call for a local prototype, but it is not built for production scale and the migration is much easier to do before data accumulates.

## What I Learned

Getting Python 3.14 compatible wheels for every library took longer than writing the actual RAG logic. In a real team setting I would have locked Python to 3.11 from the start and used a virtual environment with pinned versions. These feel like engineering details but they directly affect how fast teammates can onboard, which is a product problem as much as a technical one.

The iTunes RSS feed found apps correctly but returned zero reviews silently with no error and no explanation. Good terminal logging was what surfaced the issue. I rewrote the scraper to call Apple's public iTunes API directly, which turned out to be more reliable, but the lesson is that silent failures are the hardest bugs to catch and logging should never be an afterthought.

The first sentiment breakdown looked clean and convincing but was not accurate. Adding a confidence threshold revealed that roughly five to ten percent of reviews were genuinely ambiguous, and surfacing that honestly is more useful to a PM than a breakdown that implies false precision.

## Next Steps

- Add time-based filtering for the last 30, 60, or 90 days to track sentiment shifts after competitor releases
- Support side-by-side comparison of two apps in the same query
- Add CSV export so PMs can paste findings directly into PRDs or stakeholder updates
- Replace iTunes RSS with a more reliable review source for international apps
- Add cache expiry so stale reviews refresh automatically after seven days

## Built With

- [Streamlit](https://streamlit.io) for the UI
- [ChromaDB](https://www.trychroma.com) as the local vector database
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for local embeddings via ONNX runtime
- [OpenRouter](https://openrouter.ai) for free, model-agnostic LLM access
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) as the fallback sentiment analyser
- Python, httpx, sentence-transformers, onnxruntime
