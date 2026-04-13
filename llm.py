"""
LLM module: calls OpenRouter API using the OpenAI-compatible SDK.
Model: meta-llama/llama-3-8b-instruct:free (free tier, no cost).
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openrouter/auto"


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def build_prompt(
    question: str,
    app_name: str,
    retrieved_chunks: list[dict],
    feature_area: Optional[str] = None,
    sentiment_summary: Optional[dict] = None,
) -> str:
    """
    Construct the RAG prompt from retrieved review chunks.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown").replace("_", " ").title()
        rating = meta.get("rating", "")
        rating_str = f" | Rating: {rating}/5" if rating else ""
        context_parts.append(f"[Review {i} – {source}{rating_str}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    feature_line = f" specifically about '{feature_area}'" if feature_area else ""
    sentiment_line = ""
    if sentiment_summary:
        pos = sentiment_summary.get("positive", 0)
        neg = sentiment_summary.get("negative", 0)
        neu = sentiment_summary.get("neutral", 0)
        sentiment_line = (
            f"\n\nSentiment breakdown of retrieved reviews: "
            f"{pos:.0%} positive, {neg:.0%} negative, {neu:.0%} neutral."
        )

    prompt = f"""You are a product analyst helping a team understand user feedback for the app '{app_name}'.

Below are real user reviews retrieved from the App Store and Google Play{feature_line}.{sentiment_line}

--- REVIEWS ---
{context}
--- END REVIEWS ---

Based solely on the reviews above, answer the following question concisely and specifically.
Cite patterns you see across multiple reviews. Do not invent information not present in the reviews.

Question: {question}

Answer:"""
    return prompt


def ask_llm(
    question: str,
    app_name: str,
    retrieved_chunks: list[dict],
    feature_area: Optional[str] = None,
    sentiment_summary: Optional[dict] = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    """
    Send a RAG prompt to OpenRouter and return the model's answer as a string.
    """
    if not retrieved_chunks:
        return "No relevant reviews were found to answer this question."

    client = _get_client()
    prompt = build_prompt(
        question=question,
        app_name=app_name,
        retrieved_chunks=retrieved_chunks,
        feature_area=feature_area,
        sentiment_summary=sentiment_summary,
    )

    logger.info(f"Sending prompt to {MODEL} ({len(retrieved_chunks)} review chunks)...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise, data-driven product analyst. "
                        "Answer questions based only on the provided user reviews."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()
    except Exception as e:
        logger.error(f"OpenRouter API call failed: {e}")
        raise
