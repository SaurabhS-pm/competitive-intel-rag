"""
Sentiment analysis module.

Uses a two-tier approach:
  1. Star rating (when available): 4-5 stars -> positive, 3 -> neutral, 1-2 -> negative.
     Confidence is derived from how far the rating is from the nearest decision boundary.
  2. VADER (valence-aware dictionary) for text-based scoring when no rating exists.
     Confidence maps from the absolute compound score (0..1).

Reviews whose confidence falls below CONFIDENCE_THRESHOLD are labelled "uncertain"
rather than being counted toward positive/negative/neutral totals.
"""

import logging

logger = logging.getLogger(__name__)

# Default threshold — overridable per-call via the `threshold` argument.
CONFIDENCE_THRESHOLD = 0.6

_VADER_ANALYZER = None


def _get_vader():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning(
                "vaderSentiment not installed. "
                "Falling back to rating-only sentiment. "
                "Run: pip install vaderSentiment"
            )
    return _VADER_ANALYZER


# ── Per-source scoring ─────────────────────────────────────────────────────────

def _score_from_rating(rating) -> tuple[str, float] | tuple[None, None]:
    """
    Map a star rating to (label, confidence).

    Decision boundaries: 1-2 = negative, 3 = neutral, 4-5 = positive.
    Confidence is the normalised distance from the nearest boundary on a 1-5 scale:
      - Rating 1 or 5 -> 1.0 (unambiguously at an extreme)
      - Rating 2 or 4 -> 0.75 (one step from boundary)
      - Rating 3      -> 0.5  (sits exactly on the neutral boundary)
    Returns (None, None) when the rating is missing or non-numeric.
    """
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return None, None

    if r >= 4.0:
        label = "positive"
        # 5 -> 1.0, 4 -> 0.75
        confidence = 0.5 + (r - 3.0) / 4.0
    elif r >= 3.0:
        label = "neutral"
        # 3 -> 0.5 exactly
        confidence = 0.5
    else:
        label = "negative"
        # 1 -> 1.0, 2 -> 0.75
        confidence = 0.5 + (3.0 - r) / 4.0

    return label, min(confidence, 1.0)


def _score_from_text(text: str) -> tuple[str, float]:
    """
    Use VADER to classify text sentiment and derive a confidence score.

    VADER compound score is in [-1, 1].
    Decision thresholds: >= 0.05 positive, <= -0.05 negative, else neutral.
    Confidence = abs(compound), clipped to [0, 1].
    Reviews near 0 compound (e.g. +/- 0.03) get very low confidence and will
    typically be filtered as "uncertain" at the default threshold.
    """
    analyzer = _get_vader()
    if analyzer is None:
        # No VADER available — return neutral with zero confidence so it is
        # always flagged uncertain rather than silently miscategorised.
        return "neutral", 0.0

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    confidence = min(abs(compound), 1.0)

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, confidence


# ── Public API ─────────────────────────────────────────────────────────────────

def score_chunk(chunk: dict) -> tuple[str, float]:
    """
    Return (label, confidence) for a single chunk dict.
    Prefers star rating (deterministic), falls back to VADER on review text.
    """
    rating = chunk.get("metadata", {}).get("rating")
    label, confidence = _score_from_rating(rating)
    if label is not None:
        return label, confidence
    return _score_from_text(chunk.get("text", ""))


def label_chunk(chunk: dict, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """
    Convenience wrapper: return the final label string including 'uncertain'.
    Kept for backwards compatibility with any callers that used the old API.
    """
    label, confidence = score_chunk(chunk)
    if confidence < threshold:
        return "uncertain"
    return label


def analyze_sentiment(
    chunks: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Compute sentiment for a list of retrieved chunk dicts.

    Reviews whose confidence is below `threshold` are labelled "uncertain"
    and counted separately — they do not contribute to the positive/negative/
    neutral fractions, keeping those signals clean.

    Returns:
        {
            "positive":   float,   # fraction of total (excluding uncertain)
            "negative":   float,
            "neutral":    float,
            "uncertain":  float,   # fraction of total that is low-confidence
            "counts": {
                "positive":  int,
                "negative":  int,
                "neutral":   int,
                "uncertain": int,
            },
            "total":       int,    # total chunks analysed
            "labels":      list[str],  # per-chunk final label (same order as input)
            "confidences": list[float],  # per-chunk raw confidence score
            "threshold":   float,  # threshold that was applied
        }
    """
    empty = {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
        "uncertain": 0.0,
        "counts": {"positive": 0, "negative": 0, "neutral": 0, "uncertain": 0},
        "total": 0,
        "labels": [],
        "confidences": [],
        "threshold": threshold,
    }
    if not chunks:
        return empty

    labels: list[str] = []
    confidences: list[float] = []

    for chunk in chunks:
        raw_label, confidence = score_chunk(chunk)
        confidences.append(confidence)
        if confidence < threshold:
            labels.append("uncertain")
        else:
            labels.append(raw_label)

    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0, "uncertain": 0}
    for lbl in labels:
        counts[lbl] += 1

    total = len(labels)
    return {
        "positive": counts["positive"] / total,
        "negative": counts["negative"] / total,
        "neutral": counts["neutral"] / total,
        "uncertain": counts["uncertain"] / total,
        "counts": counts,
        "total": total,
        "labels": labels,
        "confidences": confidences,
        "threshold": threshold,
    }


def top_reviews_by_sentiment(
    chunks: list[dict],
    labels: list[str],
    sentiment: str,
    n: int = 3,
) -> list[dict]:
    """
    Return up to n chunks that match the given sentiment label,
    ordered by relevance (distance ascending — most relevant first).
    Works with all four labels including 'uncertain'.
    """
    matched = [c for c, lbl in zip(chunks, labels) if lbl == sentiment]
    matched.sort(key=lambda c: c.get("distance", 1.0))
    return matched[:n]
