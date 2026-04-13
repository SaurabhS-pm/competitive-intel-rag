"""
Scraper module for App Store and Google Play reviews.

App Store  — uses the iTunes WebObjects MZStore API (the internal endpoint that
             returns JSON review lists). The old RSS feed at itunes.apple.com/rss/
             customerreviews/ is deprecated and returns 0 entries. The AMP API at
             amp-api.apps.apple.com requires a bearer token that Apple no longer
             embeds in the server-rendered page HTML.

Play Store — uses google-play-scraper. The library's search() function can return
             appId=None for certain apps (e.g. Wise). We fall back to scraping the
             Play Store search results page with httpx to extract the real package ID.
"""

import logging
import re
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── App Store ──────────────────────────────────────────────────────────────────

_ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
_WEBOBJECTS_URL = (
    "https://itunes.apple.com/WebObjects/MZStore.woa/wa/userReviewsRow"
)
_WEBOBJECTS_HEADERS = {
    # These headers make the endpoint believe it's talking to the iTunes desktop
    # client, which returns the full JSON review list.
    "X-Apple-Store-Front": "143441-1,29",  # US store, English
    "X-Apple-Tz": "-18000",
    "User-Agent": "iTunes/12.11 (Macintosh; OS X 10.15.7) AppleWebKit/606",
}
_PAGE_SIZE = 19  # Apple's WebObjects API returns exactly 19 reviews per page


def _appstore_lookup(app_name: str, country: str = "us") -> Optional[int]:
    """
    Search the iTunes Search API for an iOS app by name.
    Returns the numeric trackId or None.
    """
    params = {
        "term": app_name,
        "country": country,
        "entity": "software",
        "limit": 5,
    }
    try:
        resp = httpx.get(_ITUNES_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            hit = results[0]
            logger.info(
                "App Store: found '%s' (id=%s) for query '%s'",
                hit.get("trackName"), hit.get("trackId"), app_name,
            )
            return hit.get("trackId")
    except Exception as exc:
        logger.warning("App Store lookup failed for '%s': %s", app_name, exc)
    return None


def _appstore_page(app_id: int, start: int) -> list[dict]:
    """
    Fetch one page of reviews from the iTunes WebObjects API.
    Each page contains up to 19 reviews.
    Returns a list of normalised review dicts.
    """
    params = {
        "id": str(app_id),
        "displayable-kind": "11",
        "startIndex": str(start),
        "endIndex": str(start + _PAGE_SIZE - 1),
        "sort": "4",       # sort=4 → most recent
        "appVersion": "",
    }
    try:
        resp = httpx.get(
            _WEBOBJECTS_URL,
            params=params,
            headers=_WEBOBJECTS_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        raw_list = resp.json().get("userReviewList", [])
        reviews = []
        for r in raw_list:
            body = (r.get("body") or "").strip()
            if not body:
                continue
            reviews.append({
                "id": str(r.get("userReviewId", "")),
                "text": body,
                "rating": float(r.get("rating", 0)),
                "date": (r.get("date") or "")[:10],
                "title": r.get("title", ""),
                "source": "app_store",
            })
        return reviews
    except Exception as exc:
        logger.warning("App Store page (offset=%d) failed: %s", start, exc)
        return []


def scrape_app_store(app_name: str, count: int = 200) -> list[dict]:
    """Scrape up to `count` reviews from the Apple App Store."""
    app_id = _appstore_lookup(app_name)
    if not app_id:
        logger.warning("App Store: could not find app '%s'", app_name)
        return []

    all_reviews: list[dict] = []
    seen_ids: set[str] = set()
    offset = 0

    while len(all_reviews) < count:
        page = _appstore_page(app_id, start=offset)
        if not page:
            break
        for r in page:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_reviews.append(r)
        if len(page) < _PAGE_SIZE:
            break  # last page
        offset += _PAGE_SIZE
        time.sleep(0.4)

    result = all_reviews[:count]
    logger.info("App Store: scraped %d reviews for '%s'", len(result), app_name)
    return result


# ── Play Store ─────────────────────────────────────────────────────────────────

_PLAY_SEARCH_URL = "https://play.google.com/store/search"
_PLAY_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _play_resolve_package(app_name: str) -> Optional[str]:
    """
    Resolve an app name to a Google Play package ID.

    Strategy:
    1. PRIMARY: Scrape the Play Store search results page with httpx.
       This is the most reliable method — it returns the same ranking Google
       shows to users, and package IDs are always present in the page HTML.
    2. FALLBACK: Use google-play-scraper's search() — sometimes returns
       appId=None for the top result (e.g. Wise), but useful as a backup.
    """
    # Primary: HTML scrape — most reliable ranking
    try:
        resp = httpx.get(
            _PLAY_SEARCH_URL,
            params={"q": app_name, "c": "apps", "hl": "en", "gl": "us"},
            headers={"User-Agent": _PLAY_BROWSER_UA},
            follow_redirects=True,
            timeout=15,
        )
        resp.raise_for_status()
        # Package IDs appear in URLs like /store/apps/details?id=com.example.app
        pkg_ids = re.findall(r"/store/apps/details\?id=([\w.]+)", resp.text)
        # Deduplicate while preserving order
        seen: list[str] = []
        for p in pkg_ids:
            if p not in seen:
                seen.append(p)
        if seen:
            logger.info(
                "Play Store (HTML scrape): '%s' -> %s (from %d candidates)",
                app_name, seen[0], len(seen),
            )
            return seen[0]
    except Exception as exc:
        logger.warning("Play Store HTML scrape failed for '%s': %s", app_name, exc)

    # Fallback: google-play-scraper library search
    logger.info(
        "Play Store HTML scrape found nothing — trying library search for '%s'",
        app_name,
    )
    try:
        from google_play_scraper import search as gp_search

        results = gp_search(app_name, n_hits=10, lang="en", country="us")
        valid = [r for r in results if r.get("appId")]
        if valid:
            pkg = valid[0]["appId"]
            logger.info("Play Store (library fallback): '%s' -> %s", app_name, pkg)
            return pkg
    except Exception as exc:
        logger.debug("Play Store library search failed: %s", exc)

    return None


def scrape_play_store(app_name: str, count: int = 200) -> list[dict]:
    """Scrape up to `count` reviews from Google Play Store."""
    try:
        from google_play_scraper import reviews as gp_reviews, Sort
    except ImportError:
        logger.error(
            "google-play-scraper not installed. Run: pip install google-play-scraper"
        )
        return []

    pkg_id = _play_resolve_package(app_name)
    if not pkg_id:
        logger.warning("Play Store: could not resolve package ID for '%s'", app_name)
        return []

    try:
        # Fetch in batches; google-play-scraper supports continuation tokens
        all_reviews: list[dict] = []
        seen_ids: set[str] = set()
        continuation_token = None

        while len(all_reviews) < count:
            batch_size = min(200, count - len(all_reviews))
            kwargs: dict = {
                "app_id": pkg_id,
                "lang": "en",
                "country": "us",
                "sort": Sort.NEWEST,
                "count": batch_size,
            }
            if continuation_token is not None:
                kwargs["continuation_token"] = continuation_token

            result, continuation_token = gp_reviews(**kwargs)

            if not result:
                break

            for r in result:
                text = (r.get("content") or "").strip()
                rid = str(r.get("reviewId", ""))
                if not text or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                all_reviews.append({
                    "id": rid,
                    "text": text,
                    "rating": r.get("score"),
                    "date": str(r.get("at", ""))[:10],
                    "title": "",
                    "source": "play_store",
                })

            if continuation_token is None:
                break
            time.sleep(0.4)

        logger.info(
            "Play Store: scraped %d reviews for '%s' (%s)",
            len(all_reviews), app_name, pkg_id,
        )
        return all_reviews

    except Exception as exc:
        logger.warning(
            "Play Store reviews failed for '%s' (%s): %s", app_name, pkg_id, exc
        )
        return []


# ── Combined ───────────────────────────────────────────────────────────────────

def scrape_reviews(
    app_name: str,
    count_per_store: int = 200,
    feature_area: Optional[str] = None,
) -> list[dict]:
    """
    Scrape reviews from both App Store and Play Store.
    Optionally filter by feature_area keyword (substring match on review text).
    Deduplicates by review text content.
    """
    logger.info(
        "Scraping reviews for '%s' (up to %d per store)...",
        app_name, count_per_store,
    )

    ios_reviews = scrape_app_store(app_name, count=count_per_store)
    time.sleep(1)
    android_reviews = scrape_play_store(app_name, count=count_per_store)

    all_reviews = ios_reviews + android_reviews

    # Deduplicate by normalised text fingerprint
    seen: set[str] = set()
    unique: list[dict] = []
    for r in all_reviews:
        key = r["text"].lower().strip()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Optional keyword filter
    if feature_area and feature_area.strip():
        kw = feature_area.lower()
        unique = [r for r in unique if kw in r["text"].lower()]
        logger.info(
            "After filtering by '%s': %d reviews remain", feature_area, len(unique)
        )

    logger.info("Total unique reviews collected: %d", len(unique))
    return unique
