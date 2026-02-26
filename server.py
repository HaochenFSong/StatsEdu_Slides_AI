#!/usr/bin/env python3
"""StatEdu Slides backend.

Lightweight local server for:
- prompt + file upload -> generated deck metadata + .qmd
- human-in-the-loop refinement
- optional Quarto revealjs render (if `quarto` is installed)
- mock Google Slides export handshake
"""

from __future__ import annotations

import cgi
import base64
import html
import io
import json
import os
import re
import socket
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs, quote_plus, unquote, urlparse


def parse_optional_timeout(raw_value: str | None, default: float | None) -> float | None:
    raw = str(raw_value or "").strip().lower()
    if not raw:
        return default
    if raw in {"none", "off", "false", "0", "infinite", "infinity", "unlimited"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return None
    return value


HOST = os.getenv("STATEDU_HOST", "127.0.0.1")
PORT = int(os.getenv("STATEDU_PORT", "8000"))
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".statedu"
DECK_DIR = DATA_DIR / "decks"
UPLOAD_DIR = DATA_DIR / "uploads"

SUPPORTED_TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".py", ".r", ".qmd"}
LLM_TIMEOUT_SEC = parse_optional_timeout(os.getenv("STATEDU_LLM_TIMEOUT_SEC"), None)
LLM_RETRY_COUNT = max(0, int(os.getenv("STATEDU_LLM_RETRY_COUNT", "1")))
LLM_RETRY_BACKOFF_SEC = float(os.getenv("STATEDU_LLM_RETRY_BACKOFF_SEC", "1.2"))
DEFAULT_SLIDE_COUNT = int(os.getenv("STATEDU_DEFAULT_SLIDE_COUNT", "8"))
MAX_SLIDE_COUNT = int(os.getenv("STATEDU_MAX_SLIDE_COUNT", "40"))
RENDER_STEP_DELAY_SEC = float(os.getenv("STATEDU_RENDER_STEP_DELAY_SEC", "0.35"))
VALID_TEACHING_STYLES = {"balanced", "conceptual", "mathematical", "simulation"}
WEB_RESEARCH_ENABLED = os.getenv("STATEDU_WEB_RESEARCH_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
WEB_RESEARCH_MAX_RESULTS = int(os.getenv("STATEDU_WEB_RESEARCH_MAX_RESULTS", "5"))
WEB_RESEARCH_TIMEOUT_SEC = int(os.getenv("STATEDU_WEB_RESEARCH_TIMEOUT_SEC", "6"))
SLIDE_AUTO_SPLIT_ENABLED = os.getenv("STATEDU_SLIDE_AUTO_SPLIT_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
IMAGE_GENERATION_ENABLED = os.getenv("STATEDU_IMAGE_GENERATION_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
IMAGE_PROVIDER = os.getenv("STATEDU_IMAGE_PROVIDER", "local").strip().lower()
IMAGE_MAX_SLIDES = max(1, min(MAX_SLIDE_COUNT, int(os.getenv("STATEDU_IMAGE_MAX_SLIDES", "12"))))
IMAGE_STYLE_PROMPT = os.getenv(
    "STATEDU_IMAGE_STYLE_PROMPT",
    "clean academic educational illustration, minimal, high readability, no text labels",
).strip()
OPENAI_IMAGE_MODEL = os.getenv("STATEDU_OPENAI_IMAGE_MODEL", "gpt-image-1").strip()
OPENAI_IMAGE_SIZE = os.getenv("STATEDU_OPENAI_IMAGE_SIZE", "1536x1024").strip()

MAX_SUBTITLE_CHARS = 120
MAX_BULLET_CHARS = 150
MAX_EXAMPLE_CHARS = 260
MAX_ACTIVITY_CHARS = 220
MAX_NOTES_CHARS = 170

JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
RESEARCH_IGNORE_PROMPT_TERMS = {
    "slide",
    "slides",
    "lecture",
    "lectures",
    "student",
    "students",
    "class",
    "classroom",
    "undergraduate",
    "intro",
    "introduction",
    "teach",
    "teaching",
    "example",
    "examples",
    "live",
    "coding",
}
RESEARCH_STATS_ANCHORS = {
    "statistics",
    "statistical",
    "probability",
    "inference",
    "hypothesis",
    "p-value",
    "confidence interval",
    "regression",
    "classification",
    "clustering",
    "cluster",
    "k-means",
    "kmeans",
    "hierarchical clustering",
    "silhouette",
    "wcss",
    "distance metric",
    "euclidean",
    "manhattan",
    "standardization",
    "z-score",
    "dendrogram",
    "sampling",
    "distribution",
    "variance",
    "covariance",
    "machine learning",
    "unsupervised learning",
    "data analysis",
}
RESEARCH_BIO_NOISE_TERMS = {
    "musician",
    "composer",
    "artist",
    "album",
    "song",
    "band",
    "film",
    "actor",
    "actress",
    "painter",
    "painting",
    "biography",
    "born",
    "died",
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, DECK_DIR, UPLOAD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_quarto_bin() -> str | None:
    explicit = os.getenv("STATEDU_QUARTO_BIN", "").strip()
    if explicit and Path(explicit).is_file() and os.access(explicit, os.X_OK):
        return explicit

    local_bin = BASE_DIR / ".tools" / "bin" / "quarto"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    return shutil.which("quarto")


def resolve_llm_provider() -> str:
    explicit = os.getenv("STATEDU_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("STATEDU_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("STATEDU_GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("STATEDU_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def resolve_llm_model(provider: str) -> str:
    if provider == "openai":
        return os.getenv("STATEDU_OPENAI_MODEL", "gpt-4.1-mini")
    if provider == "gemini":
        return os.getenv("STATEDU_GEMINI_MODEL", "gemini-1.5-flash")
    if provider == "anthropic":
        return os.getenv("STATEDU_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    return "heuristic"


def resolve_llm_key(provider: str) -> str:
    if provider == "openai":
        return os.getenv("STATEDU_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    if provider == "gemini":
        return os.getenv("STATEDU_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
    if provider == "anthropic":
        return os.getenv("STATEDU_ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")).strip()
    return ""


def llm_status() -> dict[str, object]:
    provider = resolve_llm_provider()
    configured = provider in {"mock", "none", "off"} or bool(resolve_llm_key(provider))
    return {
        "provider": provider,
        "model": resolve_llm_model(provider),
        "configured": configured,
    }


def resolve_image_provider() -> str:
    provider = IMAGE_PROVIDER or "local"
    if provider in {"openai", "local", "none", "off"}:
        return provider
    return "local"


def resolve_image_model(provider: str) -> str:
    if provider == "openai":
        return OPENAI_IMAGE_MODEL or "gpt-image-1"
    if provider == "local":
        return "local-svg"
    return "none"


def resolve_image_key(provider: str) -> str:
    if provider == "openai":
        return resolve_llm_key("openai")
    return ""


def image_status() -> dict[str, object]:
    provider = resolve_image_provider()
    configured = (not IMAGE_GENERATION_ENABLED) or provider in {"local", "none", "off"} or bool(resolve_image_key(provider))
    return {
        "enabled": IMAGE_GENERATION_ENABLED,
        "provider": provider,
        "model": resolve_image_model(provider),
        "configured": configured,
        "maxSlides": IMAGE_MAX_SLIDES,
    }


def fetch_json_url(url: str, timeout_sec: int) -> dict[str, object]:
    req = urlrequest.Request(
        url,
        headers={
            "User-Agent": "StatEduSlidesHelperAI/0.3 (+https://localhost)",
            "Accept": "application/json, text/plain, */*",
        },
        method="GET",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object response.")
    return parsed


def clean_html_snippet(text: object, limit: int = 260) -> str:
    value = str(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip(" ,;:") + "..."


def tokenize_research_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", str(text or "").lower())
    return set(words)


def is_research_item_relevant(*, prompt: str, title: str, snippet: str) -> bool:
    prompt_tokens = tokenize_research_terms(prompt) - RESEARCH_IGNORE_PROMPT_TERMS
    text = f"{title} {snippet}".lower()
    text_tokens = tokenize_research_terms(text)

    if any(anchor in text for anchor in RESEARCH_STATS_ANCHORS):
        return True

    overlap = prompt_tokens & text_tokens
    if len(overlap) >= 2:
        return True

    bio_noise_hits = sum(1 for term in RESEARCH_BIO_NOISE_TERMS if term in text)
    if bio_noise_hits >= 2:
        return False

    return False


def normalize_source_url(raw: object) -> str:
    text = str(raw or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return ""


def source_domain(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def add_research_item(
    bucket: list[dict[str, str]],
    seen: set[str],
    *,
    title: object,
    url: object,
    snippet: object,
    provider: str,
    limit: int,
) -> None:
    normalized_url = normalize_source_url(url)
    normalized_title = sanitize_text(title)
    normalized_snippet = clean_html_snippet(snippet)
    if not normalized_url or not normalized_title:
        return
    key = normalized_url.lower()
    if key in seen:
        return
    seen.add(key)
    bucket.append(
        {
            "title": normalized_title,
            "url": normalized_url,
            "snippet": normalized_snippet,
            "provider": provider,
            "domain": source_domain(normalized_url),
        }
    )
    if len(bucket) > limit:
        del bucket[limit:]


def build_web_queries(prompt: str, source_excerpt: str) -> list[str]:
    prompt_short = re.sub(r"\s+", " ", prompt).strip()
    prompt_short = prompt_short[:180]
    source_seed = re.sub(r"\s+", " ", source_excerpt).strip()
    source_seed = source_seed[:140]

    queries = [
        prompt_short,
        f"{prompt_short} statistics explanation",
        f"{prompt_short} classroom example",
    ]
    if source_seed:
        queries.append(f"{prompt_short} {source_seed}")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in queries:
        q = item.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(q)
    return normalized[:4]


def search_duckduckgo_instant(query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
    data = fetch_json_url(url, WEB_RESEARCH_TIMEOUT_SEC)
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    add_research_item(
        results,
        seen,
        title=data.get("Heading", ""),
        url=data.get("AbstractURL", ""),
        snippet=data.get("AbstractText", ""),
        provider="duckduckgo",
        limit=limit,
    )

    related = data.get("RelatedTopics", [])
    if isinstance(related, list):
        for entry in related:
            if len(results) >= limit:
                break
            if isinstance(entry, dict) and "Topics" in entry and isinstance(entry.get("Topics"), list):
                for nested in entry.get("Topics", []):
                    if len(results) >= limit:
                        break
                    if not isinstance(nested, dict):
                        continue
                    add_research_item(
                        results,
                        seen,
                        title=nested.get("Text", ""),
                        url=nested.get("FirstURL", ""),
                        snippet=nested.get("Text", ""),
                        provider="duckduckgo",
                        limit=limit,
                    )
                continue
            if not isinstance(entry, dict):
                continue
            add_research_item(
                results,
                seen,
                title=entry.get("Text", ""),
                url=entry.get("FirstURL", ""),
                snippet=entry.get("Text", ""),
                provider="duckduckgo",
                limit=limit,
            )
    return results


def search_wikipedia(query: str, limit: int) -> list[dict[str, str]]:
    api_url = (
        "https://en.wikipedia.org/w/api.php?"
        f"action=query&list=search&format=json&utf8=1&srlimit={max(1, min(limit, 10))}&srsearch={quote_plus(query)}"
    )
    data = fetch_json_url(api_url, WEB_RESEARCH_TIMEOUT_SEC)
    query_obj = data.get("query", {})
    entries = query_obj.get("search", []) if isinstance(query_obj, dict) else []

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(entries, list):
        return results

    for item in entries:
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue
        title = sanitize_text(item.get("title", ""))
        pageid = item.get("pageid")
        snippet = clean_html_snippet(item.get("snippet", ""))
        if not title or not pageid:
            continue
        url = f"https://en.wikipedia.org/?curid={pageid}"
        add_research_item(
            results,
            seen,
            title=title,
            url=url,
            snippet=snippet,
            provider="wikipedia",
            limit=limit,
        )
    return results


def collect_web_research(prompt: str, source_excerpt: str, max_results: int) -> tuple[list[dict[str, str]], str | None]:
    if not WEB_RESEARCH_ENABLED:
        return [], None
    limit = max(1, min(max_results, 10))
    queries = build_web_queries(prompt, source_excerpt)
    if not queries:
        return [], None

    aggregated: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    warnings: list[str] = []
    failures = 0

    for query in queries:
        if len(aggregated) >= limit:
            break
        try:
            ddg_results = search_duckduckgo_instant(query, limit=limit)
            for item in ddg_results:
                url = item.get("url", "")
                title = str(item.get("title", ""))
                snippet = str(item.get("snippet", ""))
                if not url or url in seen_urls:
                    continue
                if not is_research_item_relevant(prompt=prompt, title=title, snippet=snippet):
                    continue
                seen_urls.add(url)
                aggregated.append(item)
                if len(aggregated) >= limit:
                    break
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"duckduckgo: {exc}")
            failures += 1
            if failures >= 2 and not aggregated:
                break

    if len(aggregated) < limit:
        for query in queries:
            if len(aggregated) >= limit:
                break
            try:
                wiki_results = search_wikipedia(query, limit=limit)
                for item in wiki_results:
                    url = item.get("url", "")
                    title = str(item.get("title", ""))
                    snippet = str(item.get("snippet", ""))
                    if not url or url in seen_urls:
                        continue
                    if not is_research_item_relevant(prompt=prompt, title=title, snippet=snippet):
                        continue
                    seen_urls.add(url)
                    aggregated.append(item)
                    if len(aggregated) >= limit:
                        break
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"wikipedia: {exc}")
                failures += 1
                if failures >= 4 and not aggregated:
                    break

    warning = None
    if not aggregated and warnings:
        warning = "Web research unavailable in this environment; continued with prompt/source only."
    return aggregated[:limit], warning


def format_research_context(results: list[dict[str, str]]) -> str:
    if not results:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        title = sanitize_text(item.get("title", ""))
        domain = sanitize_text(item.get("domain", ""))
        url = normalize_source_url(item.get("url", ""))
        snippet = clean_html_snippet(item.get("snippet", ""))
        if not title or not url:
            continue
        label = f"{idx}. {title}"
        if domain:
            label += f" ({domain})"
        lines.append(label)
        if snippet:
            lines.append(f"   Summary: {snippet}")
        lines.append(f"   Link: {url}")
    return "\n".join(lines)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_slide_count(value: int) -> int:
    return max(3, min(MAX_SLIDE_COUNT, value))


def slugify(text: str, fallback: str = "deck") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or fallback


def infer_slide_count(text: str) -> int:
    m = re.search(r"(\d{1,2})\s*(?:-|\s)?slides?", text, flags=re.IGNORECASE)
    if not m:
        return clamp_slide_count(DEFAULT_SLIDE_COUNT)
    value = int(m.group(1))
    return clamp_slide_count(value)


def infer_title(text: str) -> str:
    if not text:
        return "Statistical Concepts Overview"
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text).strip()
    words = clean.split()[:10]
    return " ".join(words) if words else "Generated Lesson Deck"


def infer_bullets(seed_text: str) -> list[str]:
    baseline = [
        "Define the core concept in plain language before formulas.",
        "Show one worked example with interpretation steps.",
        "Add one misconception check for active learning.",
        "Close with a short formative assessment prompt.",
    ]

    if not seed_text:
        return baseline

    low = seed_text.lower()
    matched: list[str] = []
    if "normal" in low or "distribution" in low:
        matched.append("Introduce mean, variance, and z-score intuition with one visual.")
    if "hypothesis" in low or "test" in low:
        matched.append("Explain null vs alternative and decision boundaries.")
    if "regression" in low:
        matched.append("Interpret slope and intercept in context, not only algebraically.")
    if "confidence interval" in low or "interval" in low:
        matched.append("Connect confidence level to repeated sampling interpretation.")
    if "beginner" in low or "intro" in low:
        matched.append("Use low-jargon language and one idea per slide.")
    if "advanced" in low or "graduate" in low:
        matched.append("Include assumptions, edge cases, and diagnostics.")

    return (matched + baseline)[:5]


def normalize_teaching_style(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value in VALID_TEACHING_STYLES:
        return value
    return "balanced"


def sentence_completion_boundary(text: str, preferred_max: int, extra_window: int = 120) -> int | None:
    if not text:
        return None
    upper = min(len(text), preferred_max + max(20, extra_window))
    if upper <= preferred_max:
        return None
    segment = text[preferred_max:upper]
    match = re.search(r"[.!?](?:\s|$)", segment)
    if not match:
        return None
    return preferred_max + match.end()


def clamp_sentence(text: object, max_chars: int) -> str:
    value = sanitize_text(text)
    if len(value) <= max_chars:
        return value

    completion_idx = sentence_completion_boundary(value, max_chars, extra_window=120)
    if completion_idx is not None and completion_idx <= len(value):
        completed = value[:completion_idx].strip()
        if completed:
            return completed

    min_keep = max(20, int(max_chars * 0.55))

    punct_positions = [m.end() for m in re.finditer(r"[.!?;:]", value) if m.end() <= max_chars]
    punct_positions = [p for p in punct_positions if p >= min_keep]
    if punct_positions:
        candidate = value[: punct_positions[-1]].strip()
        candidate = candidate.rstrip(";:").strip()
        return candidate or value[: punct_positions[-1]].strip()

    comma_positions = [m.start() for m in re.finditer(r",", value) if m.start() <= max_chars]
    comma_positions = [p for p in comma_positions if p >= min_keep]
    if comma_positions:
        candidate = value[: comma_positions[-1]].rstrip(" ,;:-").strip()
        if candidate:
            if candidate[-1:] not in ".!?":
                candidate += "."
            return candidate

    clipped = value[:max_chars].rstrip(" ,;:-")
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip(" ,;:-")
    clipped = re.sub(
        r"\b(of|to|for|with|by|from|and|or|that|which|when|where|while|as|in|on|at|into|about|using|based)\s*$",
        "",
        clipped,
        flags=re.IGNORECASE,
    ).strip()
    if clipped:
        if clipped[-1:] not in ".!?":
            clipped += "."
        return clipped
    fallback = value[:max_chars].strip()
    if fallback and fallback[-1:] not in ".!?":
        fallback += "."
    return fallback


def is_unhelpful_source_sentence(text: str) -> bool:
    low = text.lower()
    return (
        "not extractable in this environment" in low
        or "not configured yet in this local build" in low
        or "unsupported source type for parsing" in low
    )


def is_presenter_directive(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    patterns = [
        "ask students",
        "pause for",
        "cold-call",
        "debrief",
        "as an instructor",
        "as the instructor",
        "teacher should",
        "presenter should",
        "you should explain",
    ]
    return any(p in low for p in patterns)


def sanitize_text(value: object) -> str:
    text = str(value or "").strip()
    text = text.replace("\u2019", "'").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def sanitize_bullet_text(value: object) -> str:
    text = sanitize_text(value)
    if not text:
        return ""
    text = re.sub(r"\s*\{[^}]*\}\s*$", "", text).strip()
    text = re.sub(r"^\s*[-+*]\s*", "", text).strip()
    text = re.sub(r"^(definition|context|useful material|materials?)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\.{3,}", ".", text).strip()
    text = re.sub(r"\.{3,}\s*$", "", text).strip()
    return text


def normalize_material_list(raw: object, max_items: int = 3) -> list[str]:
    items: list[str] = []
    if isinstance(raw, list):
        source = raw
    elif isinstance(raw, str):
        source = re.split(r"[;\n]+", raw)
    else:
        source = []
    for item in source:
        text = clamp_sentence(item, 88)
        if text and not is_presenter_directive(text):
            items.append(text)
    # stable dedupe
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out[:max_items]


def polish_sentence_field(text: object, max_chars: int) -> str:
    value = clamp_sentence(text, max_chars)
    if not value:
        return ""
    value = re.sub(
        r"\b(of|to|for|with|by|from|and|or|that|which|when|where|while|as|in|on|at|into|about|using|based)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")
    if value and value[-1:] not in ".!?":
        value += "."
    return value


def compose_student_bullets(
    definition: str,
    context: str,
    materials: list[str],
    bullets: list[str],
    *,
    max_items: int = 6,
) -> list[str]:
    out: list[str] = []
    for item in bullets:
        txt = clamp_sentence(sanitize_bullet_text(item), MAX_BULLET_CHARS)
        if not txt or is_presenter_directive(txt):
            continue
        if txt not in out:
            out.append(txt)
        if len(out) >= max_items:
            break

    if len(out) < 2:
        for item in [definition, context]:
            txt = clamp_sentence(sanitize_bullet_text(item), MAX_BULLET_CHARS)
            if not txt or is_presenter_directive(txt):
                continue
            if txt not in out:
                out.append(txt)
            if len(out) >= max_items:
                break

    if len(out) < 3:
        for item in materials:
            txt = clamp_sentence(sanitize_bullet_text(item), MAX_BULLET_CHARS)
            if not txt or is_presenter_directive(txt):
                continue
            if txt not in out:
                out.append(txt)
            if len(out) >= max_items:
                break

    return out


def has_non_base_r_dependencies(code: str) -> bool:
    low = str(code or "").lower()
    patterns = [
        "library(",
        "require(",
        "pacman::",
        "ggplot(",
        "geom_",
        "theme_",
        "%>%",
        "dplyr::",
        "tidyr::",
        "readr::",
        "forcats::",
        "tibble(",
        "read_csv(",
    ]
    return any(p in low for p in patterns)


def base_r_fallback_chunk(label: str) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9_ -]", "", str(label or "simulation")).strip() or "simulation"
    return "\n".join(
        [
            "set.seed(1234)",
            "x <- rnorm(300)",
            f"hist(x, breaks = 24, col = 'lightblue', main = 'Base R Plot: {safe_label}', xlab = 'Value')",
            "abline(v = mean(x), col = 'red', lwd = 2)",
            "c(mean = mean(x), sd = sd(x))",
        ]
    )


def sanitize_r_chunk(code: str, label: str) -> str:
    raw = str(code or "").strip()
    if not raw:
        return base_r_fallback_chunk(label)

    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        striped = line.strip().lower()
        if striped.startswith("library(") or striped.startswith("require(") or "pacman::p_load" in striped:
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        return base_r_fallback_chunk(label)
    if has_non_base_r_dependencies(cleaned):
        return base_r_fallback_chunk(label)
    return cleaned


def normalize_equation_tokens(expr: str) -> str:
    out = str(expr or "")
    replacements = {
        "∑": r"\sum",
        "μ": r"\mu",
        "σ": r"\sigma",
        "β": r"\beta",
        "α": r"\alpha",
        "γ": r"\gamma",
        "λ": r"\lambda",
        "≤": r"\le",
        "≥": r"\ge",
        "≠": r"\ne",
        "∞": r"\infty",
        "∈": r"\in",
        "−": "-",
        "–": "-",
        "×": r"\times",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = re.sub(r"\|\|\s*([^|]+?)\s*\|\|", r"\\lVert \1 \\rVert", out)
    out = re.sub(r"\bargmin\b", r"\\arg\\min", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def is_formula_like_line(line: str) -> bool:
    low = str(line or "").lower()
    if not low:
        return False
    strong_markers = [
        r"\sum",
        r"\frac",
        r"\mu",
        r"\sigma",
        r"\alpha",
        r"\beta",
        r"\gamma",
        r"\lVert",
        r"\arg\min",
        "=",
        "^",
        "_",
    ]
    return any(marker in low for marker in strong_markers)


def parse_equation_payload(raw_equation: str) -> tuple[list[str], list[str]]:
    raw = str(raw_equation or "").strip()
    if not raw:
        return [], []

    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^\$\$\s*", "", text)
    text = re.sub(r"\s*\$\$$", "", text)

    formula_steps: list[str] = []
    notes: list[str] = []

    inline_math = re.findall(r"\\\((.+?)\\\)", text)
    for segment in inline_math:
        normalized = normalize_equation_tokens(segment)
        if normalized and normalized not in formula_steps:
            formula_steps.append(normalized)
    text_no_inline = re.sub(r"\\\(.+?\\\)", " ", text)

    for raw_line in text_no_inline.split("\n"):
        line = sanitize_text(raw_line)
        line = re.sub(r"^\d+\)\s*", "", line).strip()
        if not line:
            continue
        line_low = line.lower()
        if line_low.startswith("algorithm steps") or line_low.startswith("minimize over assignments"):
            notes.append(line)
            continue
        if ":" in line:
            prefix, suffix = line.split(":", 1)
            if suffix.strip() and is_formula_like_line(normalize_equation_tokens(suffix)) and not is_formula_like_line(normalize_equation_tokens(prefix)):
                line = suffix.strip()
        normalized = normalize_equation_tokens(line)
        if is_formula_like_line(normalized):
            if normalized not in formula_steps:
                formula_steps.append(normalized)
        else:
            notes.append(line)

    cleaned_steps: list[str] = []
    for step in formula_steps:
        trimmed = step.strip().strip(",;:")
        if trimmed:
            cleaned_steps.append(trimmed)
    return cleaned_steps, notes


def extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")

    snippet = cleaned[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON output was not an object.")
    return parsed


def normalize_sections(raw_title: str, raw_slides: object, fallback_title: str, target_count: int) -> tuple[str, list[dict[str, object]]]:
    title = sanitize_text(raw_title) or fallback_title
    sections: list[dict[str, object]] = []

    if isinstance(raw_slides, list):
        for idx, item in enumerate(raw_slides):
            if not isinstance(item, dict):
                continue
            slide_title = sanitize_text(item.get("title", "")) or f"{title} ({idx + 1})"
            subtitle = polish_sentence_field(item.get("subtitle", ""), MAX_SUBTITLE_CHARS)
            bullets_raw = item.get("bullets", [])
            bullets: list[str] = []
            if isinstance(bullets_raw, list):
                for bullet in bullets_raw:
                    txt = sanitize_bullet_text(bullet)
                    txt = clamp_sentence(txt, MAX_BULLET_CHARS)
                    if txt and not is_presenter_directive(txt):
                        bullets.append(txt)
            layout = sanitize_text(item.get("layout", "concept")).lower() or "concept"
            if layout not in {"title", "concept", "formula", "simulation", "example", "activity", "summary"}:
                layout = "concept"
            definition = polish_sentence_field(item.get("definition", ""), 96)
            context = polish_sentence_field(item.get("context", ""), 96)
            materials = normalize_material_list(
                item.get("studentMaterials", item.get("materials", [])),
                max_items=3,
            )
            example = polish_sentence_field(item.get("example", ""), MAX_EXAMPLE_CHARS)
            activity = polish_sentence_field(item.get("activity", ""), MAX_ACTIVITY_CHARS)
            equation = str(item.get("equation", "")).strip()
            notes = polish_sentence_field(item.get("notes", ""), MAX_NOTES_CHARS)
            r_chunk = str(item.get("rChunk", "")).strip()
            figure_path = str(item.get("figurePath", "")).strip()
            if is_unhelpful_source_sentence(notes):
                notes = ""
            if is_presenter_directive(notes):
                notes = ""

            has_content = bool(bullets or definition or context or materials or example or activity or equation or r_chunk)
            if not has_content:
                continue
            if len(bullets) < 2 and layout not in {"simulation", "formula"}:
                bullets = infer_bullets(f"{title}\n{slide_title}")[:3]
            bullets = compose_student_bullets(definition, context, materials, bullets, max_items=6)

            sections.append(
                {
                    "title": slide_title,
                    "subtitle": subtitle,
                    "bullets": bullets[:6],
                    "layout": layout,
                    "definition": definition,
                    "context": context,
                    "studentMaterials": materials,
                    "example": example,
                    "activity": activity,
                    "equation": equation,
                    "notes": notes,
                    "rChunk": r_chunk,
                    "figurePath": figure_path,
                }
            )

    if len(sections) >= 3:
        return title, sections[:MAX_SLIDE_COUNT]

    fallback = make_slide_sections(fallback_title, fallback_title, source_excerpt="")
    if target_count:
        fallback = fallback[:target_count]
    return fallback_title, fallback


def sanitize_slide_indexes(raw: object, max_len: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for value in raw:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < max_len:
            out.append(idx)
    return sorted(set(out))


def normalize_slide_obj(raw: object, idx: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {
            "title": f"Slide {idx + 1}",
            "subtitle": "",
            "bullets": ["Add key idea.", "Add supporting example."],
            "layout": "concept",
            "definition": "",
            "context": "",
            "studentMaterials": [],
            "example": "",
            "activity": "",
            "equation": "",
            "notes": "",
            "rChunk": "",
            "figurePath": "",
        }

    title = sanitize_text(raw.get("title", "")) or f"Slide {idx + 1}"
    subtitle = polish_sentence_field(raw.get("subtitle", ""), MAX_SUBTITLE_CHARS)
    layout = sanitize_text(raw.get("layout", "concept")).lower() or "concept"
    if layout not in {"title", "concept", "formula", "simulation", "example", "activity", "summary"}:
        layout = "concept"
    definition = polish_sentence_field(raw.get("definition", ""), 96)
    context = polish_sentence_field(raw.get("context", ""), 96)
    materials = normalize_material_list(raw.get("studentMaterials", raw.get("materials", [])), max_items=3)
    example = polish_sentence_field(raw.get("example", ""), MAX_EXAMPLE_CHARS)
    activity = polish_sentence_field(raw.get("activity", ""), MAX_ACTIVITY_CHARS)
    equation = str(raw.get("equation", "")).strip()
    notes = polish_sentence_field(raw.get("notes", ""), MAX_NOTES_CHARS)
    r_chunk = str(raw.get("rChunk", "")).strip()
    figure_path = str(raw.get("figurePath", "")).strip()
    bullets_raw = raw.get("bullets", [])
    bullets: list[str] = []
    if isinstance(bullets_raw, list):
        for b in bullets_raw:
            text = sanitize_bullet_text(b)
            text = clamp_sentence(text, MAX_BULLET_CHARS)
            if text and not is_presenter_directive(text):
                bullets.append(text)
    if len(bullets) < 2 and layout not in {"simulation", "formula"}:
        bullets = ["Add key idea.", "Add supporting example."]
    bullets = compose_student_bullets(definition, context, materials, bullets, max_items=6)
    if is_unhelpful_source_sentence(notes):
        notes = ""
    if is_presenter_directive(notes):
        notes = ""
    return {
        "title": title,
        "subtitle": subtitle,
        "bullets": bullets[:6],
        "layout": layout,
        "definition": definition,
        "context": context,
        "studentMaterials": materials,
        "example": example,
        "activity": activity,
        "equation": equation,
        "notes": notes,
        "rChunk": r_chunk,
        "figurePath": figure_path,
    }


def split_text_into_chunks(text: str, max_chars: int, max_chunks: int = 4) -> list[str]:
    value = sanitize_text(text)
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]

    sentences = re.split(r"(?<=[.!?])\s+", value)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sent = sanitize_text(sentence)
        if not sent:
            continue
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            if len(chunks) >= max_chunks:
                return chunks[:max_chunks]
        if len(sent) <= max_chars:
            current = sent
        else:
            words = sent.split()
            block = ""
            for word in words:
                try_block = f"{block} {word}".strip() if block else word
                if len(try_block) <= max_chars:
                    block = try_block
                else:
                    if block:
                        chunks.append(block)
                        if len(chunks) >= max_chunks:
                            return chunks[:max_chunks]
                    block = word
            current = block
    if current and len(chunks) < max_chunks:
        chunks.append(current)
    return chunks[:max_chunks]


def slide_layout_limits(layout: str) -> dict[str, int]:
    key = str(layout or "concept").lower()
    base = {"max_bullets": 4, "max_bullet_chars_total": 420, "max_aux_chars": 170, "max_equation_steps": 3, "max_r_lines": 14}
    if key == "title":
        return {**base, "max_bullets": 3, "max_bullet_chars_total": 340, "max_aux_chars": 120}
    if key == "formula":
        return {**base, "max_bullets": 4, "max_bullet_chars_total": 340, "max_aux_chars": 140, "max_equation_steps": 2}
    if key == "simulation":
        return {**base, "max_bullets": 4, "max_bullet_chars_total": 320, "max_aux_chars": 140, "max_r_lines": 12}
    if key == "example":
        return {**base, "max_bullets": 4, "max_bullet_chars_total": 360, "max_aux_chars": 155}
    if key == "activity":
        return {**base, "max_bullets": 4, "max_bullet_chars_total": 340, "max_aux_chars": 145}
    if key == "summary":
        return {**base, "max_bullets": 4, "max_bullet_chars_total": 360, "max_aux_chars": 150}
    return base


def rendered_bullet_limit(layout: str) -> int:
    key = str(layout or "concept").lower()
    if key == "title":
        return 3
    return 4


def split_code_into_chunks(code: str, max_lines: int) -> list[str]:
    lines = [line.rstrip() for line in str(code or "").splitlines()]
    lines = [line for line in lines if line.strip() or line == ""]
    if not lines:
        return []
    chunks: list[str] = []
    for start in range(0, len(lines), max_lines):
        block = "\n".join(lines[start : start + max_lines]).strip()
        if block:
            chunks.append(block)
    return chunks


def continuation_title(base_title: str, continuation_idx: int) -> str:
    title = sanitize_text(base_title)
    if not title:
        title = "Slide"
    low = title.lower()
    if "(cont" in low:
        return title
    if continuation_idx <= 1:
        return f"{title} (cont.)"
    return f"{title} (cont. {continuation_idx})"


def split_slide_for_readability(raw_slide: object, idx: int) -> list[dict[str, object]]:
    slide = normalize_slide_obj(raw_slide, idx)
    layout = str(slide.get("layout", "concept")).lower()
    limits = slide_layout_limits(layout)
    max_bullets = rendered_bullet_limit(layout)
    max_aux_chars = limits["max_aux_chars"]
    max_eq_steps = max(1, limits["max_equation_steps"])
    max_r_lines = max(6, limits["max_r_lines"])

    first = dict(slide)
    first["bullets"] = list(first.get("bullets", []))[:max_bullets]
    continuation_slides: list[dict[str, object]] = []

    if layout == "example":
        example_text = str(first.get("example", "")).strip()
        chunks = split_text_into_chunks(example_text, max_aux_chars, max_chunks=4)
        if len(chunks) > 1:
            first["example"] = chunks[0]
            for chunk in chunks[1:]:
                continuation_slides.append(
                    {
                        "layout": "example",
                        "subtitle": "Worked example continuation",
                        "bullets": [
                            "Continue the worked example using the same interpretation framework.",
                            "Focus on what changes in the setup and what stays fixed.",
                        ],
                        "example": chunk,
                    }
                )

    if layout == "activity":
        activity_text = str(first.get("activity", "")).strip()
        chunks = split_text_into_chunks(activity_text, max_aux_chars, max_chunks=4)
        if len(chunks) > 1:
            first["activity"] = chunks[0]
            for chunk in chunks[1:]:
                continuation_slides.append(
                    {
                        "layout": "activity",
                        "subtitle": "Activity continuation",
                        "bullets": [
                            "Continue the activity steps and keep the decision rule explicit.",
                            "Check whether each group can justify its conclusion with evidence.",
                        ],
                        "activity": chunk,
                    }
                )

    equation_text = str(first.get("equation", "")).strip()
    if equation_text:
        steps, _ = parse_equation_payload(equation_text)
        if len(steps) > max_eq_steps:
            first["equation"] = "\n".join(steps[:max_eq_steps])
            remaining = steps[max_eq_steps:]
            continuation_slides.append(
                {
                    "layout": "formula",
                    "subtitle": "Formula continuation",
                    "bullets": [
                        "Continue equation steps and tie each symbol to interpretation.",
                        "Highlight the decision implication after each transformation.",
                    ],
                    "equation": "\n".join(remaining),
                }
            )

    code_text = str(first.get("rChunk", "")).strip()
    if code_text:
        code_chunks = split_code_into_chunks(code_text, max_r_lines)
        if len(code_chunks) > 1:
            carry_layout = "simulation"
            start_idx = 1
            if layout != "example":
                first["rChunk"] = code_chunks[0]
            else:
                first["rChunk"] = ""
                start_idx = 0
            for block in code_chunks[start_idx:]:
                continuation_slides.append(
                    {
                        "layout": carry_layout,
                        "subtitle": "Code continuation",
                        "bullets": [
                            "Continue code execution and verify intermediate results.",
                            "Interpret output before moving to the next code block.",
                        ],
                        "rChunk": block,
                    }
                )

    if not continuation_slides:
        return [normalize_slide_obj(first, idx)]

    split_slides: list[dict[str, object]] = [normalize_slide_obj(first, idx)]
    base_title = str(first.get("title", ""))
    for cont_idx, partial in enumerate(continuation_slides, start=1):
        cont = {
            "title": continuation_title(base_title, cont_idx),
            "subtitle": str(partial.get("subtitle", "Continuation")),
            "layout": str(partial.get("layout", "concept")),
            "bullets": list(partial.get("bullets", []))[:max_bullets],
            "definition": "",
            "context": "",
            "studentMaterials": [],
            "example": str(partial.get("example", "")),
            "activity": str(partial.get("activity", "")),
            "equation": str(partial.get("equation", "")),
            "notes": "",
            "rChunk": str(partial.get("rChunk", "")),
            "figurePath": "",
        }
        split_slides.append(normalize_slide_obj(cont, idx + cont_idx))

    return split_slides


def rebalance_slides_for_readability(slides: list[dict[str, object]], max_total: int) -> list[dict[str, object]]:
    if max_total <= 0:
        return []
    out: list[dict[str, object]] = []
    for raw_slide in slides:
        if len(out) >= max_total:
            break
        parts = split_slide_for_readability(raw_slide, len(out))
        for part in parts:
            if len(out) >= max_total:
                break
            out.append(normalize_slide_obj(part, len(out)))
    return out[:max_total]


def fallback_slide_from_template(
    deck_title: str,
    template_slide: dict[str, object] | object,
    slide_idx: int,
) -> dict[str, object]:
    if isinstance(template_slide, dict):
        template_title = sanitize_text(template_slide.get("title", ""))
        template_layout = sanitize_text(template_slide.get("layout", "concept")).lower()
        purpose = sanitize_text(template_slide.get("purpose", ""))
    else:
        template_title = ""
        template_layout = "concept"
        purpose = ""

    layout = template_layout if template_layout in {"title", "concept", "formula", "simulation", "example", "activity", "summary"} else "concept"
    title = template_title or f"{deck_title}: Part {slide_idx + 1}"
    subtitle = purpose or "StatEdu generated teaching slide"
    seed = f"{deck_title}\n{title}\n{purpose}"
    bullets = infer_bullets(seed)[:4]

    definition = ""
    context = ""
    equation = ""
    example = ""
    activity = ""
    r_chunk = ""
    if layout == "formula":
        definition = "This formula summarizes how the method quantifies fit or evidence."
        context = "Interpret symbols before using the formula in a worked example."
        equation = r"\text{Metric}=\sum_{i=1}^{n}(\text{observed}_i-\text{model}_i)^2"
    elif layout == "simulation":
        definition = "Simulation approximates repeated sampling behavior under clear assumptions."
        context = "Use simulation to build intuition before formal derivations."
        r_chunk = base_r_fallback_chunk(title)
    elif layout == "activity":
        definition = "Activity asks students to make a statistical decision and justify it."
        context = "Use pair discussion, then compare reasoning across groups."
        activity = "Mini activity: compute one value, interpret it, and defend your choice."
    elif layout == "example":
        definition = "Worked example translates a word problem into a statistical workflow."
        context = "Show setup, calculation, and interpretation in one coherent sequence."
        example = "Walk through one concrete example step-by-step and interpret the result."
    elif layout == "summary":
        definition = "Summary consolidates key definitions, assumptions, and decisions."
        context = "End with one short check-for-understanding prompt."
    else:
        definition = "State the core concept in plain language first."
        context = "Connect the concept to one realistic data-analysis scenario."

    return normalize_slide_obj(
        {
            "title": title,
            "subtitle": subtitle,
            "layout": layout,
            "bullets": bullets,
            "definition": definition,
            "context": context,
            "studentMaterials": ["One-page checklist", "Worked example prompt"],
            "equation": equation,
            "example": example,
            "activity": activity,
            "rChunk": r_chunk,
        },
        slide_idx,
    )


def align_slide_count(
    slides: list[dict[str, object]],
    target_count: int,
    previous_slides: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if target_count <= 0:
        return slides

    normalized = [normalize_slide_obj(s, i) for i, s in enumerate(slides[:target_count])]
    prev = previous_slides or []

    while len(normalized) < target_count:
        idx = len(normalized)
        if idx < len(prev):
            normalized.append(normalize_slide_obj(prev[idx], idx))
        elif prev:
            fallback = normalize_slide_obj(prev[-1], idx)
            fallback["title"] = f"{fallback['title']} ({idx + 1})"
            normalized.append(fallback)
        else:
            normalized.append(normalize_slide_obj({}, idx))

    return normalized


def merge_protected_slides(
    previous_slides: list[dict[str, object]] | None,
    generated_slides: list[dict[str, object]],
    protected_indexes: list[int],
    target_count: int,
) -> list[dict[str, object]]:
    prev = previous_slides or []
    merged = align_slide_count(generated_slides, target_count, prev)
    prev_norm = align_slide_count(prev, target_count, prev)
    locked_set = set(protected_indexes)

    for idx in locked_set:
        if idx < len(prev_norm):
            merged[idx] = prev_norm[idx]
    return merged


def create_job(kind: str, payload: dict[str, object] | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "state": "queued",
            "progress": 0.0,
            "currentSlide": 0,
            "totalSlides": 0,
            "stage": "queued",
            "error": None,
            "deck": None,
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
            "payload": payload or {},
        }
    return job_id


def update_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            job[key] = value
        job["updatedAt"] = now_iso()


def get_job(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return json.loads(json.dumps(job))


def parse_qmd_title_and_slides(qmd: str, fallback_title: str) -> tuple[str, list[dict[str, object]]]:
    text = qmd or ""
    title = fallback_title
    body = text

    fm = re.match(r"(?s)\A---\n(.*?)\n---\n?", text)
    if fm:
        front = fm.group(1)
        body = text[fm.end() :]
        m_title = re.search(r'(?m)^\s*title:\s*"?(.+?)"?\s*$', front)
        if m_title:
            candidate = m_title.group(1).strip()
            if candidate:
                title = candidate

    heading_matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", body))
    slides: list[dict[str, object]] = []
    for i, m_h2 in enumerate(heading_matches):
        slide_title_raw = m_h2.group(1).strip()
        slide_title = re.sub(r"\s+\{[^{}]*\}\s*$", "", slide_title_raw).strip()
        m_bg = re.search(r'background-image="([^"]+)"', slide_title_raw)
        figure_path = m_bg.group(1).strip() if m_bg else ""
        start = m_h2.end()
        end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(body)
        block = body[start:end].strip()

        subtitle = ""
        bullets: list[str] = []
        equation = ""
        notes = ""
        r_lines: list[str] = []
        in_code = False
        in_math = False
        for raw_line in block.splitlines():
            striped = raw_line.strip()
            if striped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                if len(r_lines) < 60:
                    r_lines.append(raw_line)
                continue
            if striped.startswith("$$"):
                in_math = not in_math
                continue
            if in_math:
                if not equation and striped:
                    equation = striped
                continue
            if not subtitle and striped.startswith("*") and striped.endswith("*") and len(striped) > 2:
                subtitle = striped.strip("*").strip()
                continue
            if not notes and striped.startswith("> "):
                notes = striped[2:].strip()
                continue
            if re.match(r"^[-+*]\s+", striped):
                bullet = re.sub(r"^[-+*]\s+", "", striped).strip()
                bullet = re.sub(r"<[^>]*>", "", bullet)
                bullet = sanitize_bullet_text(bullet)
                if bullet:
                    bullets.append(bullet)

        layout = "concept"
        low_title = slide_title.lower()
        if "simulation" in low_title:
            layout = "simulation"
        elif "formula" in low_title:
            layout = "formula"
        elif "activity" in low_title:
            layout = "activity"
        elif "summary" in low_title or "exit ticket" in low_title:
            layout = "summary"
        slides.append(
            normalize_slide_obj(
                {
                    "title": slide_title or f"Slide {i + 1}",
                    "subtitle": subtitle,
                    "bullets": bullets or ["Add key idea.", "Add supporting example."],
                    "layout": layout,
                    "equation": equation,
                    "notes": notes,
                    "rChunk": "\n".join(r_lines).strip(),
                    "figurePath": figure_path,
                },
                i,
            )
        )

    return title, slides


def render_deck_incrementally(job_id: str, deck: dict[str, object]) -> dict[str, object]:
    slides = deck.get("slides", [])
    if not isinstance(slides, list) or not slides:
        raise RuntimeError("No slides available to render.")

    total = len(slides)
    update_job(job_id, totalSlides=total, stage="rendering", progress=0.90)

    final_deck = deck
    for idx in range(1, total + 1):
        staged = dict(deck)
        staged_slides = slides[:idx]
        staged["slides"] = staged_slides
        staged["bullets"] = staged_slides[0].get("bullets", []) if staged_slides else []
        staged["qmd"] = build_qmd(str(staged.get("title", "Generated Deck")), staged_slides, str(staged.get("animation", "step")))
        staged["updatedAt"] = now_iso()
        save_deck(staged)

        if str(staged.get("format", "quarto")) == "quarto":
            render_url, warning = render_quarto(str(staged.get("id", "")))
            staged["renderUrl"] = render_url
            staged["renderWarning"] = warning
            save_deck(staged)

        final_deck = staged
        update_job(
            job_id,
            state="running",
            stage=f"rendering_slide_{idx}",
            currentSlide=idx,
            totalSlides=total,
            progress=round(0.90 + (idx / total) * 0.10, 4),
            deck=final_deck,
        )
        if idx < total and RENDER_STEP_DELAY_SEC > 0:
            time.sleep(RENDER_STEP_DELAY_SEC)

    return final_deck


def run_generation_job(job_id: str, deck_params: dict[str, object]) -> None:
    try:
        requested_raw = int(deck_params.get("requested_slide_count", 0) or 0)
        if requested_raw > 0:
            estimated_total = clamp_slide_count(requested_raw)
        else:
            prev_slides = deck_params.get("previous_slides", None)
            if isinstance(prev_slides, list) and prev_slides:
                estimated_total = clamp_slide_count(len(prev_slides))
            else:
                estimated_total = infer_slide_count(str(deck_params.get("prompt", "")))

        update_job(job_id, state="running", stage="generating", progress=0.05, totalSlides=estimated_total, currentSlide=0)

        def progress_cb(stage: str, progress: float) -> None:
            updates: dict[str, object] = {
                "state": "running",
                "stage": stage,
                "progress": max(0.0, min(0.9, float(progress))),
                "totalSlides": estimated_total,
            }
            if str(stage).startswith("content_slide_"):
                try:
                    idx = int(str(stage).replace("content_slide_", ""))
                except ValueError:
                    idx = 0
                if idx > 0:
                    updates["currentSlide"] = min(idx, estimated_total)
            if str(stage).startswith("image_slide_"):
                try:
                    idx = int(str(stage).replace("image_slide_", ""))
                except ValueError:
                    idx = 0
                if idx > 0:
                    updates["currentSlide"] = min(idx, estimated_total)
            update_job(job_id, **updates)
        raw_deck_id = deck_params.get("deck_id")
        deck_id = None
        if raw_deck_id is not None and str(raw_deck_id).strip():
            deck_id = str(raw_deck_id).strip()
        deck = create_deck(
            prompt=str(deck_params.get("prompt", "")),
            output_format=str(deck_params.get("output_format", "quarto")),
            animation=str(deck_params.get("animation", "step")),
            teaching_style=str(deck_params.get("teaching_style", "balanced")),
            revision=int(deck_params.get("revision", 1)),
            source_name=str(deck_params.get("source_name", "")),
            source_excerpt=str(deck_params.get("source_excerpt", "")),
            feedback=str(deck_params.get("feedback", "")),
            previous_slides=deck_params.get("previous_slides", None),
            locked_slide_indexes=deck_params.get("locked_slide_indexes", []),
            approved_slide_indexes=deck_params.get("approved_slide_indexes", []),
            requested_slide_count=int(deck_params.get("requested_slide_count", 0) or 0) or None,
            render_now=False,
            deck_id=deck_id,
            progress_cb=progress_cb,
        )
        if "sourceType" in deck_params:
            deck["sourceType"] = str(deck_params.get("sourceType", "none"))
            save_deck(deck)
        final_deck = render_deck_incrementally(job_id, deck)
        update_job(
            job_id,
            state="completed",
            stage="completed",
            progress=1.0,
            currentSlide=int(final_deck.get("requestedSlideCount", len(final_deck.get("slides", [])))),
            totalSlides=int(final_deck.get("requestedSlideCount", len(final_deck.get("slides", [])))),
            deck=final_deck,
        )
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, state="error", stage="error", error=str(exc))


def post_json(url: str, payload: dict[str, object], headers: dict[str, str] | None = None) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    max_attempts = 1 + max(0, int(LLM_RETRY_COUNT))
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        req = urlrequest.Request(url, data=body, headers=req_headers, method="POST")
        try:
            if LLM_TIMEOUT_SEC is None:
                resp_ctx = urlrequest.urlopen(req)
            else:
                resp_ctx = urlrequest.urlopen(req, timeout=LLM_TIMEOUT_SEC)
            with resp_ctx as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                parsed = json.loads(raw or "{}")
                if not isinstance(parsed, dict):
                    raise RuntimeError("Non-object JSON response from model API.")
                return parsed
        except urlerror.HTTPError as exc:
            last_error = exc
            detail = exc.read().decode("utf-8", errors="ignore")
            retriable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if retriable and attempt < max_attempts - 1:
                time.sleep(max(0.1, LLM_RETRY_BACKOFF_SEC) * (attempt + 1))
                continue
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:300]}") from exc
        except (urlerror.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(max(0.1, LLM_RETRY_BACKOFF_SEC) * (attempt + 1))
                continue
            raise RuntimeError(f"LLM network error: {exc}") from exc

    raise RuntimeError(f"LLM request failed after {max_attempts} attempt(s): {last_error}")


def call_openai_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    key = resolve_llm_key("openai")
    if not key:
        raise RuntimeError("Missing OpenAI API key (`OPENAI_API_KEY` or `STATEDU_OPENAI_API_KEY`).")

    payload = {
        "model": resolve_llm_model("openai"),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = post_json(
        "https://api.openai.com/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI response missing choices.")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return extract_json_object(str(content))


def call_gemini_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    key = resolve_llm_key("gemini")
    if not key:
        raise RuntimeError("Missing Gemini API key (`GOOGLE_API_KEY` or `STATEDU_GEMINI_API_KEY`).")

    model = resolve_llm_model("gemini")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    data = post_json(url, payload)
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini response missing candidates.")
    content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
    parts = content.get("parts", []) if isinstance(content, dict) else []
    text = ""
    if isinstance(parts, list):
        text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    return extract_json_object(text)


def call_anthropic_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    key = resolve_llm_key("anthropic")
    if not key:
        raise RuntimeError("Missing Anthropic API key (`ANTHROPIC_API_KEY` or `STATEDU_ANTHROPIC_API_KEY`).")

    payload = {
        "model": resolve_llm_model("anthropic"),
        "max_tokens": 2600,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    data = post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    content = data.get("content", [])
    if not isinstance(content, list) or not content:
        raise RuntimeError("Anthropic response missing content.")
    text_parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
    return extract_json_object("\n".join(text_parts))


def call_provider_json(provider: str, system_prompt: str, user_prompt: str) -> dict[str, object]:
    if provider == "openai":
        return call_openai_json(system_prompt, user_prompt)
    if provider == "gemini":
        return call_gemini_json(system_prompt, user_prompt)
    if provider == "anthropic":
        return call_anthropic_json(system_prompt, user_prompt)
    raise RuntimeError(f"Unsupported STATEDU_LLM_PROVIDER: {provider}")


def maybe_generate_with_llm(
    *,
    prompt: str,
    source_excerpt: str,
    feedback: str,
    output_format: str,
    animation: str,
    teaching_style: str,
    target_count: int,
    previous_slides: list[dict[str, object]] | None,
    progress_cb: Callable[[str, float], None] | None = None,
) -> tuple[str, list[dict[str, object]], bool, str | None, str, list[dict[str, str]]]:
    provider = resolve_llm_provider()
    model = resolve_llm_model(provider)
    fallback_title = infer_title(prompt)
    source_excerpt_short = source_excerpt[:7000] if source_excerpt else ""
    stage_warnings: list[str] = []

    if progress_cb:
        progress_cb("web_research", 0.12)
    web_research, research_warning = collect_web_research(prompt, source_excerpt_short, WEB_RESEARCH_MAX_RESULTS)
    research_context = format_research_context(web_research)
    if research_warning:
        stage_warnings.append(research_warning)

    if provider in {"mock", "none", "off"}:
        if progress_cb:
            progress_cb("fallback_generation", 0.45)
        merged_excerpt = source_excerpt
        if research_context:
            merged_excerpt = (merged_excerpt + "\n\nOnline research notes:\n" + research_context).strip()
        warning = "; ".join(stage_warnings) if stage_warnings else None
        return (
            fallback_title,
            make_slide_sections(
                fallback_title,
                prompt + "\n" + feedback,
                merged_excerpt,
                teaching_style=teaching_style,
            ),
            False,
            warning,
            provider,
            web_research,
        )

    previous_context = ""
    if previous_slides:
        slim = []
        for slide in previous_slides[:10]:
            if isinstance(slide, dict):
                slim.append(
                    {
                        "title": str(slide.get("title", "")),
                        "layout": str(slide.get("layout", "concept")),
                        "bullets": list(slide.get("bullets", []))[:4],
                        "equation": str(slide.get("equation", ""))[:220],
                        "hasRChunk": bool(str(slide.get("rChunk", "")).strip()),
                    }
                )
        previous_context = json.dumps(slim, ensure_ascii=True)

    if progress_cb:
        progress_cb("template_generation", 0.24)
    template_system = (
        "You are Agent 1: slide template planner for statistics education. "
        "Return strict JSON only with keys: title, template. "
        "template must be an array of 3-40 objects with keys: "
        "slideIndex (integer), title (string), layout (one of: title, concept, formula, simulation, example, activity, summary), "
        "purpose (string). "
        "Plan a coherent teaching progression."
    )
    template_user = (
        f"Primary prompt:\n{prompt.strip()}\n\n"
        f"Requested slide count: {target_count}\n"
        f"Teaching style profile: {teaching_style}\n"
        f"Output format target: {output_format}\n"
        "Audience: undergraduate statistics learners unless specified otherwise.\n"
    )
    if research_context:
        template_user += (
            "\nWeb research context (use this to plan a stronger progression and examples):\n"
            f"{research_context}\n"
        )
    if feedback:
        template_user += f"\nRefinement feedback:\n{feedback}\n"
    if previous_context:
        template_user += f"\nExisting slides context:\n{previous_context}\n"
    template_user += "\nReturn JSON only."

    template_raw = call_provider_json(provider, template_system, template_user)
    template_title = sanitize_text(template_raw.get("title", "")) or fallback_title
    template_slides = template_raw.get("template", template_raw.get("slides", []))
    if not isinstance(template_slides, list) or len(template_slides) < 3:
        raise RuntimeError(f"{provider}/{model} template stage returned invalid template.")

    if target_count:
        template_slides = template_slides[:target_count]
    template_count = len(template_slides)
    if template_count < 3:
        raise RuntimeError(f"{provider}/{model} template stage produced too few slides after count alignment.")

    if progress_cb:
        progress_cb("content_generation", 0.30)

    slide_system = (
        "You are Agent 2: single-slide writer for statistics education. "
        "Generate exactly one slide at a time. "
        "Return strict JSON only with key: slide. "
        "slide must be an object with keys: title, subtitle, layout, bullets (2-6), "
        "definition (string), context (string), studentMaterials (array of 1-3 strings), "
        "example (optional), activity (optional), equation (optional), rChunk (optional). "
        "Audience-facing slide copy only. No presenter coaching language. "
        "Do not write label prefixes like 'Definition:'/'Context:'/'Useful material:' inside bullets. "
        "Avoid ellipses and unfinished clauses; all bullet items must be complete sentences. "
        "For equation field, provide pure math expressions only (no prose inside equations). "
        "Do not add unrelated anecdotes or pop-culture references unless user explicitly asks. "
        "Use base R only in rChunk (no library()/require(), no tidyverse/ggplot dependencies)."
    )

    generated_slides: list[dict[str, object]] = []
    for idx, template_slide in enumerate(template_slides, start=1):
        if progress_cb:
            content_progress = 0.30 + (idx / template_count) * 0.28
            progress_cb(f"content_slide_{idx}", round(content_progress, 4))

        recent_context = []
        for prev_slide in generated_slides[-3:]:
            if isinstance(prev_slide, dict):
                recent_context.append(
                    {
                        "title": str(prev_slide.get("title", "")),
                        "layout": str(prev_slide.get("layout", "concept")),
                        "bullets": list(prev_slide.get("bullets", []))[:3],
                    }
                )
        slide_user = (
            f"Deck prompt:\n{prompt.strip()}\n\n"
            f"Deck title:\n{template_title}\n"
            f"Teaching style: {teaching_style}\n"
            f"Animation: {animation}\n"
            f"Slide number: {idx} of {template_count}\n"
            f"Template slide spec:\n{json.dumps(template_slide, ensure_ascii=True)}\n"
            f"Recent generated slides context:\n{json.dumps(recent_context, ensure_ascii=True)}\n"
            "Slide quality requirements:\n"
            "- Include definition, context, and useful student materials when pedagogically appropriate.\n"
            "- Keep bullets concise, grammatical, and readable on a projected slide.\n"
            "- Prefer complete statements over fragments.\n"
            "- End each bullet with proper punctuation.\n"
            "- For simulation/formula slides, include concrete statistical value.\n"
            "- Never output literal labels like 'Useful material:' in the visible bullet text.\n"
            "- Do not use ellipses (...); use full statements.\n"
        )
        if research_context:
            slide_user += (
                "\nWeb research facts to incorporate only if relevant:\n"
                f"{research_context}\n"
            )
        if source_excerpt_short:
            slide_user += f"\nSource excerpt:\n{source_excerpt_short}\n"
        if feedback:
            slide_user += f"\nRefinement feedback:\n{feedback}\n"
        if previous_context:
            slide_user += f"\nPrior deck context:\n{previous_context}\n"
        slide_user += "\nReturn JSON only."

        try:
            slide_raw = call_provider_json(provider, slide_system, slide_user)
            raw_slide_obj = slide_raw.get("slide", slide_raw)
            if not isinstance(raw_slide_obj, dict):
                raise RuntimeError("single-slide stage returned non-object slide")
            generated_slides.append(raw_slide_obj)
        except Exception as exc:  # noqa: BLE001
            stage_warnings.append(f"Slide {idx} fallback used ({sanitize_text(str(exc))[:140]}).")
            generated_slides.append(fallback_slide_from_template(template_title, template_slide, idx - 1))

    draft_title = template_title
    draft_slides = generated_slides

    if progress_cb:
        progress_cb("review_stage", 0.64)
    review_system = (
        "You are Agent 3: quality reviewer for statistics lecture slides. "
        "Return strict JSON only with keys: needsCorrection (boolean), issues (array), summary (string). "
        "Each issue must include: slideIndex (integer), severity (low|medium|high), problem (string), fix (string). "
        "Review for correctness, clarity, and whether each slide has definition/context/student material value. "
        "Flag irrelevant anecdotes or references unrelated to statistics learning goals. "
        "Flag incomplete/truncated sentence endings, ellipses, and any visible label-style bullet text. "
        "Flag if there is no visible plotting code in any rChunk."
    )
    review_user = (
        f"Prompt:\n{prompt.strip()}\n\n"
        f"Teaching style: {teaching_style}\n"
        f"Slides draft JSON:\n{json.dumps(draft_slides, ensure_ascii=True)}\n"
        "\nReturn JSON only."
    )
    if research_context:
        review_user += f"\nCheck alignment with these research notes when relevant:\n{research_context}\n"
    review_raw = call_provider_json(provider, review_system, review_user)
    review_issues_raw = review_raw.get("issues", [])
    review_issues = review_issues_raw if isinstance(review_issues_raw, list) else []

    fact_issues: list[object] = []
    fact_needs_correction = False
    if progress_cb:
        progress_cb("fact_check_stage", 0.72)
    fact_system = (
        "You are Agent 3B: fact-checker for statistics slides. "
        "Return strict JSON only with keys: needsCorrection (boolean), issues (array), summary (string). "
        "Each issue must include: slideIndex (integer), severity (low|medium|high), problem (string), fix (string). "
        "Check statistical correctness, formula meaning, interpretation accuracy, and whether claims overstate conclusions. "
        "If no factual problems, return needsCorrection false and empty issues."
    )
    fact_user = (
        f"Prompt:\n{prompt.strip()}\n\n"
        f"Slides draft JSON:\n{json.dumps(draft_slides, ensure_ascii=True)}\n"
    )
    if research_context:
        fact_user += f"\nReference context:\n{research_context}\n"
    if source_excerpt_short:
        fact_user += f"\nSource excerpt:\n{source_excerpt_short}\n"
    fact_user += "\nReturn JSON only."
    try:
        fact_raw = call_provider_json(provider, fact_system, fact_user)
        fact_needs_correction = bool(fact_raw.get("needsCorrection", False))
        fact_issues_raw = fact_raw.get("issues", [])
        if isinstance(fact_issues_raw, list):
            fact_issues = fact_issues_raw
    except Exception as exc:  # noqa: BLE001
        stage_warnings.append(f"Fact-check stage failed ({sanitize_text(str(exc))[:140]}).")

    issues: list[dict[str, object]] = []
    for issue in [*review_issues, *fact_issues]:
        if isinstance(issue, dict):
            issues.append(issue)
    needs_correction = (
        bool(review_raw.get("needsCorrection", False))
        or fact_needs_correction
        or (len(issues) > 0)
    )

    final_title = draft_title
    final_slides = draft_slides
    if needs_correction:
        if progress_cb:
            progress_cb("correction_stage", 0.80)
        correct_system = (
            "You are Agent 4: correction editor for slide decks. "
            "Return strict JSON only with keys: title, slides. "
            "Fix incomplete sentences, remove ellipses, and keep equation fields as math-only expressions. "
            "Remove label prefixes like 'Definition:'/'Context:'/'Useful material:' from bullets. "
            "Apply reviewer issues while preserving narrative flow and slide count."
        )
        correct_user = (
            f"Original prompt:\n{prompt.strip()}\n\n"
            f"Review issues:\n{json.dumps(issues, ensure_ascii=True)}\n"
            f"\nDraft slides:\n{json.dumps(draft_slides, ensure_ascii=True)}\n"
            "\nReturn corrected JSON only."
        )
        if research_context:
            correct_user += f"\nResearch notes:\n{research_context}\n"
        try:
            corrected_raw = call_provider_json(provider, correct_system, correct_user)
            corrected_title = sanitize_text(corrected_raw.get("title", "")) or draft_title
            corrected_slides = corrected_raw.get("slides", [])
            if isinstance(corrected_slides, list) and len(corrected_slides) >= 3:
                final_title = corrected_title
                final_slides = corrected_slides
            else:
                stage_warnings.append("Correction stage returned invalid slides; kept draft.")
        except Exception as exc:  # noqa: BLE001
            stage_warnings.append(f"Correction stage failed; kept draft ({exc}).")

    if progress_cb:
        progress_cb("finalizing", 0.86)
    title, sections = normalize_sections(
        final_title,
        final_slides,
        fallback_title=fallback_title,
        target_count=target_count,
    )
    if target_count:
        sections = sections[:target_count]

    if len(sections) < 3:
        sections = make_slide_sections(
            title,
            prompt + "\n" + feedback,
            source_excerpt=source_excerpt,
            teaching_style=teaching_style,
        )
        sections = sections[:target_count] if target_count else sections
        return (
            title,
            sections,
            False,
            f"{provider}/{model} pipeline returned too few slides; fallback applied.",
            provider,
            web_research,
        )

    if progress_cb:
        progress_cb("ready_for_render", 0.90)
    warning = "; ".join(stage_warnings) if stage_warnings else None
    return title, sections, True, warning, provider, web_research


def make_slide_sections(title: str, prompt: str, source_excerpt: str = "", teaching_style: str = "balanced") -> list[dict[str, object]]:
    count = infer_slide_count(prompt)
    style = normalize_teaching_style(teaching_style)
    merged_context = f"{title}\n{prompt}\n{source_excerpt}".lower()
    shared = infer_bullets(merged_context)

    formula = r"\bar{x} \pm t_{n-1,\alpha/2}\frac{s}{\sqrt{n}}"
    simulation_code = "\n".join(
        [
            "set.seed(1234)",
            "n <- 60",
            "xbar <- replicate(400, mean(rnorm(n, mean = 0, sd = 1)))",
            "hist(xbar, breaks = 25, col = 'lightblue', main = 'Sampling Distribution of xbar', xlab = 'xbar')",
            "abline(v = mean(xbar), col = 'red', lwd = 2)",
            "mean(xbar)",
        ]
    )
    example_text = "Compute the estimate, construct the interval, and interpret it in context."

    if "hypothesis" in merged_context or "p-value" in merged_context or "p value" in merged_context:
        formula = r"z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}, \quad p=P(|Z|\ge |z_{obs}|)"
        simulation_code = "\n".join(
            [
                "set.seed(42)",
                "n <- 40",
                "mu0 <- 0",
                "pvals <- replicate(500, {",
                "  x <- rnorm(n, mean = mu0, sd = 1)",
                "  z <- (mean(x) - mu0) / (sd(x) / sqrt(n))",
                "  2 * (1 - pnorm(abs(z)))",
                "})",
                "hist(pvals, breaks = 25, col = 'lightgreen', main = 'P-value Distribution Under H0', xlab = 'p-value')",
                "abline(v = 0.05, col = 'red', lwd = 2)",
                "mean(pvals < 0.05)",
            ]
        )
        example_text = "State H0 and H1, compute test statistic, then make the decision."
    elif "regression" in merged_context:
        formula = r"y_i = \beta_0 + \beta_1 x_i + \varepsilon_i"
        simulation_code = "\n".join(
            [
                "set.seed(2025)",
                "n <- 120",
                "x <- rnorm(n, 5, 2)",
                "y <- 2 + 1.4 * x + rnorm(n, 0, 1.5)",
                "fit <- lm(y ~ x)",
                "plot(x, y, pch = 19, col = rgb(0, 0, 1, 0.35), main = 'Regression Simulation', xlab = 'x', ylab = 'y')",
                "abline(fit, col = 'red', lwd = 2)",
                "summary(fit)$coefficients",
            ]
        )
        example_text = "Interpret slope as expected change in y for one-unit increase in x."
    elif "cluster" in merged_context or "k-means" in merged_context or "kmeans" in merged_context:
        formula = r"\text{WCSS}=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|^2"
        simulation_code = "\n".join(
            [
                "set.seed(7)",
                "x1 <- matrix(rnorm(120, mean = 0, sd = 0.6), ncol = 2)",
                "x2 <- matrix(rnorm(120, mean = 3, sd = 0.6), ncol = 2)",
                "x <- rbind(x1, x2)",
                "fit <- kmeans(x, centers = 2, nstart = 25)",
                "plot(x, col = fit$cluster, pch = 19, main = 'K-means Cluster Assignment', xlab = 'Feature 1', ylab = 'Feature 2')",
                "points(fit$centers, pch = 4, cex = 2.2, lwd = 2, col = 'black')",
                "fit$tot.withinss",
            ]
        )
        example_text = "Compare cluster assignments and discuss practical interpretability."
    elif "normal" in merged_context or "distribution" in merged_context:
        formula = r"Z=\frac{X-\mu}{\sigma}, \quad X\sim\mathcal{N}(\mu,\sigma^2)"
        simulation_code = "\n".join(
            [
                "set.seed(99)",
                "x <- rnorm(500, mean = 70, sd = 10)",
                "z <- (x - mean(x)) / sd(x)",
                "hist(x, prob = TRUE, breaks = 25, col = 'lavender', main = 'Normal Data Simulation', xlab = 'x')",
                "curve(dnorm(x, mean = 70, sd = 10), add = TRUE, col = 'blue', lwd = 2)",
                "summary(z)",
            ]
        )
        example_text = "Convert raw scores to z-scores and interpret relative position."

    if style == "conceptual":
        cycle = ["concept", "example", "activity", "concept", "example"]
    elif style == "mathematical":
        cycle = ["formula", "example", "formula", "concept", "activity"]
    elif style == "simulation":
        cycle = ["simulation", "example", "simulation", "concept", "activity"]
    else:
        cycle = ["example", "activity", "concept", "example", "activity"]

    if count <= 3:
        plan = ["title", "concept", "summary"]
    elif count == 4:
        if style == "conceptual":
            plan = ["title", "concept", "example", "summary"]
        elif style == "mathematical":
            plan = ["title", "formula", "example", "summary"]
        elif style == "simulation":
            plan = ["title", "simulation", "example", "summary"]
        else:
            plan = ["title", "formula", "simulation", "summary"]
    else:
        plan = ["title", "concept", "formula", "simulation"]
        idx = 0
        while len(plan) < count - 1:
            plan.append(cycle[idx % len(cycle)])
            idx += 1
        plan.append("summary")

    sections: list[dict[str, object]] = []
    for i, layout in enumerate(plan):
        slide_title = title if i == 0 else f"{title}: Part {i + 1}"
        subtitle = "StatEdu auto-generated lesson"
        bullets = shared.copy()
        definition = ""
        context = ""
        student_materials: list[str] = []
        activity = ""
        equation = ""
        notes = ""
        r_chunk = ""

        if layout == "title":
            slide_title = title
            subtitle = "Learning goals and roadmap"
            bullets = [
                f"Learning objective: {title}",
                "Motivation: where this appears in real statistical analysis.",
                "Roadmap: concept -> formula -> simulation -> interpretation.",
            ]
            definition = "Key terms and notation for this lesson."
            context = "Why this topic matters in real statistical decisions."
            student_materials = ["One-page glossary", "Roadmap checklist"]
            notes = "Open by asking students what decision this tool helps us make."
        elif layout == "concept":
            bullets = [
                shared[0],
                shared[1],
                "Name one assumption and one consequence if it is violated.",
            ]
            definition = "Core concept stated in plain language."
            context = "Connect the concept to a realistic classroom scenario."
            student_materials = ["Mini concept map", "Common misconception prompt"]
            notes = "Pause for a 30-second check-for-understanding before moving on."
        elif layout == "formula":
            slide_title = f"{title}: Core Formula"
            bullets = [
                "Define each symbol before using the formula.",
                "Link the formula to a decision rule students can apply.",
            ]
            definition = "Formula components and variable meanings."
            context = "When this formula is appropriate and when it is not."
            student_materials = ["Formula sheet", "Symbol legend"]
            equation = formula
            notes = "Have learners annotate the formula in pairs."
        elif layout == "simulation":
            slide_title = f"{title}: Simulation in R"
            bullets = [
                "Run a small simulation to observe sampling behavior.",
                "Interpret output in plain language, not only numbers.",
            ]
            definition = "Simulation approximates repeated sampling behavior."
            context = "Use simulation to build intuition before formal inference."
            student_materials = ["R starter code", "Parameter tweak challenge"]
            r_chunk = simulation_code
            notes = "Ask students to change one parameter (n or sd) and predict impact."
        elif layout == "example":
            slide_title = f"{title}: Worked Example"
            bullets = [
                "State the problem setup and known quantities.",
                "Compute step-by-step and justify each step.",
                "Interpret the result in domain language.",
            ]
            definition = "Identify the target quantity and inputs."
            context = "Translate a word problem into a statistical setup."
            student_materials = ["Worked solution template", "Checkpoint questions"]
            notes = "Cold-call for interpretation, not only calculation."
        elif layout == "activity":
            slide_title = f"{title}: Classroom Activity"
            bullets = [
                "Work in pairs for 3 minutes.",
                "Write one decision and one justification.",
                "Compare with a neighboring group.",
            ]
            definition = "Clarify the decision students must produce."
            context = "Activity mirrors a practical analysis decision."
            student_materials = ["Short worksheet", "Discussion rubric"]
            activity = "Mini-activity: solve one quick scenario and defend your conclusion."
            notes = "Debrief misconceptions explicitly after share-out."
        elif layout == "summary":
            slide_title = f"{title}: Summary and Exit Ticket"
            bullets = [
                "Summarize the key statistical takeaway in one sentence.",
                "State one common misconception and its correction.",
                "Exit ticket: apply the method to a new short prompt.",
            ]
            definition = "Most important definition to remember."
            context = "How this method connects to future topics."
            student_materials = ["Exit ticket prompt", "Practice set link"]
            activity = "Exit ticket: write a one-sentence decision and confidence level."
            notes = "Collect 2-3 responses and use them to start next class."

        if style == "conceptual":
            if layout in {"concept", "example"}:
                bullets = bullets[:2] + ["Use one concrete real-world scenario students recognize."]
            if layout == "formula":
                bullets = ["Only use formula as supporting language, not the center of the slide."]
        elif style == "mathematical":
            if layout in {"concept", "formula"}:
                bullets = bullets[:2] + ["State assumptions explicitly before applying the method."]
            if layout == "simulation":
                notes = "Use simulation to verify the derivation result."
        elif style == "simulation":
            if layout in {"simulation", "example"}:
                bullets = bullets[:2] + ["Vary one parameter and discuss how results change."]
            if layout == "formula":
                bullets = ["Present formula briefly, then move quickly to empirical demonstration."]

        sections.append(
            {
                "title": slide_title,
                "subtitle": subtitle,
                "bullets": compose_student_bullets(definition, context, student_materials, bullets, max_items=6),
                "layout": layout,
                "definition": definition,
                "context": context,
                "studentMaterials": student_materials,
                "example": example_text if layout == "example" else "",
                "activity": activity,
                "equation": equation,
                "notes": notes,
                "rChunk": r_chunk,
            }
        )

    return sections[:count]


def build_qmd(title: str, sections: list[dict[str, object]], animation: str) -> str:
    incremental = "true" if animation in {"step", "fade"} else "false"
    transition = "fade" if animation == "fade" else "slide"
    has_r_content = any(
        isinstance(raw, dict) and str(raw.get("rChunk", "")).strip()
        for raw in sections
    ) or any(
        isinstance(raw, dict) and str(raw.get("layout", "")).strip().lower() == "simulation"
        for raw in sections
    )

    lines = [
        "---",
        f'title: "{title}"',
        "format:",
        "  revealjs:",
        "    theme: [default]",
        "    width: 1366",
        "    height: 768",
        "    margin: 0.05",
        "    title-slide: false",
        "    center: false",
        "    scrollable: true",
        "    controls: true",
        "    progress: true",
        "    slide-number: true",
        "    navigation-mode: linear",
        "    css: style.css",
        f"    incremental: {incremental}",
        f"    transition: {transition}",
        "---",
        "",
    ]

    if has_r_content:
        lines.extend(
            [
                "```{r setup, include=FALSE}",
                "set.seed(1234)",
                "options(scipen = 4)",
                "```",
                "",
            ]
        )

    def esc(text: object) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def append_bullets(bullets: list[str], limit: int | None = None) -> None:
        seq = bullets if limit is None else bullets[:limit]
        for bullet in seq:
            txt = esc(sanitize_bullet_text(bullet))
            if txt:
                lines.append(f"- {txt}")

    def append_equation(eq: str) -> list[str]:
        steps, eq_notes = parse_equation_payload(eq)
        if not steps:
            return []
        if animation in {"step", "fade"} and len(steps) > 1:
            for step in steps:
                lines.append("::: {.fragment}")
                lines.append("$$")
                lines.append(step)
                lines.append("$$")
                lines.append(":::")
        else:
            lines.append("$$")
            lines.append(steps[0])
            lines.append("$$")
        return eq_notes

    def append_r_chunk(code: str, label: str) -> None:
        body = sanitize_r_chunk(code, label)
        if not body:
            return
        lines.append(f"```{{r {label}, echo=TRUE}}")
        lines.extend(body.splitlines())
        lines.append("```")

    def heading_for_slide(slide_title: str, figure_path: str) -> str:
        clean_title = str(slide_title or "").replace("{", "").replace("}", "")
        fig = str(figure_path or "").strip()
        if not fig:
            return f"## {clean_title}"
        return (
            f'## {clean_title} '
            f'{{background-image="{fig}" background-size="cover" background-position="right center" background-opacity="0.06"}}'
        )

    for idx, raw_slide in enumerate(sections):
        slide = normalize_slide_obj(raw_slide, idx)
        slide_title = str(slide["title"])
        subtitle = str(slide.get("subtitle", "")).strip()
        bullets = [str(x) for x in slide.get("bullets", [])]
        layout = str(slide.get("layout", "concept"))
        example = str(slide.get("example", "")).strip()
        activity = str(slide.get("activity", "")).strip()
        equation = str(slide.get("equation", "")).strip()
        r_chunk = str(slide.get("rChunk", "")).strip()
        figure_path = str(slide.get("figurePath", "")).strip()
        if layout == "simulation" and not r_chunk:
            r_chunk = "\n".join(
                [
                    "set.seed(1234)",
                    "x <- rnorm(300)",
                    "hist(x, breaks = 22, col = 'lightblue', main = 'Simulation Output', xlab = 'Value')",
                    "abline(v = mean(x), col = 'red', lwd = 2)",
                    "c(mean = mean(x), sd = sd(x))",
                ]
            )

        lines.append(heading_for_slide(slide_title, figure_path))
        if subtitle:
            lines.append("")
            lines.append(f"*{esc(subtitle)}*")
        lines.append("")

        if layout == "formula":
            lines.extend(
                [
                    ":::: {.columns}",
                    '::: {.column width="58%"}',
                    "### Interpretation Guide",
                ]
            )
            append_bullets(bullets, limit=4)
            lines.extend(
                [
                    ":::",
                    '::: {.column width="42%"}',
                    "### Formula",
                ]
            )
            eq_notes = append_equation(equation or r"\bar{x} \pm t_{n-1,\alpha/2}\frac{s}{\sqrt{n}}")
            if eq_notes:
                lines.append("")
                for note in eq_notes[:3]:
                    lines.append(f"- {esc(note)}")
            lines.extend([":::", "::::"])
        elif layout == "simulation":
            lines.extend(
                [
                    ":::: {.columns}",
                    '::: {.column width="50%"}',
                    "### Simulation Goal",
                ]
            )
            append_bullets(bullets, limit=4)
            lines.extend(
                [
                    ":::",
                    '::: {.column width="50%"}',
                    "### R Code",
                ]
            )
            append_r_chunk(r_chunk, f"sim_{idx + 1}")
            lines.extend([":::", "::::"])
        elif layout == "example":
            lines.extend(
                [
                    ":::: {.columns}",
                    '::: {.column width="56%"}',
                    "### Key Ideas",
                ]
            )
            append_bullets(bullets, limit=4)
            lines.extend(
                [
                    ":::",
                    '::: {.column width="44%"}',
                    "### Worked Example",
                ]
            )
            lines.append(esc(example or "Walk through one concrete example step-by-step and interpret the result."))
            if equation:
                lines.append("")
                eq_notes = append_equation(equation)
                if eq_notes:
                    lines.extend([f"- {esc(note)}" for note in eq_notes[:2]])
            if r_chunk:
                lines.append("")
                append_r_chunk(r_chunk, f"example_{idx + 1}")
            lines.extend([":::", "::::"])
        elif layout == "activity":
            lines.extend(
                [
                    ":::: {.columns}",
                    '::: {.column width="56%"}',
                    "### Discuss",
                ]
            )
            append_bullets(bullets, limit=4)
            lines.extend(
                [
                    ":::",
                    '::: {.column width="44%"}',
                    "### In-Class Activity",
                    esc(activity or "Pair up and solve one short question; compare reasoning with another group."),
                    ":::",
                    "::::",
                ]
            )
        elif layout == "summary":
            append_bullets(bullets, limit=4)
            lines.extend(
                [
                    "",
                    "### Exit Ticket",
                    esc(activity or "In one sentence: what is the key decision rule and why does it matter?"),
                ]
            )
        elif layout == "title":
            append_bullets(bullets, limit=3)
        else:
            append_bullets(bullets, limit=4)
            if equation:
                lines.append("")
                eq_notes = append_equation(equation)
                if eq_notes:
                    lines.extend([f"- {esc(note)}" for note in eq_notes[:2]])

        if idx != len(sections) - 1:
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def extract_source_excerpt(filename: str, content: bytes) -> tuple[str, str]:
    ext = Path(filename).suffix.lower()
    if ext in {".qmd", ".md"}:
        text = content.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        excerpt = "\n".join(lines[:300]).strip()
        return excerpt[:9000], "qmd"

    if ext in SUPPORTED_TEXT_EXTS:
        text = content.decode("utf-8", errors="ignore")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2500], "text"

    if ext == ".pdf":
        return "", "pdf"

    if ext in {".ppt", ".pptx"}:
        return "", "slides"

    if ext in {".png", ".jpg", ".jpeg"}:
        return "", "image"

    return "", "unknown"


def safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def wrap_svg_lines(text: str, max_chars: int = 34, max_lines: int = 3) -> list[str]:
    words = sanitize_text(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


def svg_palette_for_layout(layout: str) -> tuple[str, str, str]:
    key = str(layout or "concept").lower()
    if key == "formula":
        return "#eef5ff", "#9bb8f5", "#315da8"
    if key == "simulation":
        return "#effbf4", "#8ed9b2", "#2f7b5f"
    if key == "example":
        return "#fff8ef", "#f8c88a", "#9c5a00"
    if key == "activity":
        return "#fff1f1", "#f4abab", "#9f3f3f"
    if key == "summary":
        return "#f6f3ff", "#c2b4f0", "#5a4e9c"
    return "#f4f8ff", "#adc4ec", "#3f5f98"


def build_slide_illustration_svg(slide: dict[str, object], idx: int) -> str:
    layout = sanitize_text(slide.get("layout", "concept")).lower()
    bg, accent_soft, accent = svg_palette_for_layout(layout)

    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
            f'<rect width="1600" height="900" fill="{bg}" />',
            f'<circle cx="1330" cy="140" r="245" fill="{accent_soft}" opacity="0.24"/>',
            f'<circle cx="1460" cy="770" r="265" fill="{accent_soft}" opacity="0.21"/>',
            f'<circle cx="220" cy="120" r="170" fill="{accent_soft}" opacity="0.12"/>',
            f'<rect x="76" y="96" width="1448" height="708" rx="30" fill="#ffffff" opacity="0.64" />',
            f'<rect x="76" y="96" width="1448" height="708" rx="30" fill="none" stroke="{accent}" stroke-width="3" opacity="0.35" />',
            f'<path d="M1010 240 L1425 240" fill="none" stroke="{accent}" stroke-width="8" opacity="0.28"/>',
            f'<path d="M1010 310 L1390 310" fill="none" stroke="{accent}" stroke-width="8" opacity="0.22"/>',
            f'<path d="M1010 380 L1440 380" fill="none" stroke="{accent}" stroke-width="8" opacity="0.18"/>',
            f'<path d="M1040 560 L1130 470 L1210 515 L1295 410 L1385 468" fill="none" stroke="{accent}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" opacity="0.34"/>',
            f'<circle cx="1040" cy="560" r="10" fill="{accent}" opacity="0.40"/>',
            f'<circle cx="1130" cy="470" r="10" fill="{accent}" opacity="0.40"/>',
            f'<circle cx="1210" cy="515" r="10" fill="{accent}" opacity="0.40"/>',
            f'<circle cx="1295" cy="410" r="10" fill="{accent}" opacity="0.40"/>',
            f'<circle cx="1385" cy="468" r="10" fill="{accent}" opacity="0.40"/>',
            f'<rect x="150" y="660" width="340" height="18" rx="9" fill="{accent}" opacity="0.16"/>',
            f'<rect x="150" y="700" width="260" height="18" rx="9" fill="{accent}" opacity="0.14"/>',
            f'<rect x="150" y="740" width="300" height="18" rx="9" fill="{accent}" opacity="0.12"/>',
            "</svg>",
        ]
    )


def build_image_prompt(deck_title: str, slide: dict[str, object], idx: int) -> str:
    title = sanitize_text(slide.get("title", f"Slide {idx + 1}"))
    subtitle = sanitize_text(slide.get("subtitle", ""))
    layout = sanitize_text(slide.get("layout", "concept")).lower() or "concept"
    bullets = slide.get("bullets", []) if isinstance(slide.get("bullets"), list) else []
    bullet_text = "; ".join(sanitize_bullet_text(item) for item in bullets[:4] if sanitize_bullet_text(item))
    layout_hint = {
        "title": "opening lecture cover visual",
        "concept": "clean conceptual diagram with data points/shapes",
        "formula": "mathematical visual metaphor with geometric structure, no equations rendered as text",
        "simulation": "data simulation style chart-like abstract figure",
        "example": "applied classroom example visual with simple objects",
        "activity": "students collaborating around data cards visual",
        "summary": "recap workflow visual with arrows and milestones",
    }.get(layout, "educational data concept visual")

    prompt_parts = [
        f"Create a 16:9 background illustration for a statistics lecture slide.",
        f"Deck theme: {deck_title}.",
        f"Slide title: {title}.",
        f"Slide subtitle/context: {subtitle}.",
        f"Key learning points: {bullet_text}.",
        f"Visual intent: {layout_hint}.",
        f"Style: {IMAGE_STYLE_PROMPT}.",
        "Requirements: no text labels, no logos, no watermarks, no people faces close-up, high contrast but soft enough behind slide text.",
    ]
    return " ".join(part for part in prompt_parts if part)


def call_openai_image(prompt: str) -> bytes:
    key = resolve_image_key("openai")
    if not key:
        raise RuntimeError("Missing OpenAI API key for image generation.")
    payload = {
        "model": resolve_image_model("openai"),
        "prompt": prompt,
        "size": OPENAI_IMAGE_SIZE,
        "quality": "medium",
        "n": 1,
        "response_format": "b64_json",
    }
    data = post_json(
        "https://api.openai.com/v1/images/generations",
        payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    items = data.get("data", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError("OpenAI image response missing data.")
    first = items[0] if isinstance(items[0], dict) else {}
    b64 = str(first.get("b64_json", "")).strip()
    if not b64:
        raise RuntimeError("OpenAI image response missing b64_json.")
    return base64.b64decode(b64)


def generate_external_figures(
    *,
    deck_id: str,
    deck_title: str,
    sections: list[dict[str, object]],
    progress_cb: Callable[[str, float], None] | None = None,
) -> tuple[int, str | None]:
    if progress_cb:
        progress_cb("image_generation", 0.84)
    provider = resolve_image_provider()
    if not IMAGE_GENERATION_ENABLED or provider in {"none", "off", "local"}:
        return 0, None
    if provider != "openai":
        return 0, f"Image provider '{provider}' is not implemented; using local illustrations."
    if not resolve_image_key("openai"):
        return 0, "Image generation skipped: OpenAI key not configured."

    max_count = min(IMAGE_MAX_SLIDES, len(sections))
    if max_count <= 0:
        return 0, None

    deck_dir = DECK_DIR / deck_id
    figures_dir = deck_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors: list[str] = []
    for idx in range(max_count):
        slide = normalize_slide_obj(sections[idx], idx)
        prompt = build_image_prompt(deck_title, slide, idx)
        try:
            image_bytes = call_openai_image(prompt)
            file_name = f"slide-{idx + 1:02d}.png"
            file_path = figures_dir / file_name
            safe_write_bytes(file_path, image_bytes)
            if isinstance(sections[idx], dict):
                sections[idx]["figurePath"] = f"figures/{file_name}"
            success += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"slide {idx + 1}: {sanitize_text(str(exc))[:120]}")
        if progress_cb:
            p = 0.84 + ((idx + 1) / max_count) * 0.04
            progress_cb(f"image_slide_{idx + 1}", round(p, 4))

    warning = None
    if errors and success == 0:
        warning = "Image generation failed; kept local illustrations."
    elif errors:
        warning = f"Some slide images failed ({len(errors)}); local illustrations kept for the rest."
    return success, warning


def attach_slide_figure_paths(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for idx, raw in enumerate(sections):
        slide = normalize_slide_obj(raw, idx)
        fig = str(slide.get("figurePath", "")).strip()
        if not fig:
            slide["figurePath"] = f"figures/slide-{idx + 1:02d}.svg"
        out.append(slide)
    return out


def write_deck_figures(deck: dict[str, object]) -> None:
    deck_id = str(deck.get("id", "")).strip()
    if not deck_id:
        return
    slides = deck.get("slides", [])
    if not isinstance(slides, list):
        return
    deck_dir = DECK_DIR / deck_id
    figures_dir = deck_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for idx, raw_slide in enumerate(slides):
        slide = normalize_slide_obj(raw_slide, idx)
        fig_rel = str(slide.get("figurePath", f"figures/slide-{idx + 1:02d}.svg")).strip() or f"figures/slide-{idx + 1:02d}.svg"
        fig_name = Path(fig_rel).name
        fig_path = figures_dir / fig_name
        suffix = fig_path.suffix.lower()
        if suffix == ".svg":
            svg = build_slide_illustration_svg(slide, idx)
            safe_write(fig_path, svg)
        elif not fig_path.exists():
            # Keep expected file path stable by creating SVG fallback when external image is missing.
            fallback_name = f"slide-{idx + 1:02d}.svg"
            fig_name = fallback_name
            fig_path = figures_dir / fig_name
            svg = build_slide_illustration_svg(slide, idx)
            safe_write(fig_path, svg)
        if isinstance(raw_slide, dict):
            raw_slide["figurePath"] = f"figures/{fig_name}"


def load_deck(deck_id: str) -> dict[str, object] | None:
    meta_path = DECK_DIR / deck_id / "deck.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_deck(deck: dict[str, object]) -> None:
    deck_id = str(deck["id"])
    deck_dir = DECK_DIR / deck_id
    deck_dir.mkdir(parents=True, exist_ok=True)
    slides = deck.get("slides", [])
    if isinstance(slides, list):
        deck["slides"] = attach_slide_figure_paths(slides)
        write_deck_figures(deck)
    shared_css = BASE_DIR / "style.css"
    if shared_css.exists():
        shutil.copy2(shared_css, deck_dir / "style.css")
    safe_write(deck_dir / "deck.json", json.dumps(deck, ensure_ascii=True, indent=2))
    safe_write(deck_dir / "deck.qmd", str(deck.get("qmd", "")))


def render_quarto(deck_id: str) -> tuple[str | None, str | None]:
    quarto_bin = resolve_quarto_bin()
    if not quarto_bin:
        return None, "Quarto CLI not found; install Quarto to enable live revealjs render iframe."

    deck_dir = DECK_DIR / deck_id
    shared_css = BASE_DIR / "style.css"
    if shared_css.exists():
        shutil.copy2(shared_css, deck_dir / "style.css")

    qmd_name = "deck.qmd"
    cmd = [quarto_bin, "render", qmd_name, "--to", "revealjs", "--output", "index.html"]

    proc = subprocess.run(
        cmd,
        cwd=deck_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "render failed").strip()
        return None, f"Quarto render failed: {detail[:250]}"

    html_path = deck_dir / "index.html"
    if not html_path.exists():
        return None, "Quarto render did not produce index.html"

    return f"/decks/{deck_id}/index.html", None


def build_deck_bundle(deck_id: str) -> tuple[bytes, str]:
    deck = load_deck(deck_id)
    if not deck:
        raise FileNotFoundError(f"Deck not found: {deck_id}")

    save_deck(deck)
    deck_dir = DECK_DIR / deck_id
    title_slug = slugify(str(deck.get("title", "statedu-deck")), fallback=f"statedu-{deck_id}")
    bundle_name = f"{title_slug}.zip"

    qmd_path = deck_dir / "deck.qmd"
    style_path = deck_dir / "style.css"
    figures_dir = deck_dir / "figures"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if qmd_path.exists():
            zf.write(qmd_path, arcname="statedu-deck.qmd")
        else:
            zf.writestr("statedu-deck.qmd", str(deck.get("qmd", "")))

        if style_path.exists():
            zf.write(style_path, arcname="style.css")
        elif (BASE_DIR / "style.css").exists():
            zf.write(BASE_DIR / "style.css", arcname="style.css")

        if figures_dir.exists():
            for fig_path in sorted(figures_dir.glob("*")):
                if fig_path.is_file():
                    zf.write(fig_path, arcname=f"figures/{fig_path.name}")

        zf.writestr(
            "README-BUNDLE.txt",
            "\n".join(
                [
                    "StatEdu Deck Bundle",
                    "",
                    "Contents:",
                    "- statedu-deck.qmd",
                    "- style.css",
                    "- figures/ (illustrative image assets used by the deck)",
                    "",
                    "Render locally with:",
                    "quarto render statedu-deck.qmd --to revealjs",
                ]
            ),
        )

    return buffer.getvalue(), bundle_name


def create_deck(
    prompt: str,
    output_format: str,
    animation: str,
    teaching_style: str,
    revision: int,
    source_name: str = "",
    source_excerpt: str = "",
    feedback: str = "",
    previous_slides: list[dict[str, object]] | None = None,
    locked_slide_indexes: list[int] | None = None,
    approved_slide_indexes: list[int] | None = None,
    requested_slide_count: int | None = None,
    render_now: bool = True,
    deck_id: str | None = None,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict[str, object]:
    fallback_title = infer_title(prompt)
    teaching_style = normalize_teaching_style(teaching_style)
    if requested_slide_count and requested_slide_count > 0:
        requested_count = clamp_slide_count(int(requested_slide_count))
    elif previous_slides:
        requested_count = clamp_slide_count(len(previous_slides))
    else:
        requested_count = infer_slide_count(prompt)
    llm_used = False
    llm_warning: str | None = None
    llm_provider = resolve_llm_provider()
    web_research: list[dict[str, str]] = []

    locked_slide_indexes = locked_slide_indexes or []
    approved_slide_indexes = approved_slide_indexes or []

    try:
        title, sections, llm_used, llm_warning, llm_provider, web_research = maybe_generate_with_llm(
            prompt=prompt,
            source_excerpt=source_excerpt,
            feedback=feedback,
            output_format=output_format,
            animation=animation,
            teaching_style=teaching_style,
            target_count=requested_count,
            previous_slides=previous_slides,
            progress_cb=progress_cb,
        )
    except Exception as exc:  # noqa: BLE001
        title = fallback_title
        seed_prompt = f"{prompt}\n\nRefinement request: {feedback}".strip()
        sections = make_slide_sections(
            title,
            seed_prompt,
            source_excerpt=source_excerpt,
            teaching_style=teaching_style,
        )[:requested_count]
        llm_warning = f"LLM fallback used: {exc}"

    sections = align_slide_count(sections, requested_count, previous_slides)
    locked_norm = sanitize_slide_indexes(locked_slide_indexes, requested_count)
    approved_norm = sanitize_slide_indexes(approved_slide_indexes, requested_count)
    protected_indexes = sorted(set(locked_norm + approved_norm))
    if protected_indexes and previous_slides:
        sections = merge_protected_slides(previous_slides, sections, protected_indexes, requested_count)

    if output_format == "quarto" and SLIDE_AUTO_SPLIT_ENABLED and not protected_indexes:
        before_count = len(sections)
        sections = rebalance_slides_for_readability(sections, MAX_SLIDE_COUNT)
        after_count = len(sections)
        if after_count > before_count:
            split_msg = f"Auto-split {after_count - before_count} dense slide segment(s) for readability."
            llm_warning = f"{llm_warning}; {split_msg}" if llm_warning else split_msg

    requested_count = len(sections)
    locked_norm = sanitize_slide_indexes(locked_norm, requested_count)
    approved_norm = sanitize_slide_indexes(approved_norm, requested_count)
    sections = attach_slide_figure_paths(sections)

    description = (
        "Rendered from .qmd (Reveal.js)"
        if output_format == "quarto"
        else "Prepared for Google Slides export"
    )
    if source_name:
        description += f" | Source: {source_name}"

    resolved_id = deck_id or uuid.uuid4().hex[:12]
    image_count = 0
    image_provider_used = resolve_image_provider()
    image_warning: str | None = None
    if output_format == "quarto":
        image_count, image_warning = generate_external_figures(
            deck_id=resolved_id,
            deck_title=title,
            sections=sections,
            progress_cb=progress_cb,
        )
        if image_provider_used == "local":
            image_count = len(sections)
        if image_warning:
            llm_warning = f"{llm_warning}; {image_warning}" if llm_warning else image_warning

    qmd = build_qmd(title, sections, animation)
    deck: dict[str, object] = {
        "id": resolved_id,
        "title": title,
        "prompt": prompt,
        "lastFeedback": feedback,
        "description": description,
        "slides": sections,
        "bullets": sections[0]["bullets"] if sections else [],
        "qmd": qmd,
        "revision": revision,
        "format": output_format,
        "animation": animation,
        "teachingStyle": teaching_style,
        "sourceName": source_name,
        "sourceExcerpt": source_excerpt,
        "requestedSlideCount": requested_count,
        "lockedSlideIndexes": locked_norm,
        "approvedSlideIndexes": approved_norm,
        "llmUsed": llm_used,
        "llmProvider": llm_provider,
        "generationPipeline": ["web_research", "template", "content_per_slide", "review", "fact_check", "correction", "image_generation"] if llm_used else ["web_research", "fallback", "image_generation"],
        "llmWarning": llm_warning,
        "webResearch": web_research,
        "imageGenerationEnabled": IMAGE_GENERATION_ENABLED,
        "imageProvider": image_provider_used,
        "imageModel": resolve_image_model(image_provider_used),
        "imageCount": image_count,
        "renderUrl": None,
        "renderWarning": None,
        "updatedAt": now_iso(),
    }
    save_deck(deck)

    if output_format == "quarto" and render_now:
        render_url, warning = render_quarto(resolved_id)
        deck["renderUrl"] = render_url
        deck["renderWarning"] = warning
        save_deck(deck)

    return deck


class Handler(BaseHTTPRequestHandler):
    server_version = "StatEduServer/0.2"

    def _send_json(self, code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        code: int,
        body: bytes,
        *,
        content_type: str,
        filename: str | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_static_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            self.send_error(404, "Not Found")
            return

        data = file_path.read_bytes()
        content_type = "text/plain; charset=utf-8"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif file_path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
            if file_path.suffix == ".png":
                content_type = "image/png"
            elif file_path.suffix in {".jpg", ".jpeg"}:
                content_type = "image/jpeg"
            elif file_path.suffix == ".gif":
                content_type = "image/gif"
            else:
                content_type = "image/svg+xml"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _parse_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _parse_multipart(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return None

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )
        return form

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_static_file(BASE_DIR / "index.html")
            return

        if path == "/api/health":
            quarto_bin = resolve_quarto_bin()
            llm = llm_status()
            image = image_status()
            self._send_json(
                200,
                {
                    "ok": True,
                    "quartoInstalled": bool(quarto_bin),
                    "quartoBin": quarto_bin or "",
                    "llmProvider": llm["provider"],
                    "llmModel": llm["model"],
                    "llmConfigured": llm["configured"],
                    "llmTimeoutSec": LLM_TIMEOUT_SEC,
                    "llmRetryCount": LLM_RETRY_COUNT,
                    "renderStepDelaySec": RENDER_STEP_DELAY_SEC,
                    "slideAutoSplitEnabled": SLIDE_AUTO_SPLIT_ENABLED,
                    "webResearchEnabled": WEB_RESEARCH_ENABLED,
                    "webResearchMaxResults": WEB_RESEARCH_MAX_RESULTS,
                    "webResearchTimeoutSec": WEB_RESEARCH_TIMEOUT_SEC,
                    "imageGenerationEnabled": image["enabled"],
                    "imageProvider": image["provider"],
                    "imageModel": image["model"],
                    "imageConfigured": image["configured"],
                    "imageMaxSlides": image["maxSlides"],
                    "timestamp": now_iso(),
                },
            )
            return

        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :].strip()
            job = get_job(job_id)
            if not job:
                self._send_json(404, {"error": f"Job not found: {job_id}"})
                return
            self._send_json(200, {"job": job})
            return

        if path == "/api/deck/download":
            query = parse_qs(parsed.query)
            deck_id = str((query.get("deckId", [""])[0] or "")).strip()
            if not deck_id:
                self._send_json(400, {"error": "deckId is required"})
                return
            try:
                body, filename = build_deck_bundle(deck_id)
            except FileNotFoundError:
                self._send_json(404, {"error": f"Deck not found: {deck_id}"})
                return
            self._send_bytes(
                200,
                body,
                content_type="application/zip",
                filename=filename,
            )
            return

        if path.startswith("/decks/"):
            rel = unquote(path[len("/decks/") :]).lstrip("/")
            target = (DECK_DIR / rel).resolve()
            if not str(target).startswith(str(DECK_DIR.resolve())):
                self.send_error(403, "Forbidden")
                return
            self._send_static_file(target)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/generate":
                self._handle_generate()
                return
            if parsed.path == "/api/generate-job":
                self._handle_generate_job()
                return
            if parsed.path == "/api/refine":
                self._handle_refine()
                return
            if parsed.path == "/api/refine-job":
                self._handle_refine_job()
                return
            if parsed.path == "/api/deck/update-qmd":
                self._handle_update_qmd()
                return
            if parsed.path == "/api/export/google-slides":
                self._handle_export()
                return
            self.send_error(404, "Not Found")
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})

    def _handle_generate(self):
        form = self._parse_multipart()
        prompt = ""
        output_format = "quarto"
        animation = "step"
        teaching_style = "balanced"
        requested_slide_count = 0
        source_name = ""
        source_excerpt = ""
        source_type = "none"

        if form is not None:
            prompt = form.getfirst("prompt", "").strip()
            output_format = form.getfirst("format", "quarto").strip() or "quarto"
            animation = form.getfirst("animation", "step").strip() or "step"
            teaching_style = normalize_teaching_style(form.getfirst("teachingStyle", "balanced"))
            try:
                requested_slide_count = int(form.getfirst("requestedSlideCount", "0") or "0")
            except ValueError:
                requested_slide_count = 0

            upload = form["file"] if "file" in form else None
            if upload is not None and getattr(upload, "filename", None):
                source_name = os.path.basename(upload.filename)
                file_bytes = upload.file.read() if upload.file else b""
                if file_bytes:
                    upload_id = uuid.uuid4().hex[:8]
                    upload_path = UPLOAD_DIR / f"{upload_id}-{source_name}"
                    upload_path.write_bytes(file_bytes)
                    source_excerpt, source_type = extract_source_excerpt(source_name, file_bytes)
        else:
            data = self._parse_json_body()
            prompt = str(data.get("prompt", "")).strip()
            output_format = str(data.get("format", "quarto")).strip() or "quarto"
            animation = str(data.get("animation", "step")).strip() or "step"
            teaching_style = normalize_teaching_style(data.get("teachingStyle", "balanced"))
            requested_slide_count = int(data.get("requestedSlideCount", 0) or 0)
            source_name = str(data.get("sourceName", "")).strip()

        prompt_to_use = prompt or "Use uploaded source to generate a clear stats lesson deck."
        deck = create_deck(
            prompt=prompt_to_use,
            output_format=output_format,
            animation=animation,
            teaching_style=teaching_style,
            revision=1,
            source_name=source_name,
            source_excerpt=source_excerpt,
            feedback="",
            previous_slides=None,
            locked_slide_indexes=[],
            approved_slide_indexes=[],
            requested_slide_count=requested_slide_count if requested_slide_count > 0 else None,
        )
        deck["sourceType"] = source_type
        save_deck(deck)
        self._send_json(200, {"deck": deck})

    def _handle_refine(self):
        data = self._parse_json_body()
        feedback = str(data.get("feedback", "")).strip()
        current = data.get("currentDeck", {}) or {}

        current_id = str(current.get("id", "")).strip() or uuid.uuid4().hex[:12]
        previous = load_deck(current_id) or current

        output_format = str(current.get("format", data.get("format", "quarto"))) or "quarto"
        animation = str(current.get("animation", data.get("animation", "step"))) or "step"
        teaching_style = normalize_teaching_style(
            current.get("teachingStyle", previous.get("teachingStyle", data.get("teachingStyle", "balanced")))
        )
        prev_revision = int(previous.get("revision", current.get("revision", 1)) or 1)
        requested_slide_count = int(
            current.get(
                "requestedSlideCount",
                previous.get("requestedSlideCount", len(previous.get("slides", [])) if isinstance(previous.get("slides", []), list) else 0),
            )
            or 0
        )

        base_prompt = str(previous.get("prompt", current.get("title", ""))).strip()
        source_name = str(previous.get("sourceName", "")).strip()
        source_excerpt = str(previous.get("sourceExcerpt", "")).strip()
        prev_slides = previous.get("slides", []) if isinstance(previous.get("slides", []), list) else None
        locked_raw = current.get("lockedSlideIndexes", previous.get("lockedSlideIndexes", []))
        approved_raw = current.get("approvedSlideIndexes", previous.get("approvedSlideIndexes", []))

        deck = create_deck(
            prompt=base_prompt,
            output_format=output_format,
            animation=animation,
            teaching_style=teaching_style,
            revision=prev_revision + 1,
            source_name=source_name,
            source_excerpt=source_excerpt,
            feedback=feedback,
            previous_slides=prev_slides,
            locked_slide_indexes=locked_raw if isinstance(locked_raw, list) else [],
            approved_slide_indexes=approved_raw if isinstance(approved_raw, list) else [],
            requested_slide_count=requested_slide_count if requested_slide_count > 0 else None,
            deck_id=current_id,
        )
        self._send_json(200, {"deck": deck})

    def _handle_generate_job(self):
        form = self._parse_multipart()
        prompt = ""
        output_format = "quarto"
        animation = "step"
        teaching_style = "balanced"
        requested_slide_count = 0
        source_name = ""
        source_excerpt = ""
        source_type = "none"

        if form is not None:
            prompt = form.getfirst("prompt", "").strip()
            output_format = form.getfirst("format", "quarto").strip() or "quarto"
            animation = form.getfirst("animation", "step").strip() or "step"
            teaching_style = normalize_teaching_style(form.getfirst("teachingStyle", "balanced"))
            try:
                requested_slide_count = int(form.getfirst("requestedSlideCount", "0") or "0")
            except ValueError:
                requested_slide_count = 0

            upload = form["file"] if "file" in form else None
            if upload is not None and getattr(upload, "filename", None):
                source_name = os.path.basename(upload.filename)
                file_bytes = upload.file.read() if upload.file else b""
                if file_bytes:
                    upload_id = uuid.uuid4().hex[:8]
                    upload_path = UPLOAD_DIR / f"{upload_id}-{source_name}"
                    upload_path.write_bytes(file_bytes)
                    source_excerpt, source_type = extract_source_excerpt(source_name, file_bytes)
        else:
            data = self._parse_json_body()
            prompt = str(data.get("prompt", "")).strip()
            output_format = str(data.get("format", "quarto")).strip() or "quarto"
            animation = str(data.get("animation", "step")).strip() or "step"
            teaching_style = normalize_teaching_style(data.get("teachingStyle", "balanced"))
            requested_slide_count = int(data.get("requestedSlideCount", 0) or 0)
            source_name = str(data.get("sourceName", "")).strip()

        prompt_to_use = prompt or "Use uploaded source to generate a clear stats lesson deck."
        payload = {
            "prompt": prompt_to_use,
            "output_format": output_format,
            "animation": animation,
            "teaching_style": teaching_style,
            "revision": 1,
            "source_name": source_name,
            "source_excerpt": source_excerpt,
            "feedback": "",
            "previous_slides": None,
            "locked_slide_indexes": [],
            "approved_slide_indexes": [],
            "requested_slide_count": requested_slide_count if requested_slide_count > 0 else None,
            "deck_id": None,
            "sourceType": source_type,
        }
        job_id = create_job("generate", payload=payload)
        thread = threading.Thread(target=run_generation_job, args=(job_id, payload), daemon=True)
        thread.start()
        self._send_json(200, {"jobId": job_id})

    def _handle_refine_job(self):
        data = self._parse_json_body()
        feedback = str(data.get("feedback", "")).strip()
        current = data.get("currentDeck", {}) or {}

        current_id = str(current.get("id", "")).strip() or uuid.uuid4().hex[:12]
        previous = load_deck(current_id) or current

        output_format = str(current.get("format", data.get("format", "quarto"))) or "quarto"
        animation = str(current.get("animation", data.get("animation", "step"))) or "step"
        teaching_style = normalize_teaching_style(
            current.get("teachingStyle", previous.get("teachingStyle", data.get("teachingStyle", "balanced")))
        )
        prev_revision = int(previous.get("revision", current.get("revision", 1)) or 1)
        requested_slide_count = int(
            current.get(
                "requestedSlideCount",
                previous.get("requestedSlideCount", len(previous.get("slides", [])) if isinstance(previous.get("slides", []), list) else 0),
            )
            or 0
        )

        base_prompt = str(previous.get("prompt", current.get("title", ""))).strip()
        source_name = str(previous.get("sourceName", "")).strip()
        source_excerpt = str(previous.get("sourceExcerpt", "")).strip()
        prev_slides = previous.get("slides", []) if isinstance(previous.get("slides", []), list) else None
        locked_raw = current.get("lockedSlideIndexes", previous.get("lockedSlideIndexes", []))
        approved_raw = current.get("approvedSlideIndexes", previous.get("approvedSlideIndexes", []))

        payload = {
            "prompt": base_prompt,
            "output_format": output_format,
            "animation": animation,
            "teaching_style": teaching_style,
            "revision": prev_revision + 1,
            "source_name": source_name,
            "source_excerpt": source_excerpt,
            "feedback": feedback,
            "previous_slides": prev_slides,
            "locked_slide_indexes": locked_raw if isinstance(locked_raw, list) else [],
            "approved_slide_indexes": approved_raw if isinstance(approved_raw, list) else [],
            "requested_slide_count": requested_slide_count if requested_slide_count > 0 else None,
            "deck_id": current_id,
        }
        job_id = create_job("refine", payload={"deckId": current_id})
        thread = threading.Thread(target=run_generation_job, args=(job_id, payload), daemon=True)
        thread.start()
        self._send_json(200, {"jobId": job_id})

    def _handle_export(self):
        data = self._parse_json_body()
        deck = data.get("deck", {}) if isinstance(data.get("deck"), dict) else {}
        title = str(deck.get("title", "deck"))
        deck_id = str(deck.get("id", ""))
        slug = slugify(title)

        export_payload = {
            "ok": True,
            "message": "Export job queued.",
            "url": f"https://docs.google.com/presentation/d/mock-{slug}",
            "deckId": deck_id,
            "queuedAt": now_iso(),
            "note": "Replace this mock endpoint with Google Slides API integration.",
        }
        self._send_json(200, export_payload)

    def _handle_update_qmd(self):
        data = self._parse_json_body()
        deck_id = str(data.get("deckId", "")).strip()
        qmd = str(data.get("qmd", ""))
        if not deck_id:
            self._send_json(400, {"error": "deckId is required"})
            return

        deck = load_deck(deck_id)
        if not deck:
            self._send_json(404, {"error": f"Deck not found: {deck_id}"})
            return

        old_title = str(deck.get("title", "Generated Lesson Deck"))
        parsed_title, parsed_slides = parse_qmd_title_and_slides(qmd, old_title)

        deck["qmd"] = qmd
        deck["title"] = parsed_title
        if parsed_slides:
            deck["slides"] = parsed_slides
            deck["bullets"] = parsed_slides[0].get("bullets", []) if parsed_slides else []
            deck["requestedSlideCount"] = len(parsed_slides)
        deck["updatedAt"] = now_iso()
        deck["llmUsed"] = False
        deck["llmWarning"] = "Manual .qmd edit applied."
        save_deck(deck)

        if str(deck.get("format", "quarto")) == "quarto":
            render_url, warning = render_quarto(deck_id)
            deck["renderUrl"] = render_url
            deck["renderWarning"] = warning
            save_deck(deck)

        self._send_json(200, {"deck": deck})


def main() -> None:
    ensure_dirs()
    quarto_bin = resolve_quarto_bin()
    llm = llm_status()
    image = image_status()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server running at http://{HOST}:{PORT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Quarto installed: {'yes' if quarto_bin else 'no'}")
    if quarto_bin:
        print(f"Quarto bin: {quarto_bin}")
    print(f"LLM provider: {llm['provider']} ({'configured' if llm['configured'] else 'not configured'})")
    print(f"LLM model: {llm['model']}")
    print(f"Slide auto-split: {'on' if SLIDE_AUTO_SPLIT_ENABLED else 'off'}")
    print(
        "Image stage: "
        f"{image['provider']} ({image['model']}, "
        f"{'configured' if image['configured'] else 'not configured'}, "
        f"maxSlides={image['maxSlides']})"
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
