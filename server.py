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
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import unquote, urlparse

HOST = os.getenv("STATEDU_HOST", "127.0.0.1")
PORT = int(os.getenv("STATEDU_PORT", "8000"))
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".statedu"
DECK_DIR = DATA_DIR / "decks"
UPLOAD_DIR = DATA_DIR / "uploads"

SUPPORTED_TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".py", ".r", ".qmd"}
LLM_TIMEOUT_SEC = int(os.getenv("STATEDU_LLM_TIMEOUT_SEC", "45"))
DEFAULT_SLIDE_COUNT = int(os.getenv("STATEDU_DEFAULT_SLIDE_COUNT", "8"))
MAX_SLIDE_COUNT = int(os.getenv("STATEDU_MAX_SLIDE_COUNT", "40"))
RENDER_STEP_DELAY_SEC = float(os.getenv("STATEDU_RENDER_STEP_DELAY_SEC", "0.35"))
VALID_TEACHING_STYLES = {"balanced", "conceptual", "mathematical", "simulation"}

MAX_SUBTITLE_CHARS = 120
MAX_BULLET_CHARS = 118
MAX_EXAMPLE_CHARS = 260
MAX_ACTIVITY_CHARS = 220
MAX_NOTES_CHARS = 170

JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


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


def clamp_sentence(text: object, max_chars: int) -> str:
    value = sanitize_text(text)
    if len(value) <= max_chars:
        return value
    trimmed = value[: max_chars - 3].rstrip(" ,;:")
    return f"{trimmed}..."


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
    return text


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
            subtitle = clamp_sentence(item.get("subtitle", ""), MAX_SUBTITLE_CHARS)
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
            example = clamp_sentence(item.get("example", ""), MAX_EXAMPLE_CHARS)
            activity = clamp_sentence(item.get("activity", ""), MAX_ACTIVITY_CHARS)
            equation = str(item.get("equation", "")).strip()
            notes = clamp_sentence(item.get("notes", ""), MAX_NOTES_CHARS)
            r_chunk = str(item.get("rChunk", "")).strip()
            if is_unhelpful_source_sentence(notes):
                notes = ""
            if is_presenter_directive(notes):
                notes = ""

            has_content = bool(bullets or example or activity or equation or r_chunk)
            if not has_content:
                continue
            if len(bullets) < 2 and layout not in {"simulation", "formula"}:
                bullets = infer_bullets(f"{title}\n{slide_title}")[:3]

            sections.append(
                {
                    "title": slide_title,
                    "subtitle": subtitle,
                    "bullets": bullets[:6],
                    "layout": layout,
                    "example": example,
                    "activity": activity,
                    "equation": equation,
                    "notes": notes,
                    "rChunk": r_chunk,
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
            "example": "",
            "activity": "",
            "equation": "",
            "notes": "",
            "rChunk": "",
        }

    title = sanitize_text(raw.get("title", "")) or f"Slide {idx + 1}"
    subtitle = clamp_sentence(raw.get("subtitle", ""), MAX_SUBTITLE_CHARS)
    layout = sanitize_text(raw.get("layout", "concept")).lower() or "concept"
    if layout not in {"title", "concept", "formula", "simulation", "example", "activity", "summary"}:
        layout = "concept"
    example = clamp_sentence(raw.get("example", ""), MAX_EXAMPLE_CHARS)
    activity = clamp_sentence(raw.get("activity", ""), MAX_ACTIVITY_CHARS)
    equation = str(raw.get("equation", "")).strip()
    notes = clamp_sentence(raw.get("notes", ""), MAX_NOTES_CHARS)
    r_chunk = str(raw.get("rChunk", "")).strip()
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
    if is_unhelpful_source_sentence(notes):
        notes = ""
    if is_presenter_directive(notes):
        notes = ""
    return {
        "title": title,
        "subtitle": subtitle,
        "bullets": bullets[:6],
        "layout": layout,
        "example": example,
        "activity": activity,
        "equation": equation,
        "notes": notes,
        "rChunk": r_chunk,
    }


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
        slide_title = m_h2.group(1).strip()
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
    update_job(job_id, totalSlides=total, stage="rendering")

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
            progress=round(idx / total, 4),
            deck=final_deck,
        )
        if idx < total and RENDER_STEP_DELAY_SEC > 0:
            time.sleep(RENDER_STEP_DELAY_SEC)

    return final_deck


def run_generation_job(job_id: str, deck_params: dict[str, object]) -> None:
    try:
        update_job(job_id, state="running", stage="generating", progress=0.05)
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

    req = urlrequest.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=LLM_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw or "{}")
            if not isinstance(parsed, dict):
                raise RuntimeError("Non-object JSON response from model API.")
            return parsed
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"LLM network error: {exc}") from exc


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
) -> tuple[str, list[dict[str, object]], bool, str | None, str]:
    provider = resolve_llm_provider()
    model = resolve_llm_model(provider)
    fallback_title = infer_title(prompt)

    if provider in {"mock", "none", "off"}:
        return (
            fallback_title,
            make_slide_sections(
                fallback_title,
                prompt + "\n" + feedback,
                source_excerpt,
                teaching_style=teaching_style,
            ),
            False,
            None,
            provider,
        )

    system_prompt = (
        "You are a senior instructional designer and statistics professor. "
        "Return strict JSON only with keys: title, slides. "
        "slides must be an array of 3-40 objects. "
        "Each slide object must include: "
        "title (string), subtitle (string), bullets (array of 1-6 concise strings), "
        "layout (one of: title, concept, formula, simulation, example, activity, summary), "
        "example (string, optional), activity (string, optional), equation (string, optional), "
        "notes (string, optional), rChunk (string, optional R code body without markdown fences). "
        "Do not output literal '{.fragment}' tokens. "
        "Design for teaching: setup assumptions, show one formula, run one simulation or computation, then interpret. "
        "Write grammatical, learner-friendly sentences only. "
        "Audience-facing slide copy only; do not include presenter coaching language."
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

    user_prompt = (
        f"Primary prompt:\n{prompt.strip()}\n\n"
        f"Output format target: {output_format}\n"
        f"Animation preference: {animation}\n"
        f"Requested slide count: {target_count}\n"
        f"Teaching style profile: {teaching_style}\n"
        "Audience: undergraduate statistics learners unless user says otherwise.\n"
        "Slide quality bar: avoid generic bullets; include applied context and interpretation.\n"
    )
    if source_excerpt:
        user_prompt += f"\nSource excerpt (possibly truncated):\n{source_excerpt}\n"
    if feedback:
        user_prompt += f"\nRefinement feedback:\n{feedback}\n"
    if previous_context:
        user_prompt += f"\nExisting slide draft to improve:\n{previous_context}\n"

    user_prompt += (
        "\nReturn JSON only. Do not add markdown fences or explanation. "
        "Maintain a coherent narrative from intro to recap. "
        "Include at least one `formula` slide and one `simulation` or `example` slide with runnable `rChunk` when format is quarto."
        " Keep each bullet under 18 words and avoid verbose speaker-note style paragraphs."
    )

    if provider == "openai":
        raw = call_openai_json(system_prompt, user_prompt)
    elif provider == "gemini":
        raw = call_gemini_json(system_prompt, user_prompt)
    elif provider == "anthropic":
        raw = call_anthropic_json(system_prompt, user_prompt)
    else:
        raise RuntimeError(f"Unsupported STATEDU_LLM_PROVIDER: {provider}")

    title, sections = normalize_sections(
        str(raw.get("title", "")),
        raw.get("slides", []),
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
        return title, sections, False, f"{provider}/{model} returned too few slides; fallback applied.", provider

    return title, sections, True, None, provider


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
            "x <- rnorm(n, mean = 0, sd = 1)",
            "x_bar <- mean(x)",
            "s <- sd(x)",
            "t_crit <- qt(0.975, df = n - 1)",
            "ci <- x_bar + c(-1, 1) * t_crit * s / sqrt(n)",
            "ci",
        ]
    )
    example_text = "Compute the estimate, construct the interval, and interpret it in context."

    if "hypothesis" in merged_context or "p-value" in merged_context or "p value" in merged_context:
        formula = r"z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}, \quad p=P(|Z|\ge |z_{obs}|)"
        simulation_code = "\n".join(
            [
                "set.seed(42)",
                "n <- 80",
                "mu0 <- 0",
                "x <- rnorm(n, mean = 0.25, sd = 1)",
                "z <- (mean(x) - mu0) / (sd(x) / sqrt(n))",
                "p_value <- 2 * (1 - pnorm(abs(z)))",
                "c(z_stat = z, p_value = p_value)",
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
            notes = "Open by asking students what decision this tool helps us make."
        elif layout == "concept":
            bullets = [
                shared[0],
                shared[1],
                "Name one assumption and one consequence if it is violated.",
            ]
            notes = "Pause for a 30-second check-for-understanding before moving on."
        elif layout == "formula":
            slide_title = f"{title}: Core Formula"
            bullets = [
                "Define each symbol before using the formula.",
                "Link the formula to a decision rule students can apply.",
            ]
            equation = formula
            notes = "Have learners annotate the formula in pairs."
        elif layout == "simulation":
            slide_title = f"{title}: Simulation in R"
            bullets = [
                "Run a small simulation to observe sampling behavior.",
                "Interpret output in plain language, not only numbers.",
            ]
            r_chunk = simulation_code
            notes = "Ask students to change one parameter (n or sd) and predict impact."
        elif layout == "example":
            slide_title = f"{title}: Worked Example"
            bullets = [
                "State the problem setup and known quantities.",
                "Compute step-by-step and justify each step.",
                "Interpret the result in domain language.",
            ]
            notes = "Cold-call for interpretation, not only calculation."
        elif layout == "activity":
            slide_title = f"{title}: Classroom Activity"
            bullets = [
                "Work in pairs for 3 minutes.",
                "Write one decision and one justification.",
                "Compare with a neighboring group.",
            ]
            activity = "Mini-activity: solve one quick scenario and defend your conclusion."
            notes = "Debrief misconceptions explicitly after share-out."
        elif layout == "summary":
            slide_title = f"{title}: Summary and Exit Ticket"
            bullets = [
                "Summarize the key statistical takeaway in one sentence.",
                "State one common misconception and its correction.",
                "Exit ticket: apply the method to a new short prompt.",
            ]
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
                "bullets": bullets[:6],
                "layout": layout,
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

    def append_equation(eq: str) -> None:
        expr = str(eq or "").strip()
        if not expr:
            return
        if expr.startswith("$$") and expr.endswith("$$"):
            lines.append(expr)
            return
        lines.append("$$")
        lines.append(expr)
        lines.append("$$")

    def append_r_chunk(code: str, label: str) -> None:
        body = str(code or "").strip()
        if not body:
            return
        lines.append(f"```{{r {label}, echo=TRUE}}")
        lines.extend(body.splitlines())
        lines.append("```")

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
        if layout == "simulation" and not r_chunk:
            r_chunk = "\n".join(
                [
                    "set.seed(1234)",
                    "n <- 80",
                    "x <- rnorm(n)",
                    "mean(x)",
                    "sd(x)",
                ]
            )

        lines.append(f"## {slide_title}")
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
            append_equation(equation or r"\bar{x} \pm t_{n-1,\alpha/2}\frac{s}{\sqrt{n}}")
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
                append_equation(equation)
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
                append_equation(equation)

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


def load_deck(deck_id: str) -> dict[str, object] | None:
    meta_path = DECK_DIR / deck_id / "deck.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_deck(deck: dict[str, object]) -> None:
    deck_id = str(deck["id"])
    deck_dir = DECK_DIR / deck_id
    deck_dir.mkdir(parents=True, exist_ok=True)
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

    locked_slide_indexes = locked_slide_indexes or []
    approved_slide_indexes = approved_slide_indexes or []

    try:
        title, sections, llm_used, llm_warning, llm_provider = maybe_generate_with_llm(
            prompt=prompt,
            source_excerpt=source_excerpt,
            feedback=feedback,
            output_format=output_format,
            animation=animation,
            teaching_style=teaching_style,
            target_count=requested_count,
            previous_slides=previous_slides,
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

    description = (
        "Rendered from .qmd (Reveal.js)"
        if output_format == "quarto"
        else "Prepared for Google Slides export"
    )
    if source_name:
        description += f" | Source: {source_name}"

    resolved_id = deck_id or uuid.uuid4().hex[:12]
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
        "llmWarning": llm_warning,
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
            self._send_json(
                200,
                {
                    "ok": True,
                    "quartoInstalled": bool(quarto_bin),
                    "quartoBin": quarto_bin or "",
                    "llmProvider": llm["provider"],
                    "llmModel": llm["model"],
                    "llmConfigured": llm["configured"],
                    "renderStepDelaySec": RENDER_STEP_DELAY_SEC,
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
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server running at http://{HOST}:{PORT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Quarto installed: {'yes' if quarto_bin else 'no'}")
    if quarto_bin:
        print(f"Quarto bin: {quarto_bin}")
    print(f"LLM provider: {llm['provider']} ({'configured' if llm['configured'] else 'not configured'})")
    print(f"LLM model: {llm['model']}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
