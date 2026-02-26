# StatEdu Quarto Slides Helper GENAI

StatEdu Slides Helper AI is a human-in-the-loop web app for building classroom-ready statistics slide decks from prompts and source materials (textbook excerpts, notes, and existing slides).

The app generates editable Quarto Reveal.js slides, renders live previews, supports iterative refinement, and keeps users in control with slide lock/approve controls.

## Core Features

- Prompt-to-deck generation for statistics lessons
- Multi-stage GenAI pipeline for quality:
  - `web_research`
  - `template_generation`
  - `content_generation` (one slide at a time)
  - `review_stage`
  - `fact_check_stage`
  - `correction_stage`
  - `image_generation`
  - `rendering`
- Source upload support (`.pdf`, `.pptx`, `.md`, `.txt`, images)
- Human-in-the-loop refine workflow
- Teaching style profiles:
  - `balanced`
  - `conceptual`
  - `mathematical`
  - `simulation`
- Quarto `.qmd` output with live preview
- Auto-generated illustrative slide figures (`figures/*.svg`) with subtle background integration
- Audience-facing slide language (no presenter coaching text in visible slide body)
- Content structure emphasis per slide: definition, context, and student-useful materials
- R-enabled simulation/plot slides (e.g., histogram, scatter, cluster plot) for concept illustration
- Base-R-safe rendering path for generated `rChunk` code (auto-avoids missing tidyverse/ggplot dependencies)
- Automatic dense-slide splitting into continuation slides to reduce overflow in preview/export
- In-app `.qmd` editing with re-render
- One-click deck bundle download (`.qmd` + `style.css` + `figures/` as zip)
- Slide-level protection during refine (`approve` / `lock`)
- Stage-aware progress tracking in workspace and cover page pipeline panel
- Research-first generation flow (Genspark-style): source upload + web findings -> template -> per-slide content generation -> review -> fact-check -> correction -> image generation -> render
- Optional Google Slides export handshake endpoint

## Tech Stack

- Frontend: HTML, CSS, vanilla JavaScript
- Backend: Python (`http.server`)
- Rendering: Quarto CLI + Reveal.js
- LLM providers: OpenAI / Gemini / Anthropic (or fallback heuristic mode)

## Run Locally

1. Install Quarto CLI (optional but recommended for live render):

```bash
quarto --version
```

2. Set environment variables (example with OpenAI):

```bash
export STATEDU_LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
export STATEDU_OPENAI_MODEL=gpt-4.1-mini
```

3. Start server:

```bash
./server.py
```

4. Open:

```text
http://127.0.0.1:8000
```

## Configuration

- `STATEDU_HOST` (default: `127.0.0.1`)
- `STATEDU_PORT` (default: `8000`)
- `STATEDU_LLM_PROVIDER` (`openai`, `gemini`, `anthropic`, `mock`)
- `STATEDU_OPENAI_MODEL` / `STATEDU_GEMINI_MODEL` / `STATEDU_ANTHROPIC_MODEL`
- `STATEDU_LLM_TIMEOUT_SEC` (default: unlimited; set to `0`, `none`, or leave unset)
- `STATEDU_LLM_RETRY_COUNT` (default: `1`, total attempts = `1 + retry_count`)
- `STATEDU_LLM_RETRY_BACKOFF_SEC` (default: `1.2`)
- `STATEDU_DEFAULT_SLIDE_COUNT` (default: `8`)
- `STATEDU_MAX_SLIDE_COUNT` (default: `40`)
- `STATEDU_RENDER_STEP_DELAY_SEC` (default: `0.35`)
- `STATEDU_SLIDE_AUTO_SPLIT_ENABLED` (`1` or `0`, default: `1`)
- `STATEDU_WEB_RESEARCH_ENABLED` (`1` or `0`, default: `1`)
- `STATEDU_WEB_RESEARCH_MAX_RESULTS` (default: `5`)
- `STATEDU_WEB_RESEARCH_TIMEOUT_SEC` (default: `6`)
- `STATEDU_IMAGE_GENERATION_ENABLED` (`1` or `0`, default: `1`)
- `STATEDU_IMAGE_PROVIDER` (`local`, `openai`, `none`; default: `local`)
- `STATEDU_IMAGE_MAX_SLIDES` (default: `12`)
- `STATEDU_IMAGE_STYLE_PROMPT` (style hint for generated visuals)
- `STATEDU_OPENAI_IMAGE_MODEL` (default: `gpt-image-1`)
- `STATEDU_OPENAI_IMAGE_SIZE` (default: `1536x1024`)

## Notes

- Keep API keys in environment variables only.
- Generated artifacts are stored under `.statedu/` and are git-ignored by default.
