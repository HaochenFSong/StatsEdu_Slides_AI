# StatEdu Quarto Slides Helper GENAI

StatEdu Slides Helper AI is a human-in-the-loop web app for building classroom-ready statistics slide decks from prompts and source materials (textbook excerpts, notes, and existing slides).

The app generates editable Quarto Reveal.js slides, renders live previews, supports iterative refinement, and keeps users in control with slide lock/approve controls.

## Core Features

- Prompt-to-deck generation for statistics lessons
- Multi-stage GenAI pipeline for quality:
  - `template_generation`
  - `content_generation`
  - `review_stage`
  - `correction_stage`
  - `rendering`
- Source upload support (`.pdf`, `.pptx`, `.md`, `.txt`, images)
- Human-in-the-loop refine workflow
- Teaching style profiles:
  - `balanced`
  - `conceptual`
  - `mathematical`
  - `simulation`
- Quarto `.qmd` output with live preview
- Audience-facing slide language (no presenter coaching text in visible slide body)
- Content structure emphasis per slide: definition, context, and student-useful materials
- R-enabled simulation/plot slides (e.g., histogram, scatter, cluster plot) for concept illustration
- Base-R-safe rendering path for generated `rChunk` code (auto-avoids missing tidyverse/ggplot dependencies)
- In-app `.qmd` editing with re-render
- Slide-level protection during refine (`approve` / `lock`)
- Stage-aware progress tracking in workspace and cover page pipeline panel
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
- `STATEDU_DEFAULT_SLIDE_COUNT` (default: `8`)
- `STATEDU_MAX_SLIDE_COUNT` (default: `40`)
- `STATEDU_RENDER_STEP_DELAY_SEC` (default: `0.35`)

## Notes

- Keep API keys in environment variables only.
- Generated artifacts are stored under `.statedu/` and are git-ignored by default.
