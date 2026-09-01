# Cbot performance improvement report

## Before and after

| Area | Before | After |
|---|---|---|
| Server initialization | Database/admin bootstrap ran from top-level code on every rerun | Cached once per Streamlit server process |
| Normal sidebar | Separate chat list, title, pin, and mode queries | One `list_chat_summaries` query |
| Chat persistence | Delete every message and insert the full history after each answer | Lock the chat and append only new messages |
| Chat rendering | Load and render the complete conversation | Load 50 newest messages and page older messages on demand |
| RAG retrieval | Semantic and lexical candidates used two SQL executions | Both candidate sets use one SQL CTE/round trip |
| Answer display | Wait for the complete Gemini response | Stream text as it arrives; discard interrupted partial output |
| Repeated questions | Cache table existed but the active response path did not use it | Successful answers and Knowledge sources are read/written through cache |
| Busy AI queue | A request could appear stuck for up to 20 seconds | Wait at most 3 seconds and show an estimated retry time |
| Admin ingestion | Multiple sessions could start CPU-heavy local embeddings | One shared ingestion lock; resumable job data remains in Neon |
| Slow/failed Gemini request | SDK and app retries could stack and leave the user waiting close to a minute | One provider attempt with a 20-second HTTP timeout, visible retry action, and incident ID |
| C code formatting | A model could flatten a `//` comment and the next statement onto one line | Prompt rule plus a deterministic fenced-C guard separates commented executable statements |
| Mode feedback | A database write could leave the old badge visible with no feedback | Spinner during the write, success toast, and short-lived cached chat summaries |
| Stale incorrect answers | An earlier bad answer could remain in the answer cache | Versioned cache keys bypass the previous cache generation safely |
| Completed answer visibility | Answer content was rendered inside a collapsed status box | Status and answer now render separately, so completed content remains visible |
| Retry visibility | A retry created after the controls had rendered appeared only on a later interaction | Failed turns rerun once and reveal the retry action immediately |
| Flattened inline C | The first guard only covered fenced `c` blocks | Inline/plain flattened C is repaired and Hello World receives a complete verified example |

## Verification result

- Source compilation validation for `app.py`, `db.py`, `generation_utils.py`, and
  `prompt.py` passed. (`py_compile` could not replace one locked Windows cache file,
  so the same Python compiler was run without writing bytecode.)
- `python -m unittest discover -s tests -v` ran 30 tests: 28 passed and
  2 OCR/file tests were skipped on the inspection machine because pandas is not
  installed there.
- Added coverage for chat summary mapping, append ownership, sequence allocation,
  50-message paging, cached Knowledge sources, streaming success, model fallback,
  interrupted streams, and the C comment/code formatting regression.

## Live measurement

Open **ประสิทธิภาพล่าสุด** in the Admin sidebar after asking a question. Record
the following values before a presentation or load test:

| Scenario | Page prepare | Embedding | Database retrieval | Gemini | Save | Total |
|---|---:|---:|---:|---:|---:|---:|
| Warm page, one user | | | | | | |
| Repeated cached question | | | | | | |
| Knowledge question | | | | | | |
| Two simultaneous users | | | | | | |
| Five simultaneous users | | | | | | |

Cold-start time after Streamlit Community Cloud sleeps is platform-controlled;
keep it separate from warm-run measurements.
