import re
import time


DEFAULT_GENERATION_MODELS = (
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
)


def parse_generation_models(configured_value=None):
    """Return an ordered, duplicate-free model fallback chain."""
    candidates = (
        str(configured_value).split(",")
        if configured_value
        else DEFAULT_GENERATION_MODELS
    )
    models = []
    for candidate in candidates:
        model = str(candidate).strip()
        if model and model not in models:
            models.append(model)
    return tuple(models) or DEFAULT_GENERATION_MODELS


def generation_error_text(exc):
    return f"{type(exc).__name__}: {exc}".lower()


def is_model_quota_error(exc):
    message = generation_error_text(exc)
    return "429" in message and any(marker in message for marker in (
        "resource_exhausted",
        "generaterequestsperdayperprojectpermodel",
        "generate_content_free_tier_requests",
        "quota exceeded",
    ))


def is_model_unavailable_error(exc):
    message = generation_error_text(exc)
    return "404" in message and any(marker in message for marker in (
        "not_found", "not found", "model",
    ))


def is_transient_generation_error(exc):
    """Retry provider/network failures, but never retry quota errors on the same model."""
    if is_model_quota_error(exc):
        return False
    message = generation_error_text(exc)
    return any(marker in message for marker in (
        "408", "500", "502", "503", "504", "unavailable",
        "deadline", "timeout", "temporarily", "connection reset",
    ))


class PartialStreamError(RuntimeError):
    """Raised when a provider fails after already emitting answer text."""


_C_FENCE_RE = re.compile(r"(```c\s*\n)(.*?)(\n```)", re.IGNORECASE | re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_C_STATEMENT_AFTER_COMMENT_RE = re.compile(
    r"\b(?:printf|puts|scanf|fprintf|fputs|fputc)\s*\([^;\n]*\)\s*;"
    r"|\breturn\s+[^;\n]+;"
)


def _split_flattened_c_line(line):
    stripped = line.lstrip()
    lowered = stripped.lower()
    if not stripped.startswith("//") or any(
        marker in lowered for marker in ("example:", "e.g.", "เช่น")
    ):
        return None
    statement = _C_STATEMENT_AFTER_COMMENT_RE.search(stripped[2:])
    if not statement:
        return None
    comment = stripped[:statement.start() + 2].rstrip()
    executable = stripped[statement.start() + 2:].lstrip()
    comment_lower = comment.lower()
    if not (
        '"' in comment
        or "'" in comment
        or "this prints" in comment_lower
        or "แสดง" in comment
    ):
        return None
    indent = line[:len(line) - len(stripped)]
    executable = re.sub(
        r";\s+(?=(?:return|printf|puts|scanf|fprintf|fputs|fputc)\b)",
        ";\n" + indent,
        executable,
    )
    return indent + comment + "\n" + indent + executable


def repair_c_code_blocks(text):
    """Repair a common LLM formatting error inside fenced C examples.

    Models occasionally flatten a source comment and the following executable
    statement onto one ``//`` line.  C then comments out that statement.  This
    guard only touches fenced C blocks and leaves explicit prose examples alone.
    """
    def repair_block(match):
        fixed_lines = []
        for line in match.group(2).splitlines():
            fixed_lines.append(_split_flattened_c_line(line) or line)
        return match.group(1) + "\n".join(fixed_lines) + match.group(3)

    repaired = _C_FENCE_RE.sub(repair_block, str(text or ""))

    def repair_inline(match):
        fixed = _split_flattened_c_line(match.group(1))
        if not fixed:
            return match.group(0)
        return f"```c\n{fixed}\n```"

    repaired = _INLINE_CODE_RE.sub(repair_inline, repaired)

    def repair_plain(segment):
        output = []
        for line in segment.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            ending = line[len(body):]
            output.append((_split_flattened_c_line(body) or body) + ending)
        return "".join(output)

    pieces = []
    cursor = 0
    for fence in _ANY_FENCE_RE.finditer(repaired):
        pieces.append(repair_plain(repaired[cursor:fence.start()]))
        pieces.append(fence.group(0))
        cursor = fence.end()
    pieces.append(repair_plain(repaired[cursor:]))
    return "".join(pieces)


def ensure_c_hello_world_example(prompt, answer):
    """Append a complete verified C example when a Hello World answer lacks one."""
    if "hello world" not in str(prompt or "").lower():
        return answer
    normalized = str(answer or "").lower()
    has_complete_example = all(marker in normalized for marker in (
        "#include <stdio.h>", "int main", "printf", "hello world", "return 0;",
    ))
    if has_complete_example:
        return answer
    verified = (
        "ตัวอย่างภาษา C ที่สมบูรณ์และคอมไพล์ได้:\n\n"
        "```c\n#include <stdio.h>\n\nint main(void) {\n"
        "    printf(\"Hello World\\n\");\n    return 0;\n}\n```\n\n"
        "ผลลัพธ์:\n\n```text\nHello World\n```"
    )
    return str(answer or "").rstrip() + "\n\n" + verified


def generate_text_stream_with_fallback(
    models,
    start_stream,
    reserve_slot,
    on_delta,
    extract_sources=None,
    max_attempts_per_model=2,
    sleep_fn=time.sleep,
):
    """Consume a text stream with model fallback before the first token.

    A stream that fails after emitting text is never retried because doing so
    would duplicate or mix two answers. Callers can replace the partial UI and
    must not persist that incomplete answer.
    """
    last_error = None
    for model_index, model_name in enumerate(models):
        for attempt in range(max_attempts_per_model):
            reserve_slot()
            parts = []
            sources = []
            seen_sources = set()
            try:
                for chunk in start_stream(model_name):
                    try:
                        delta = getattr(chunk, "text", None) or ""
                    except Exception:
                        # Grounding-only/final chunks can legitimately have no
                        # text candidate while still carrying source metadata.
                        delta = ""
                    if delta:
                        parts.append(delta)
                        on_delta(delta)
                    if extract_sources:
                        for source in extract_sources(chunk) or []:
                            key = source.get("uri") if isinstance(source, dict) else str(source)
                            if key and key not in seen_sources:
                                seen_sources.add(key)
                                sources.append(source)
                answer = "".join(parts).strip()
                if not answer:
                    raise RuntimeError("AI provider returned an empty response")
                return answer, model_name, sources
            except Exception as exc:
                last_error = exc
                if parts:
                    raise PartialStreamError(str(exc)) from exc
                if is_model_quota_error(exc) or is_model_unavailable_error(exc):
                    break
                if not is_transient_generation_error(exc) or attempt == max_attempts_per_model - 1:
                    raise
                sleep_fn(1.5 * (2 ** attempt))
    if last_error:
        raise last_error
    raise RuntimeError("no generation models configured")
