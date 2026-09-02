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


def build_direct_knowledge_answer(retrieved, max_sections=3, max_section_chars=1200):
    """Build a safe extractive answer when the generation provider is unavailable.

    This deliberately does not invent or paraphrase facts. It presents the most
    relevant retrieved passages and keeps the normal source list in the UI.
    """
    sections = []
    seen = set()
    for item in retrieved or []:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if "|" in content:
            raw_title, raw_body = content.split("|", 1)
        else:
            raw_title, raw_body = "ข้อมูลที่เกี่ยวข้อง", content
        title = re.sub(r"^\s*หัวข้อ\s*:\s*", "", raw_title, flags=re.IGNORECASE).strip()
        body = re.sub(r"^\s*เนื้อหา\s*:\s*", "", raw_body, flags=re.IGNORECASE).strip()
        key = (title.lower(), re.sub(r"\s+", " ", body.lower())[:240])
        if not body or key in seen:
            continue
        seen.add(key)
        if len(body) > max_section_chars:
            shortened = body[:max_section_chars].rsplit(" ", 1)[0].rstrip()
            body = shortened + "…"
        sections.append((title or "ข้อมูลที่เกี่ยวข้อง", body))
        if len(sections) >= max_sections:
            break

    if not sections:
        return "ไม่พบข้อความที่สามารถแสดงจาก Knowledge ได้"
    if len(sections) == 1:
        content = sections[0][1]
    else:
        content = "\n\n".join(
            f"**{title}**\n\n{body}" for title, body in sections
        )
    return (
        "ขณะนี้บริการ AI ไม่พร้อมใช้งาน จึงแสดงข้อมูลที่ค้นพบจาก Knowledge "
        "โดยตรงโดยไม่แต่งข้อมูลเพิ่ม:\n\n"
        f"{content}\n\n"
        "_คำตอบสำรองนี้ใช้ข้อความจาก Knowledge โดยตรง_"
    )


def generate_text_stream_with_fallback(
    models,
    start_stream,
    reserve_slot,
    on_delta,
    extract_sources=None,
    max_attempts_per_model=2,
    sleep_fn=time.sleep,
    on_reset=None,
):
    """Consume a text stream with model fallback before the first token.

    If a stream fails after emitting text, a different model may be tried only
    when the caller supplies ``on_reset`` to clear the partial UI first. The
    incomplete answer is never returned as successful.
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
                    if on_reset is not None and model_index < len(models) - 1:
                        on_reset()
                        break
                    raise PartialStreamError(str(exc)) from exc
                if is_model_quota_error(exc) or is_model_unavailable_error(exc):
                    break
                if not is_transient_generation_error(exc):
                    raise
                if attempt < max_attempts_per_model - 1:
                    sleep_fn(1.5 * (2 ** attempt))
                    continue
                # The current model exhausted its transient retries. Continue
                # to the next configured model instead of failing immediately.
                break
    if last_error:
        raise last_error
    raise RuntimeError("no generation models configured")
