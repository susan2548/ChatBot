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
