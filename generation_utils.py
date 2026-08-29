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
