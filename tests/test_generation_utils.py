import unittest

from generation_utils import (
    DEFAULT_GENERATION_MODELS,
    PartialStreamError,
    generate_text_stream_with_fallback,
    is_model_quota_error,
    is_model_unavailable_error,
    is_transient_generation_error,
    parse_generation_models,
)


class FakeError(Exception):
    pass


class FakeChunk:
    def __init__(self, text):
        self.text = text


class GenerationUtilsTests(unittest.TestCase):
    def test_default_chain_uses_flash_then_current_flash_lite(self):
        self.assertEqual(parse_generation_models(), DEFAULT_GENERATION_MODELS)
        self.assertEqual(
            DEFAULT_GENERATION_MODELS,
            ("gemini-2.5-flash", "gemini-3.5-flash-lite"),
        )

    def test_configured_chain_is_trimmed_and_deduplicated(self):
        self.assertEqual(
            parse_generation_models(" gemini-a,gemini-b,gemini-a "),
            ("gemini-a", "gemini-b"),
        )

    def test_daily_quota_switches_model_without_retry(self):
        error = FakeError(
            "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        self.assertTrue(is_model_quota_error(error))
        self.assertFalse(is_transient_generation_error(error))

    def test_server_error_is_retryable(self):
        self.assertTrue(is_transient_generation_error(FakeError("503 UNAVAILABLE")))

    def test_missing_model_can_fall_back(self):
        self.assertTrue(is_model_unavailable_error(FakeError("404 NOT_FOUND model")))

    def test_stream_returns_complete_answer(self):
        deltas = []
        answer, model, sources = generate_text_stream_with_fallback(
            ("model-a",),
            start_stream=lambda model_name: [FakeChunk("สวัสดี"), FakeChunk("ครับ")],
            reserve_slot=lambda: None,
            on_delta=deltas.append,
        )
        self.assertEqual(answer, "สวัสดีครับ")
        self.assertEqual(model, "model-a")
        self.assertEqual(deltas, ["สวัสดี", "ครับ"])
        self.assertEqual(sources, [])

    def test_stream_quota_falls_back_before_first_token(self):
        attempted = []

        def start_stream(model_name):
            attempted.append(model_name)
            if model_name == "model-a":
                raise FakeError("429 RESOURCE_EXHAUSTED quota exceeded")
            return [FakeChunk("สำเร็จ")]

        answer, model, _ = generate_text_stream_with_fallback(
            ("model-a", "model-b"),
            start_stream=start_stream,
            reserve_slot=lambda: None,
            on_delta=lambda delta: None,
        )
        self.assertEqual(answer, "สำเร็จ")
        self.assertEqual(model, "model-b")
        self.assertEqual(attempted, ["model-a", "model-b"])

    def test_partial_stream_is_never_retried_or_persisted_as_complete(self):
        attempted = []

        def broken_stream(model_name):
            attempted.append(model_name)
            yield FakeChunk("คำตอบครึ่ง")
            raise FakeError("503 UNAVAILABLE")

        with self.assertRaises(PartialStreamError):
            generate_text_stream_with_fallback(
                ("model-a", "model-b"),
                start_stream=broken_stream,
                reserve_slot=lambda: None,
                on_delta=lambda delta: None,
                sleep_fn=lambda seconds: None,
            )
        self.assertEqual(attempted, ["model-a"])


if __name__ == "__main__":
    unittest.main()
