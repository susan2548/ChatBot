import unittest

from generation_utils import (
    DEFAULT_GENERATION_MODELS,
    is_model_quota_error,
    is_model_unavailable_error,
    is_transient_generation_error,
    parse_generation_models,
)


class FakeError(Exception):
    pass


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


if __name__ == "__main__":
    unittest.main()
