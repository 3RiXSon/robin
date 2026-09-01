import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from llm import build_followup_context
from llm_utils import get_model_choices, resolve_model_config
from search import fetch_search_results
from scrape import scrape_multiple


class _FakeResponse:
    status_code = 200
    text = '<a href="http://examplehidden.onion/report">Incident report</a>'


class _FakeSession:
    def __init__(self):
        self.last_url = None

    def get(self, url, headers=None, timeout=None):
        self.last_url = url
        return _FakeResponse()


class RobinCoreTests(unittest.TestCase):
    def test_followup_context_keeps_scraped_dictionary_values(self):
        context = build_followup_context(
            "actor alias",
            "actor alias forum",
            [{"title": "Forum", "link": "http://forum.onion/post"}],
            {"http://forum.onion/post": "observed artifact text"},
            "Grounded summary",
        )
        self.assertIn("SOURCE: http://forum.onion/post", context)
        self.assertIn("observed artifact text", context)

    def test_search_query_is_url_encoded_before_fetch(self):
        session = _FakeSession()
        with patch("search.get_tor_session", return_value=session):
            results = fetch_search_results(
                "http://search.onion/find?q={query}",
                "ransomware & identity",
            )
        self.assertEqual(len(results), 1)
        self.assertIn("ransomware+%26+identity", session.last_url)

    def test_scrape_multiple_ignores_invalid_urls_without_crashing(self):
        self.assertEqual(scrape_multiple([{"title": "bad", "link": "file:///tmp/a"}]), {})
        self.assertEqual(scrape_multiple("not a list"), {})

    def test_runtime_custom_provider_can_supply_a_model_without_restart(self):
        original = {
            "CUSTOM_API_BASE_URL": config.CUSTOM_API_BASE_URL,
            "CUSTOM_API_KEY": config.CUSTOM_API_KEY,
            "CUSTOM_API_MODEL": config.CUSTOM_API_MODEL,
        }
        try:
            config.CUSTOM_API_BASE_URL = "https://provider.example/v1"
            config.CUSTOM_API_KEY = "session-only-key"
            config.CUSTOM_API_MODEL = "session-model"
            with patch("llm_utils.fetch_ollama_models", return_value=[]), patch(
                "llm_utils.fetch_llama_cpp_models", return_value=[]
            ), patch("llm_utils.fetch_custom_api_models", return_value=[]):
                choices = get_model_choices()
                resolved = resolve_model_config("SESSION-MODEL")
            self.assertIn("session-model", choices)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved["constructor_params"]["model_name"], "session-model")
            self.assertEqual(resolved["constructor_params"]["api_key"], "session-only-key")
        finally:
            for name, value in original.items():
                setattr(config, name, value)

    def test_local_report_archive_round_trips_and_exports(self):
        # Importing ui is intentionally avoided at module load so the core tests
        # stay lightweight; the Streamlit smoke test below covers its startup.
        import ui

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ui, "INVESTIGATIONS_DIR", Path(temp_dir)):
                filename = ui.save_investigation(
                    "actor alias",
                    "actor alias forum",
                    "session-model",
                    "🔍 Threat intelligence",
                    [{"title": "Forum", "link": "http://forum.onion/post"}],
                    "## Findings\n- observed artifact",
                    scraped={"http://forum.onion/post": "evidence text"},
                    results_count=4,
                    save_raw=True,
                )
                saved = ui.load_investigations()
                self.assertEqual(len(saved), 1)
                normalized = ui._investigation_from_saved(saved[0])
                self.assertEqual(normalized["filename"], filename)
                self.assertIn("evidence text", ui.build_json_report(normalized))
                self.assertIn("## Findings", ui.build_markdown_report(normalized))

    def test_aaa_ui_starts_without_provider_configuration(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui.py")).run(timeout=30)
        self.assertFalse(app.exception, [error.value for error in app.exception])
        self.assertIn("Run search", [button.label for button in app.button])
        self.assertTrue(app.button[0].disabled)
        app.button[1].click().run(timeout=30)
        self.assertEqual(app.text_input[0].value, "threat actor alias")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
