import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from contextvars import Context

import config
from llm import build_followup_context
from llm_utils import fetch_llama_cpp_models, fetch_custom_api_models, get_model_choices, resolve_model_config, set_provider_settings
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
    def setUp(self):
        set_provider_settings(None)

    def tearDown(self):
        set_provider_settings(None)

    def test_llama_discovery_never_sends_openai_credential(self):
        response = Mock()
        response.json.return_value = {"data": [{"id": "local-model"}]}
        with patch.object(config, "LLAMA_CPP_BASE_URL", "https://llama.example/v1/"), patch.object(
            config, "OPENAI_API_KEY", "private-openai-key"
        ), patch("llm_utils.requests.get", return_value=response) as get:
            self.assertEqual(fetch_llama_cpp_models(), ["local-model"])
        get.assert_called_once_with("https://llama.example/v1/models", timeout=3)

    def test_llama_inference_uses_only_placeholder_key(self):
        response = Mock()
        response.json.return_value = {"data": [{"id": "local-model"}]}
        with patch.object(config, "LLAMA_CPP_BASE_URL", "https://llama.example"), patch.object(
            config, "OPENAI_API_KEY", "private-openai-key"
        ), patch("llm_utils.requests.get", return_value=response):
            params = resolve_model_config("local-model")["constructor_params"]
        self.assertEqual(params["base_url"], "https://llama.example/v1")
        self.assertEqual(params["api_key"], "sk-local")
        self.assertNotIn("private-openai-key", str(params))

    def test_custom_discovery_uses_only_its_own_key(self):
        response = Mock()
        response.json.return_value = {"data": [{"id": "custom-model"}]}
        with patch.object(config, "CUSTOM_API_BASE_URL", "https://custom.example"), patch.object(
            config, "CUSTOM_API_KEY", "custom-key"
        ), patch.object(config, "OPENAI_API_KEY", "private-openai-key"), patch(
            "llm_utils.requests.get", return_value=response
        ) as get:
            self.assertEqual(fetch_custom_api_models(), ["custom-model"])
        get.assert_called_once_with(
            "https://custom.example/v1/models",
            headers={"Authorization": "Bearer custom-key"},
            timeout=3,
        )

    def test_provider_settings_do_not_cross_sessions(self):
        def resolve_with(key):
            set_provider_settings({"OPENAI_API_KEY": key})
            return resolve_model_config("gpt-5-mini")["constructor_params"]["api_key"]

        first = Context()
        second = Context()
        self.assertEqual(first.run(resolve_with, "first-key"), "first-key")
        self.assertEqual(second.run(resolve_with, "second-key"), "second-key")
        self.assertEqual(first.run(lambda: resolve_model_config("gpt-5-mini")["constructor_params"]["api_key"]), "first-key")
        self.assertEqual(second.run(lambda: resolve_model_config("gpt-5-mini")["constructor_params"]["api_key"]), "second-key")

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
        set_provider_settings({
            "CUSTOM_API_BASE_URL": "https://provider.example/v1",
            "CUSTOM_API_KEY": "session-only-key",
            "CUSTOM_API_MODEL": "session-model",
        })
        with patch("llm_utils.fetch_ollama_models", return_value=[]), patch(
            "llm_utils.fetch_llama_cpp_models", return_value=[]
        ), patch("llm_utils.fetch_custom_api_models", return_value=[]):
            choices = get_model_choices()
            resolved = resolve_model_config("SESSION-MODEL")
        self.assertIn("session-model", choices)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["constructor_params"]["model_name"], "session-model")
        self.assertEqual(resolved["constructor_params"]["api_key"], "session-only-key")

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

    def test_pivot_run_requires_lawful_use_acknowledgement(self):
        import ui

        self.assertFalse(
            ui._should_run_investigation(
                "pivot query",
                "session-model",
                False,
                "pivot query",
                False,
            )
        )
        self.assertTrue(
            ui._should_run_investigation(
                "pivot query",
                "session-model",
                False,
                "pivot query",
                True,
            )
        )

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
