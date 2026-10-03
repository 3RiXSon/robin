import config
from contextvars import ContextVar
import requests
from urllib.parse import urljoin
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Callable, Optional, List
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler
import os
from config import (
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    LLAMA_CPP_BASE_URL,
)


class BufferedStreamingHandler(BaseCallbackHandler):
    def __init__(self, buffer_limit: int = 60, ui_callback: Optional[Callable[[str], None]] = None):
        self.buffer = ""
        self.buffer_limit = buffer_limit
        self.ui_callback = ui_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer += token
        if "\n" in token or len(self.buffer) >= self.buffer_limit:
            print(self.buffer, end="", flush=True)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""

    def on_llm_end(self, response, **kwargs) -> None:
        if self.buffer:
            print(self.buffer, end="", flush=True)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""


# --- Configuration Data ---
# Instantiate common dependencies once
_common_callbacks = [BufferedStreamingHandler(buffer_limit=60)]

# Define common parameters for most LLMs
_common_llm_params = {
    "temperature": 0,
    "streaming": True,
    "callbacks": _common_callbacks,
}

# Map input model choices (lowercased) to their configuration
# Each config includes the class and any model-specific constructor parameters
_llm_config_map = {
    'gpt-4.1': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-4.1'} 
    },
    'gpt-5.2': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.2'} 
    },
    'gpt-5.1': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.1'} 
    },
    'gpt-5-mini': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-mini'} 
    },
    'gpt-5-nano': { 
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-nano'} 
    },
    'claude-sonnet-4-5': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-5'}
    },
    'claude-sonnet-4-0': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-0'}
    },
    'gemini-2.5-flash': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash', 'google_api_key': GOOGLE_API_KEY }
    },
    'gemini-2.5-flash-lite': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash-lite', 'google_api_key': GOOGLE_API_KEY}
    },
    'gemini-2.5-pro': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-pro', 'google_api_key': GOOGLE_API_KEY}
    },
    'qwen3-80b-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'qwen/qwen3-next-80b-a3b-instruct:free',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    'nemotron-nano-9b-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'nvidia/nemotron-nano-9b-v2:free',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    'gpt-oss-120b-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'openai/gpt-oss-120b:free',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    'gpt-5.1-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'openai/gpt-5.1',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    'gpt-5-mini-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'openai/gpt-5-mini',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    'claude-sonnet-4.5-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'anthropic/claude-sonnet-4.5',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    'grok-4.1-fast-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'x-ai/grok-4.1-fast',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY  # Use OpenRouter API key
        }
    },
    # 'llama3.2': {
    #     'class': ChatOllama,
    #     'constructor_params': {'model': 'llama3.2:latest', 'base_url': OLLAMA_BASE_URL}
    # },
    # 'llama3.1': {
    #     'class': ChatOllama,
    #     'constructor_params': {'model': 'llama3.1:latest', 'base_url': OLLAMA_BASE_URL}
    # },
    # 'gemma3': {
    #     'class': ChatOllama,
    #     'constructor_params': {'model': 'gemma3:latest', 'base_url': OLLAMA_BASE_URL}
    # },
    # 'deepseek-r1': {
    #     'class': ChatOllama,
    #     'constructor_params': {'model': 'deepseek-r1:latest', 'base_url': OLLAMA_BASE_URL}
    # },
    
    # Add more models here easily:
    # 'mistral7b': {
    #     'class': ChatOllama,
    #     'constructor_params': {'model': 'mistral:7b', 'base_url': OLLAMA_BASE_URL}
    # },
    # 'gpt3.5': {
    #      'class': ChatOpenAI,
    #      'constructor_params': {'model_name': 'gpt-3.5-turbo', 'base_url': OLLAMA_BASE_URL}
    # }
}


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


_provider_settings = ContextVar("robin_provider_settings", default=None)


def set_provider_settings(settings):
    """Set provider values for the current Streamlit script context."""
    _provider_settings.set(None if settings is None else dict(settings))


def _runtime_config(name: str, fallback=None):
    """Use this session's provider values, falling back to startup configuration."""
    settings = _provider_settings.get()
    if settings is not None and name in settings:
        return settings[name]
    return getattr(config, name, fallback)


def _get_ollama_base_url() -> Optional[str]:
    base_url = _runtime_config("OLLAMA_BASE_URL")
    if not base_url:
        return None
    return str(base_url).rstrip("/") + "/"


def fetch_ollama_models() -> List[str]:
    """
    Retrieve the list of locally available Ollama models by querying the Ollama HTTP API.
    Returns an empty list if the API isn't reachable or the base URL is not defined.
    """
    base_url = _get_ollama_base_url()
    if not base_url:
        return []

    try:
        resp = requests.get(urljoin(base_url, "api/tags"), timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        available = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                available.append(name)
        return available
    except (requests.RequestException, ValueError):
        import logging
        current_ollama_url = _runtime_config("OLLAMA_BASE_URL")
        if current_ollama_url and ("localhost" in str(current_ollama_url).lower() or "127.0.0.1" in str(current_ollama_url).lower()):
            logging.warning(
                "Ollama unreachable at %s. If running Robin in Docker, use "
                "http://host.docker.internal:<port> instead of localhost.", current_ollama_url
            )
        return []


# Added Support for llama.cpp models since they use OpenAI-compatible API
def fetch_llama_cpp_models() -> List[str]:
    """
    Retrieve available models from an OpenAI-compatible llama.cpp server.
    Uses /v1/models.
    """
    llama_base_url = _runtime_config("LLAMA_CPP_BASE_URL")
    if not llama_base_url:
        return []

    base = str(llama_base_url).rstrip("/")
    api_base = base[:-3] if base.lower().endswith("/v1") else base
    try:
        resp = requests.get(f"{api_base}/v1/models", timeout=3)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [m["id"] for m in data if "id" in m]
    except (requests.RequestException, ValueError, KeyError):
        return []


def fetch_custom_api_models() -> List[str]:
    """Retrieve models from any OpenAI-compatible API endpoint."""
    custom_base_url = _runtime_config("CUSTOM_API_BASE_URL")
    if not custom_base_url:
        return []
    base = str(custom_base_url).rstrip("/")
    if not base.lower().endswith("/v1"):
        base += "/v1"
    try:
        headers = {}
        api_key = _runtime_config("CUSTOM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(f"{base}/models", headers=headers, timeout=3)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if "id" in m]
    except (requests.RequestException, ValueError, KeyError):
        return []


def _is_set(v: Optional[str]) -> bool:
    return bool(v and str(v).strip() and "your_" not in str(v))


# Provider prefixes for disambiguating colliding model names
_PROVIDER_PREFIX_OLLAMA = "ollama:"
_PROVIDER_PREFIX_LLAMA_CPP = "llama.cpp:"
_PROVIDER_PREFIX_CUSTOM = "custom:"


def _strip_provider_prefix(model_choice: str) -> tuple:
    """Return (provider_prefix, raw_model_name) from a possibly-prefixed model choice."""
    model_choice = model_choice.strip()
    for prefix in (_PROVIDER_PREFIX_CUSTOM, _PROVIDER_PREFIX_LLAMA_CPP, _PROVIDER_PREFIX_OLLAMA):
        if model_choice.lower().startswith(prefix):
            return prefix, model_choice[len(prefix):]
    return "", model_choice


# Changed it so the GUI only loaded available models
def get_model_choices() -> List[str]:
    """
    Combine configured cloud models with locally available Ollama models.
    Cloud models are shown only if required API keys are present.
    
    Dynamic models (ollama, llama.cpp, custom) that collide with built-in names
    are prefixed with their provider (e.g., 'custom:gpt-4.1') to preserve identity.
    """
    gated_base_models: List[str] = []

    openai_ok = _is_set(_runtime_config("OPENAI_API_KEY"))
    anthropic_ok = _is_set(_runtime_config("ANTHROPIC_API_KEY"))
    google_ok = _is_set(_runtime_config("GOOGLE_API_KEY"))
    openrouter_ok = _is_set(_runtime_config("OPENROUTER_API_KEY")) and _is_set(_runtime_config("OPENROUTER_BASE_URL"))

    for k, cfg in _llm_config_map.items():
        cls = cfg.get("class")
        ctor = cfg.get("constructor_params", {}) or {}

        # OpenRouter models (ChatOpenAI with base_url set to OpenRouter)
        if cls is ChatOpenAI and (ctor.get("base_url") == OPENROUTER_BASE_URL or "openrouter" in k):
            if openrouter_ok:
                gated_base_models.append(k)
            continue

        # Direct OpenAI models
        if cls is ChatOpenAI:
            if openai_ok:
                gated_base_models.append(k)
            continue

        # Anthropic
        if cls is ChatAnthropic:
            if anthropic_ok:
                gated_base_models.append(k)
            continue

        # Google Gemini
        if cls is ChatGoogleGenerativeAI:
            if google_ok:
                gated_base_models.append(k)
            continue

        # Anything else: keep
        gated_base_models.append(k)

    # Build a set of normalized built-in model names for collision detection
    builtin_normalized = {_normalize_model_name(m) for m in gated_base_models}

    # Collect dynamic models with their provider prefix
    dynamic_models: List[str] = []

    # Dynamic local models via Ollama-style API (/api/tags)
    for m in fetch_ollama_models():
        if _normalize_model_name(m) in builtin_normalized:
            dynamic_models.append(f"{_PROVIDER_PREFIX_OLLAMA}{m}")
        else:
            dynamic_models.append(m)

    # Dynamic local models via llama.cpp which uses OpenAI style API
    for m in fetch_llama_cpp_models():
        if _normalize_model_name(m) in builtin_normalized:
            dynamic_models.append(f"{_PROVIDER_PREFIX_LLAMA_CPP}{m}")
        else:
            dynamic_models.append(m)

    # Custom API models
    for m in fetch_custom_api_models():
        if _normalize_model_name(m) in builtin_normalized:
            dynamic_models.append(f"{_PROVIDER_PREFIX_CUSTOM}{m}")
        else:
            dynamic_models.append(m)

    # Manual model from sidebar — add it if not already discovered
    custom_model_value = _runtime_config("CUSTOM_API_MODEL")
    custom_base_value = _runtime_config("CUSTOM_API_BASE_URL")
    if custom_base_value and custom_model_value and str(custom_model_value).strip():
        manual = str(custom_model_value).strip()
        manual_normalized = _normalize_model_name(manual)
        existing_normalized = {_normalize_model_name(m) for m in dynamic_models}
        if manual_normalized not in existing_normalized:
            # Check if manual model collides with built-in
            if manual_normalized in builtin_normalized:
                dynamic_models.append(f"{_PROVIDER_PREFIX_CUSTOM}{manual}")
            else:
                dynamic_models.append(manual)

    # Deduplicate dynamic models while preserving order and preferring prefixed versions
    normalized = {_normalize_model_name(m): m for m in gated_base_models}
    for dm in dynamic_models:
        # For prefixed models, use the full prefixed name as the key to avoid collision
        _, raw_name = _strip_provider_prefix(dm)
        raw_key = _normalize_model_name(raw_name)
        # Use the full model string as the unique key if it has a prefix
        full_key = _normalize_model_name(dm)
        if full_key not in normalized and raw_key not in normalized:
            normalized[full_key] = dm
        elif dm.startswith((_PROVIDER_PREFIX_CUSTOM, _PROVIDER_PREFIX_LLAMA_CPP, _PROVIDER_PREFIX_OLLAMA)):
            # Prefixed models should always be added as separate entries
            normalized[full_key] = dm

    ordered_dynamic = sorted(
        [name for key, name in normalized.items() if name not in gated_base_models],
        key=_normalize_model_name,
    )
    return gated_base_models + ordered_dynamic




def resolve_model_config(model_choice: str):
    """
    Resolve a model choice (case-insensitive) to the corresponding configuration.
    Supports both the predefined remote models and any locally installed Ollama models.
    
    Provider-prefixed model names (e.g., 'custom:gpt-4.1') are routed directly to
    that provider, bypassing the built-in config lookup.
    """
    model_choice_lower = _normalize_model_name(model_choice)
    
    # Check for provider prefix first — this takes precedence over built-in lookup
    prefix, raw_model = _strip_provider_prefix(model_choice)
    
    if prefix == _PROVIDER_PREFIX_CUSTOM:
        # Route directly to custom API, even if model name matches a built-in
        custom_base_value = _runtime_config("CUSTOM_API_BASE_URL")
        if custom_base_value:
            base = str(custom_base_value).rstrip("/")
            if not base.lower().endswith("/v1"):
                base += "/v1"
            return {
                "class": ChatOpenAI,
                "constructor_params": {
                    "model_name": raw_model,
                    "base_url": base,
                    "api_key": _runtime_config("CUSTOM_API_KEY") or "sk-custom",
                    "streaming": False,
                },
            }
        return None
    
    if prefix == _PROVIDER_PREFIX_LLAMA_CPP:
        # Route directly to llama.cpp
        llama_base = _runtime_config("LLAMA_CPP_BASE_URL")
        if llama_base:
            base = str(llama_base).rstrip("/")
            if not base.lower().endswith("/v1"):
                base += "/v1"
            return {
                "class": ChatOpenAI,
                "constructor_params": {
                    "model_name": raw_model,
                    "base_url": base,
                    "api_key": "sk-local",
                    "streaming": False,
                },
            }
        return None
    
    if prefix == _PROVIDER_PREFIX_OLLAMA:
        # Route directly to Ollama
        ollama_base = _runtime_config("OLLAMA_BASE_URL")
        if ollama_base:
            return {
                "class": ChatOllama,
                "constructor_params": {"model": raw_model, "base_url": ollama_base},
            }
        return None
    
    # No prefix — fall through to built-in lookup, then dynamic discovery
    cfg = _llm_config_map.get(model_choice_lower)
    if cfg:
        # Copy constructor parameters and refresh runtime credentials/URLs. This
        # is what makes provider keys entered in the UI usable immediately.
        resolved = {
            "class": cfg["class"],
            "constructor_params": dict(cfg.get("constructor_params", {}) or {}),
        }
        params = resolved["constructor_params"]
        if "openrouter" in model_choice_lower:
            params["base_url"] = _runtime_config("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL
            params["api_key"] = _runtime_config("OPENROUTER_API_KEY")
        elif resolved["class"] is ChatGoogleGenerativeAI:
            params["google_api_key"] = _runtime_config("GOOGLE_API_KEY")
        elif resolved["class"] is ChatAnthropic:
            params["api_key"] = _runtime_config("ANTHROPIC_API_KEY")
        elif resolved["class"] is ChatOpenAI:
            params["api_key"] = _runtime_config("OPENAI_API_KEY")
        return resolved

    # llama.cpp (OpenAI-compatible)
    for llama_model in fetch_llama_cpp_models():
        if _normalize_model_name(llama_model) == model_choice_lower:
            base = str(_runtime_config("LLAMA_CPP_BASE_URL") or "").rstrip("/")
            if not base.lower().endswith("/v1"):
                base += "/v1"
            return {
                "class": ChatOpenAI,
                "constructor_params": {
                    "model_name": llama_model,
                    "base_url": base,
                    "api_key": "sk-local",
                    "streaming": False,
                },
            }

    # Custom OpenAI-compatible API — manual model name or auto-discovered
    custom_candidates = list(fetch_custom_api_models())
    custom_model_value = _runtime_config("CUSTOM_API_MODEL")
    custom_base_value = _runtime_config("CUSTOM_API_BASE_URL")
    if custom_base_value and custom_model_value and str(custom_model_value).strip():
        manual = str(custom_model_value).strip()
        if _normalize_model_name(manual) not in {_normalize_model_name(m) for m in custom_candidates}:
            custom_candidates.append(manual)
    for custom_model in custom_candidates:
        if _normalize_model_name(custom_model) == model_choice_lower:
            base = str(_runtime_config("CUSTOM_API_BASE_URL") or "").rstrip("/")
            if not base.lower().endswith("/v1"):
                base += "/v1"
            return {
                "class": ChatOpenAI,
                "constructor_params": {
                    "model_name": custom_model,
                    "base_url": base,
                    "api_key": _runtime_config("CUSTOM_API_KEY") or "sk-custom",
                    "streaming": False,
                },
            }

    for ollama_model in fetch_ollama_models():
        if _normalize_model_name(ollama_model) == model_choice_lower:
            return {
                "class": ChatOllama,
                "constructor_params": {"model": ollama_model, "base_url": _runtime_config("OLLAMA_BASE_URL")},
            }

    return None


def get_model_display_names(model_keys: List[str]) -> dict:
    """Return a display label dict mapping model key -> '[provider] model_name'.
    
    For provider-prefixed keys (e.g., 'custom:gpt-4.1'), the display shows the
    raw model name with its provider tag, clearly disambiguating from built-in
    models with the same name.
    """
    ollama_set = set(fetch_ollama_models())
    llama_cpp_set = set(fetch_llama_cpp_models())
    custom_set = set(fetch_custom_api_models())

    display = {}
    for key in model_keys:
        # Check for provider prefix first
        prefix, raw_model = _strip_provider_prefix(key)
        
        if prefix == _PROVIDER_PREFIX_CUSTOM:
            display[key] = f"[custom] {raw_model}"
            continue
        if prefix == _PROVIDER_PREFIX_LLAMA_CPP:
            display[key] = f"[llama.cpp] {raw_model}"
            continue
        if prefix == _PROVIDER_PREFIX_OLLAMA:
            display[key] = f"[ollama] {raw_model}"
            continue
        
        # No prefix — determine provider from config or dynamic sources
        cfg = _llm_config_map.get(_normalize_model_name(key))
        if cfg:
            cls = cfg.get("class")
            ctor = cfg.get("constructor_params", {}) or {}
            base_url = str(ctor.get("base_url", "")).lower()
            if "openrouter" in base_url or "openrouter" in key.lower():
                prefix_label = "openrouter"
            elif cls is ChatAnthropic:
                prefix_label = "anthropic"
            elif cls is ChatGoogleGenerativeAI:
                prefix_label = "google"
            elif cls is ChatOpenAI:
                prefix_label = "openai"
            else:
                prefix_label = "other"
        elif key in ollama_set:
            prefix_label = "ollama"
        elif key in llama_cpp_set:
            prefix_label = "llama.cpp"
        elif key in custom_set:
            prefix_label = "custom"
        else:
            prefix_label = "local"
        display[key] = f"[{prefix_label}] {key}"
    return display