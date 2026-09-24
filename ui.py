"""Robin's no-code Streamlit workspace.

The search, scrape, and LLM layers stay in their own modules.  This file is the
operator-facing layer: it exposes the complete Robin workflow without requiring
users to edit Python, keeps investigations in a local JSON archive, and makes
all network-heavy actions explicit.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config as _robin_cfg
from health import check_llm_health, check_search_engines, check_tor_proxy
from llm import (
    PRESET_PROMPTS,
    answer_followup,
    build_followup_context,
    filter_results,
    generate_summary,
    get_llm,
    refine_query,
    suggest_pivots,
)
from llm_utils import (
    BufferedStreamingHandler,
    get_model_choices,
    get_model_display_names,
    set_provider_settings,
)
from scrape import scrape_multiple
from search import SEARCH_ENGINES, get_search_results


# ---------------------------------------------------------------------------
# Persistence and report helpers
# ---------------------------------------------------------------------------

INVESTIGATIONS_DIR = Path(os.getenv("ROBIN_INVESTIGATIONS_DIR", "investigations"))

PRESET_OPTIONS = {
    "🔍 Threat intelligence": "threat_intel",
    "🦠 Ransomware / malware": "ransomware_malware",
    "👤 Personal / identity exposure": "personal_identity",
    "🏢 Corporate leaks / espionage": "corporate_espionage",
}
PRESET_PLACEHOLDERS = {
    "threat_intel": "Example: pay extra attention to cryptocurrency wallet addresses and actor aliases.",
    "ransomware_malware": "Example: highlight double-extortion tactics and ransomware-as-a-service affiliates.",
    "personal_identity": "Example: flag exposed emails and phone numbers, but do not repeat sensitive values unnecessarily.",
    "corporate_espionage": "Example: prioritize source-code repositories, API keys, or internal document dumps.",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _as_nonnegative_int(value: Any, fallback: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return fallback


def _safe_slug(value: str, fallback: str = "investigation") -> str:
    """Return a filesystem-safe, human-readable report slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", _clean_text(value).lower()).strip("-")
    return (slug[:54] or fallback).strip("-")


def _json_safe(value: Any) -> Any:
    """Convert common model/search values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_investigation(
    query: str,
    refined_query: str,
    model: str,
    preset_label: str,
    sources: list,
    summary: str,
    *,
    scraped: Optional[dict] = None,
    custom_instructions: str = "",
    results_count: Optional[int] = None,
    save_raw: bool = False,
) -> str:
    """Save a completed investigation to disk and return its filename.

    Raw scraped content is opt-in because it may contain sensitive material.
    The active in-memory workspace always retains it for grounded follow-ups;
    the archive only stores it when the operator selects "save raw evidence".
    Microseconds prevent two runs in the same second from overwriting a report.
    """
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    fname = (
        f"investigation_{now.strftime('%Y%m%d_%H%M%S_%f')}_"
        f"{_safe_slug(query, 'query')}.json"
    )
    data = {
        "schema_version": 2,
        "timestamp": timestamp,
        "query": _clean_text(query),
        "refined_query": _clean_text(refined_query),
        "model": _clean_text(model),
        "preset": _clean_text(preset_label),
        "custom_instructions": _clean_text(custom_instructions),
        "sources": _json_safe(sources or []),
        "summary": _clean_text(summary),
        "results_count": _as_nonnegative_int(results_count, len(sources or [])),
        "raw_saved": bool(save_raw and scraped),
    }
    if save_raw and scraped:
        data["scraped"] = _json_safe(scraped)

    destination = INVESTIGATIONS_DIR / fname
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return fname


def load_investigations() -> list[dict]:
    """Return valid saved investigations sorted newest-first."""
    if not INVESTIGATIONS_DIR.exists():
        return []
    investigations: list[dict] = []
    files = sorted(INVESTIGATIONS_DIR.glob("investigation_*.json"), reverse=True)
    for file_path in files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            data["_filename"] = file_path.name
            investigations.append(data)
        except (OSError, ValueError, TypeError):
            # A partial/corrupt archive item should never prevent the app from
            # opening the rest of the workspace.
            continue
    return investigations


def _preset_key_from_saved(value: str) -> str:
    if value in PRESET_OPTIONS.values():
        return value
    for label, key in PRESET_OPTIONS.items():
        if value == label:
            return key
    # Labels used by the previous Streamlit interface remain loadable.
    legacy_labels = {
        "🔍 Dark Web Threat Intel": "threat_intel",
        "🦠 Ransomware / Malware Focus": "ransomware_malware",
        "👤 Personal / Identity Investigation": "personal_identity",
        "🏢 Corporate Espionage / Data Leaks": "corporate_espionage",
    }
    return legacy_labels.get(value, "threat_intel")


def _investigation_from_saved(saved: dict) -> dict:
    """Normalize old (pre-v2) and current archive records for the UI."""
    sources = saved.get("sources", [])
    scraped = saved.get("scraped")
    return {
        "query": _clean_text(saved.get("query")),
        "refined": _clean_text(saved.get("refined_query")),
        "model": _clean_text(saved.get("model")),
        "preset": _preset_key_from_saved(_clean_text(saved.get("preset"))),
        "preset_label": _clean_text(saved.get("preset")) or "🔍 Threat intelligence",
        "custom_instructions": _clean_text(saved.get("custom_instructions")),
        "sources": sources if isinstance(sources, list) else [],
        "scraped": scraped if isinstance(scraped, (dict, str)) else None,
        "summary": _clean_text(saved.get("summary")),
        "results_count": _as_nonnegative_int(saved.get("results_count"), len(sources) if isinstance(sources, list) else 0),
        "timestamp": _clean_text(saved.get("timestamp")),
        "filename": _clean_text(saved.get("_filename")),
        "raw_saved": bool(saved.get("raw_saved") or scraped),
    }


def build_markdown_report(inv: dict) -> str:
    """Create a portable Markdown report for the download button."""
    sources = inv.get("sources", []) or []
    lines = [
        "# Robin investigation report",
        "",
        f"**Original query:** {_clean_text(inv.get('query'))}",
        f"**Refined query:** {_clean_text(inv.get('refined'))}",
        f"**Model:** {_clean_text(inv.get('model')) or 'Not recorded'}",
        f"**Domain:** {_clean_text(inv.get('preset_label') or inv.get('preset'))}",
        f"**Created:** {_clean_text(inv.get('timestamp')) or datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Sources",
        "",
    ]
    if sources:
        for item in sources:
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or "Untitled source"
            link = _clean_text(item.get("link"))
            lines.append(f"- [{title}]({link})" if link else f"- {title}")
    else:
        lines.append("- No sources were returned.")
    lines.extend(["", "## Findings", "", _clean_text(inv.get("summary")) or "No summary was generated.", ""])
    return "\n".join(lines)


def build_json_report(inv: dict) -> str:
    """Create a JSON export without the internal UI-only fields."""
    exported = {
        "schema_version": 2,
        "timestamp": inv.get("timestamp"),
        "query": inv.get("query"),
        "refined_query": inv.get("refined"),
        "model": inv.get("model"),
        "preset": inv.get("preset"),
        "preset_label": inv.get("preset_label"),
        "custom_instructions": inv.get("custom_instructions", ""),
        "results_count": inv.get("results_count", 0),
        "sources": inv.get("sources", []),
        "summary": inv.get("summary", ""),
    }
    if inv.get("scraped"):
        exported["scraped"] = inv["scraped"]
    return json.dumps(_json_safe(exported), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Cached network calls
# ---------------------------------------------------------------------------

@st.cache_data(ttl=200, show_spinner=False)
def cached_search_results(refined_query: str, threads: int):
    # search.fetch_search_results owns URL encoding; keep the query readable in
    # the cache key and in the saved investigation.
    return get_search_results(refined_query, max_workers=threads)


@st.cache_data(ttl=200, show_spinner=False)
def cached_scrape_multiple(filtered: list, threads: int):
    return scrape_multiple(filtered, max_workers=threads)


# ---------------------------------------------------------------------------
# Runtime provider configuration
# ---------------------------------------------------------------------------

_PROVIDER_FIELDS = {
    "OPENAI_API_KEY": ("provider_openai_key", "OpenAI API key"),
    "ANTHROPIC_API_KEY": ("provider_anthropic_key", "Anthropic API key"),
    "GOOGLE_API_KEY": ("provider_google_key", "Google AI API key"),
    "OPENROUTER_API_KEY": ("provider_openrouter_key", "OpenRouter API key"),
    "OPENROUTER_BASE_URL": ("provider_openrouter_url", "OpenRouter base URL"),
    "OLLAMA_BASE_URL": ("provider_ollama_url", "Ollama URL"),
    "LLAMA_CPP_BASE_URL": ("provider_llama_url", "llama.cpp URL"),
    "CUSTOM_API_BASE_URL": ("custom_api_url", "Custom base URL"),
    "CUSTOM_API_KEY": ("custom_api_key", "Custom API key"),
    "CUSTOM_API_MODEL": ("custom_api_model", "Custom model"),
}


def _env_is_set(value: Any) -> bool:
    text = _clean_text(value)
    return bool(text and "your_" not in text.lower())


def _seed_provider_state() -> None:
    for config_name, (state_key, _label) in _PROVIDER_FIELDS.items():
        if state_key not in st.session_state:
            current = getattr(_robin_cfg, config_name, "")
            st.session_state[state_key] = _clean_text(current) if _env_is_set(current) else ""


def _sync_runtime_config() -> None:
    """Make this session's provider settings available to model resolution."""
    set_provider_settings({
        config_name: _clean_text(st.session_state.get(state_key, "")) or None
        for config_name, (state_key, _label) in _PROVIDER_FIELDS.items()
    })


def _provider_badge(name: str, value: Any, cloud: bool) -> str:
    if _env_is_set(value):
        return f"✅ {name} · ready"
    return f"{('⚠️' if cloud else '○')} {name} · {'needs a key' if cloud else 'optional'}"


# ---------------------------------------------------------------------------
# UI styling and reusable renderers
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Robin · OSINT workspace",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --robin-bg: #070912;
        --robin-panel: #0f1422;
        --robin-panel-2: #151b2c;
        --robin-line: rgba(148, 163, 184, .18);
        --robin-muted: #93a0b8;
        --robin-text: #f4f7fb;
        --robin-violet: #a78bfa;
        --robin-cyan: #59d7e8;
        --robin-green: #62d6a6;
        --robin-amber: #f7c873;
      }
      [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 75% -10%, rgba(86, 62, 158, .24), transparent 32rem), var(--robin-bg);
      }
      [data-testid="stHeader"] { background: rgba(7, 9, 18, .78); }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b101b 0%, #0a0e18 100%);
        border-right: 1px solid var(--robin-line);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
      .block-container { max-width: 1320px; padding: 1.55rem 2.4rem 5rem; }
      h1, h2, h3, h4 { letter-spacing: -.025em; }
      h1 { font-size: clamp(2rem, 4vw, 3.45rem) !important; line-height: 1.05 !important; }
      h2 { font-size: clamp(1.45rem, 2.4vw, 2.2rem) !important; }
      h3 { font-size: 1.2rem !important; }
      .robin-topbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.8rem; }
      .robin-brand { display: flex; align-items: center; gap: .8rem; }
      .robin-mark { display: grid; place-items: center; width: 2.75rem; height: 2.75rem; border-radius: 1rem; background: linear-gradient(145deg, #7c3aed, #312e81); box-shadow: 0 0 28px rgba(124,58,237,.26); font-size: 1.42rem; }
      .robin-wordmark { margin: 0; font-size: 1.28rem; font-weight: 800; letter-spacing: .12em; }
      .robin-submark { margin: .12rem 0 0; color: var(--robin-muted); font-size: .78rem; }
      .robin-pill { display: inline-flex; align-items: center; gap: .4rem; border: 1px solid rgba(98,214,166,.28); border-radius: 999px; color: #b8f5db; background: rgba(55, 147, 111, .12); padding: .45rem .72rem; font-size: .74rem; font-weight: 700; white-space: nowrap; }
      .robin-pill-dot { width: .42rem; height: .42rem; border-radius: 99px; background: var(--robin-green); box-shadow: 0 0 10px var(--robin-green); }
      .hero { padding: clamp(1.35rem, 3vw, 2.7rem); border: 1px solid var(--robin-line); border-radius: 1.35rem; background: linear-gradient(135deg, rgba(23, 29, 49, .92), rgba(12, 17, 30, .88)); box-shadow: 0 20px 60px rgba(0,0,0,.2); overflow: hidden; position: relative; }
      .hero:after { content: ""; position: absolute; width: 18rem; height: 18rem; right: -5rem; top: -7rem; border: 1px solid rgba(167,139,250,.18); border-radius: 50%; box-shadow: 0 0 0 2rem rgba(167,139,250,.04), 0 0 0 4rem rgba(167,139,250,.025); pointer-events: none; }
      .eyebrow { color: var(--robin-cyan); font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .hero h1 { max-width: 760px; margin: .65rem 0 .9rem; }
      .hero-copy { max-width: 650px; color: #b9c3d6; font-size: 1.03rem; line-height: 1.7; }
      .feature-card { height: 100%; min-height: 126px; padding: 1rem; border: 1px solid var(--robin-line); border-radius: 1rem; background: rgba(15,20,34,.72); }
      .feature-icon { font-size: 1.35rem; }
      .feature-title { margin: .4rem 0 .25rem; font-weight: 750; color: var(--robin-text); }
      .feature-copy { color: var(--robin-muted); font-size: .82rem; line-height: 1.45; }
      .search-shell { margin: 1.35rem 0 .65rem; padding: .8rem .9rem .15rem; border: 1px solid rgba(167,139,250,.35); border-radius: 1.05rem; background: rgba(16, 21, 36, .88); box-shadow: 0 0 0 4px rgba(124,58,237,.06); }
      .search-shell label { color: var(--robin-muted) !important; }
      .quick-label { color: var(--robin-muted); font-size: .76rem; margin: .5rem 0 .35rem; }
      .metric-card { min-height: 108px; padding: 1.02rem 1.1rem; border: 1px solid var(--robin-line); border-radius: 1rem; background: linear-gradient(145deg, rgba(21,27,44,.9), rgba(13,18,31,.9)); }
      .metric-label { color: var(--robin-muted); font-size: .73rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
      .metric-value { color: var(--robin-text); font-size: 1.78rem; font-weight: 800; line-height: 1.2; margin-top: .32rem; }
      .metric-hint { color: #7f8ca5; font-size: .75rem; margin-top: .18rem; }
      .section-kicker { color: var(--robin-violet); font-size: .72rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .25rem; }
      .active-banner { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: .95rem 1.1rem; border: 1px solid rgba(89,215,232,.22); border-radius: 1rem; background: rgba(18, 45, 57, .3); margin: 1rem 0; }
      .active-title { color: var(--robin-text); font-weight: 750; overflow-wrap: anywhere; }
      .active-meta { color: var(--robin-muted); font-size: .78rem; }
      .pipeline { display: flex; align-items: center; gap: .45rem; margin: .9rem 0 1.3rem; overflow-x: auto; padding: .2rem 0 .55rem; }
      .pipe-step { display: flex; align-items: center; gap: .42rem; white-space: nowrap; color: #71809a; font-size: .75rem; font-weight: 700; }
      .pipe-dot { width: 1.45rem; height: 1.45rem; display: grid; place-items: center; border-radius: 99px; border: 1px solid #3a455d; color: #73809a; font-size: .7rem; }
      .pipe-step.done { color: #b9f1db; }
      .pipe-step.done .pipe-dot { border-color: rgba(98,214,166,.6); background: rgba(55,147,111,.23); color: var(--robin-green); }
      .pipe-step.active { color: #ddd3ff; }
      .pipe-step.active .pipe-dot { border-color: var(--robin-violet); background: rgba(124,58,237,.28); color: #e9ddff; box-shadow: 0 0 13px rgba(167,139,250,.23); }
      .pipe-line { height: 1px; width: 2rem; background: #2b3448; flex: 0 0 auto; }
      .source-card { padding: .82rem .95rem; border: 1px solid var(--robin-line); border-radius: .85rem; background: rgba(15,20,34,.55); margin-bottom: .6rem; }
      .source-index { color: var(--robin-violet); font-size: .72rem; font-weight: 800; }
      .source-title { color: var(--robin-text); font-weight: 700; overflow-wrap: anywhere; }
      .source-url { color: #7e8da7; font-size: .75rem; overflow-wrap: anywhere; }
      .empty-state { text-align: center; padding: 2.1rem 1rem; color: var(--robin-muted); border: 1px dashed var(--robin-line); border-radius: 1rem; }
      .disclaimer { padding: .75rem .9rem; border-left: 3px solid var(--robin-amber); border-radius: .5rem; background: rgba(247,200,115,.08); color: #d8c9a9; font-size: .78rem; line-height: 1.5; }
      .sidebar-title { color: var(--robin-text); font-size: 1.14rem; font-weight: 800; letter-spacing: .08em; }
      .sidebar-caption { color: var(--robin-muted); font-size: .75rem; line-height: 1.45; }
      [data-testid="stMetricValue"] { color: var(--robin-text); }
      [data-testid="stExpander"] { border-color: var(--robin-line); background: rgba(15,20,34,.45); }
      .stButton > button, .stDownloadButton > button { border-radius: .72rem; font-weight: 700; min-height: 2.55rem; }
      .stTextInput input, .stTextArea textarea { border-radius: .7rem; }
      div[data-testid="stTabs"] button { font-weight: 700; }
      @media (max-width: 720px) {
        .block-container { padding: 1rem .75rem 4rem; }
        .robin-topbar { margin-bottom: 1rem; }
        .robin-pill { font-size: .68rem; padding: .38rem .55rem; }
        .hero { padding: 1.2rem 1rem; border-radius: 1rem; }
        .hero-copy { font-size: .93rem; }
        .search-shell { padding: .55rem .6rem 0; }
        .pipeline { margin-bottom: .85rem; }
        .active-banner { display: block; }
        .active-meta { margin-top: .3rem; }
        .metric-card { min-height: 92px; padding: .78rem; }
        .metric-value { font-size: 1.4rem; }
        .source-card { padding: .75rem; }
        [data-testid="stSidebar"] { min-width: min(88vw, 340px); }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .55rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_topbar(model_ready: bool) -> None:
    status = "LLM ready" if model_ready else "Setup required"
    color_class = "" if model_ready else " style='border-color:rgba(247,200,115,.32);color:#f7dca6;background:rgba(247,200,115,.09)'"
    st.markdown(
        f"""
        <div class="robin-topbar">
          <div class="robin-brand">
            <div class="robin-mark">🕵️</div>
            <div><p class="robin-wordmark">ROBIN</p><p class="robin-submark">No-code OSINT workspace</p></div>
          </div>
          <div class="robin-pill"{color_class}><span class="robin-pill-dot"></span>{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pipeline_markup(active: int = 0, complete: bool = False) -> str:
    labels = ["Refine", "Search", "Rank", "Scrape", "Synthesize"]
    pieces = ["<div class='pipeline'>"]
    for i, label in enumerate(labels):
        if complete or i < active:
            state, icon = "done", "✓"
        elif i == active and not complete:
            state, icon = "active", "•"
        else:
            state, icon = "", str(i + 1)
        pieces.append(f"<div class='pipe-step {state}'><span class='pipe-dot'>{icon}</span><span>{label}</span></div>")
        if i < len(labels) - 1:
            pieces.append("<span class='pipe-line'></span>")
    pieces.append("</div>")
    return "".join(pieces)


def _render_pipeline(slot, active: int = 0, complete: bool = False) -> None:
    slot.markdown(_pipeline_markup(active, complete), unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Authorized research, made approachable</div>
          <h1>Turn scattered signals into a clear investigation.</h1>
          <p class="hero-copy">Robin refines a question, searches configured dark-web indexes through Tor, ranks the useful sources, extracts evidence, and turns it into a grounded report.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    feature_cols = st.columns(3)
    features = [
        ("⚡", "One guided workflow", "Move from query to evidence-backed findings without writing a script."),
        ("🧠", "Bring your model", "Use OpenAI, Claude, Gemini, Ollama, OpenRouter, llama.cpp, or any compatible endpoint."),
        ("🗂️", "Keep your casebook", "Save investigations locally, revisit them later, and export Markdown or JSON reports."),
    ]
    for col, (icon, title, copy) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon'>{icon}</div><div class='feature-title'>{title}</div><div class='feature-copy'>{copy}</div></div>",
                unsafe_allow_html=True,
            )


def _render_metric_cards(inv: dict) -> None:
    sources = inv.get("sources", []) or []
    scraped = inv.get("scraped")
    scraped_count = len(scraped) if isinstance(scraped, dict) else (1 if scraped else 0)
    pivots = len(st.session_state.get("pivot_suggestions", []) or [])
    metrics = [
        ("Sources found", inv.get("results_count", len(sources)), "returned by search"),
        ("Sources selected", len(sources), "passed to extraction"),
        ("Pages read", scraped_count, "evidence captured"),
        ("Pivot ideas", pivots, "grounded next leads"),
    ]
    cols = st.columns(4)
    for col, (label, value, hint) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-hint'>{hint}</div></div>",
                unsafe_allow_html=True,
            )


def _safe_http_link(link: Any) -> str:
    """Only expose web URLs as clickable source links."""
    candidate = _clean_text(link)
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return candidate
    except ValueError:
        pass
    return ""


def _short_source_url(link: str) -> str:
    try:
        parsed = urlparse(link)
        host = parsed.hostname or link
        if host.endswith(".onion"):
            return f"{host[:23]}… · onion service"
        return host[:52]
    except Exception:
        return link[:52]


def _render_sources(sources: list) -> None:
    st.caption("Results are links returned by the configured search indexes. Open them only from an authorized, Tor-aware environment.")
    if not sources:
        st.markdown("<div class='empty-state'>No sources were selected for this investigation.</div>", unsafe_allow_html=True)
        return
    for index, item in enumerate(sources, 1):
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title")) or "Untitled source"
        link = _safe_http_link(item.get("link"))
        st.markdown(
            f"""
            <div class="source-card">
              <div class="source-index">SOURCE {index:02d}</div>
              <div class="source-title">{html.escape(title)}</div>
              <div class="source-url">{html.escape(_short_source_url(link) if link else 'Invalid or unavailable URL')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if link:
            st.markdown(f"[Open source ↗]({link})")


def _render_evidence(scraped: Any, source_key: str) -> None:
    if not scraped:
        st.markdown(
            "<div class='empty-state'>Raw page text is not available in this saved record. Enable <b>Save raw evidence</b> for future investigations if local evidence review is required.</div>",
            unsafe_allow_html=True,
        )
        return
    if isinstance(scraped, dict):
        items = [(str(url), _clean_text(text)) for url, text in scraped.items()]
    else:
        items = [("Investigation evidence", _clean_text(scraped))]
    if not items:
        st.markdown("<div class='empty-state'>No extractable page text was returned.</div>", unsafe_allow_html=True)
        return
    labels = [f"{i + 1:02d} · {_short_source_url(url)}" for i, (url, _text) in enumerate(items)]
    selected = st.selectbox("Evidence source", labels, key=f"evidence_select_{source_key}")
    chosen_index = labels.index(selected)
    url, text = items[chosen_index]
    st.caption(url)
    st.code(text or "No text extracted.", language="text")


def _render_downloads(inv: dict) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_slug(inv.get("query", "investigation"))
    widget_key = _safe_slug(inv.get("filename") or f"{slug}-{inv.get('timestamp', '')}")
    report_md = build_markdown_report(inv)
    report_json = build_json_report(inv)
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "↓ Download Markdown",
            data=report_md,
            file_name=f"robin_{slug}_{stamp}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"download_md_{widget_key}",
        )
    with col_b:
        st.download_button(
            "↓ Download JSON",
            data=report_json,
            file_name=f"robin_{slug}_{stamp}.json",
            mime="application/json",
            use_container_width=True,
            key=f"download_json_{widget_key}",
        )


def _followup_history_messages(chat_history: list[dict], max_turns: int = 5) -> list:
    """Convert recent UI turns into LangChain messages."""
    recent = chat_history[-(max_turns * 2):] if chat_history else []
    messages = []
    for turn in recent:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=_clean_text(turn.get("content"))))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=_clean_text(turn.get("content"))))
    return messages


def _render_chat_panel(inv: dict) -> None:
    st.divider()
    st.markdown("<div class='section-kicker'>Continue the case</div>", unsafe_allow_html=True)
    st.subheader("Follow-up chat", anchor=None)
    st.caption("Ask about this investigation. Answers are grounded in its sources, extracted text, summary, and the conversation so far.")

    pivots = st.session_state.get("pivot_suggestions", []) or []
    if pivots:
        st.markdown("**Suggested pivots**")
        st.caption("Each pivot starts a fresh investigation with the same settings.")
        for index, pivot in enumerate(pivots):
            if st.button(f"🔎  {pivot}", key=f"pivot_{index}_{_safe_slug(inv.get('query', 'case'))}", use_container_width=True):
                st.session_state["pivot_query"] = pivot
                st.rerun()

    history = st.session_state.get("chat_history", []) or []
    for turn in history:
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(turn.get("content", ""))

    if history and st.button("Clear conversation", key="clear_chat", use_container_width=False):
        st.session_state["chat_history"] = []
        st.rerun()

    followup = st.chat_input("Ask a follow-up about this investigation", key="followup_input")
    if not followup:
        return

    with st.chat_message("user"):
        st.markdown(followup)
    with st.chat_message("assistant"):
        answer_slot = st.empty()
        accumulator = {"text": ""}

        def emit(chunk: str) -> None:
            accumulator["text"] += chunk
            answer_slot.markdown(accumulator["text"])

        try:
            followup_llm = get_llm(inv.get("model", ""))
            followup_llm.callbacks = [BufferedStreamingHandler(ui_callback=emit)]
            answer = answer_followup(
                followup_llm,
                followup,
                build_followup_context(
                    inv.get("query", ""),
                    inv.get("refined", ""),
                    inv.get("sources", []),
                    inv.get("scraped"),
                    inv.get("summary", ""),
                ),
                history=_followup_history_messages(history),
                preset=inv.get("preset", "threat_intel"),
                custom_instructions=inv.get("custom_instructions", ""),
            )
            # Reasoning models may return only at the end and emit no answer
            # tokens through on_llm_new_token.
            if not accumulator["text"].strip() and answer:
                accumulator["text"] = answer
                answer_slot.markdown(answer)
        except Exception as exc:
            accumulator["text"] = f"Unable to answer this follow-up: {exc}"
            answer_slot.error(accumulator["text"])

    st.session_state.setdefault("chat_history", [])
    st.session_state["chat_history"].append({"role": "user", "content": followup})
    st.session_state["chat_history"].append({"role": "assistant", "content": accumulator["text"]})


def _render_investigation(inv: dict, pipeline_slot=None) -> None:
    query = _clean_text(inv.get("query"))
    timestamp = _clean_text(inv.get("timestamp"))
    time_label = timestamp[:16].replace("T", " · ") if timestamp else "current session"
    st.markdown(
        f"""
        <div class="active-banner">
          <div><div class="active-title">Active investigation · {html.escape(query)}</div><div class="active-meta">{html.escape(time_label)} · {html.escape(_clean_text(inv.get('preset_label') or inv.get('preset', 'Threat intelligence')))}</div></div>
          <div class="robin-pill"><span class="robin-pill-dot"></span>Workspace active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pipeline_slot is None:
        st.markdown(_pipeline_markup(complete=True), unsafe_allow_html=True)
    _render_metric_cards(inv)
    st.write("")

    tab_findings, tab_sources, tab_evidence, tab_notes = st.tabs(
        ["🔎 Findings", "🔗 Sources", "📄 Evidence", "📋 Notes"]
    )
    with tab_findings:
        st.markdown("<div class='section-kicker'>Grounded synthesis</div>", unsafe_allow_html=True)
        st.subheader("Investigation findings", anchor=None)
        summary = _clean_text(inv.get("summary"))
        if summary:
            st.markdown(summary)
        else:
            st.markdown("<div class='empty-state'>No findings were generated.</div>", unsafe_allow_html=True)
        st.write("")
        _render_downloads(inv)
    with tab_sources:
        st.markdown("<div class='section-kicker'>Collected references</div>", unsafe_allow_html=True)
        st.subheader(f"Selected sources · {len(inv.get('sources', []) or [])}", anchor=None)
        _render_sources(inv.get("sources", []) or [])
    with tab_evidence:
        st.markdown("<div class='section-kicker'>Reviewable text</div>", unsafe_allow_html=True)
        st.subheader("Extracted evidence", anchor=None)
        evidence_key = _safe_slug(f"{query}-{timestamp}")
        _render_evidence(inv.get("scraped"), evidence_key)
    with tab_notes:
        st.markdown("<div class='section-kicker'>How this case was built</div>", unsafe_allow_html=True)
        st.subheader("Investigation notes", anchor=None)
        note_cols = st.columns(2)
        with note_cols[0]:
            st.markdown(f"**Original query**\n\n`{query}`")
            st.markdown(f"**Refined query**\n\n`{_clean_text(inv.get('refined')) or 'Not available'}`")
            st.markdown(f"**Model**\n\n`{_clean_text(inv.get('model')) or 'Not recorded'}`")
        with note_cols[1]:
            st.markdown(f"**Research domain**\n\n{inv.get('preset_label') or inv.get('preset', 'Threat intelligence')}")
            st.markdown(f"**Search results**\n\n{inv.get('results_count', 0)}")
            st.markdown(f"**Raw evidence archived**\n\n{'Yes' if inv.get('raw_saved') else 'No — memory only'}")
        if _clean_text(inv.get("custom_instructions")):
            st.markdown("**Custom focus**")
            st.info(inv["custom_instructions"])

    _render_chat_panel(inv)


# ---------------------------------------------------------------------------
# Sidebar: all configuration is available without editing .env
# ---------------------------------------------------------------------------

_seed_provider_state()
_sync_runtime_config()

st.sidebar.markdown("<div class='sidebar-title'>ROBIN</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-caption'>AI-powered dark-web OSINT, with a guided operator workflow.</div>", unsafe_allow_html=True)
st.sidebar.divider()

with st.sidebar.expander("⚙️ AI provider setup", expanded=not any(_env_is_set(getattr(_robin_cfg, name, "")) for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL", "LLAMA_CPP_BASE_URL", "CUSTOM_API_BASE_URL"))):
    st.caption("Keys stay in this Streamlit session. They are not written to the investigation archive.")
    st.text_input("OpenAI API key", type="password", key="provider_openai_key", placeholder="sk-…")
    st.text_input("Anthropic API key", type="password", key="provider_anthropic_key", placeholder="sk-ant-…")
    st.text_input("Google AI API key", type="password", key="provider_google_key", placeholder="AIza…")
    st.text_input("OpenRouter API key", type="password", key="provider_openrouter_key", placeholder="sk-or-…")
    st.text_input(
        "OpenRouter base URL",
        key="provider_openrouter_url",
        placeholder="https://openrouter.ai/api/v1",
        help="Usually the default above. Change only for a compatible gateway.",
    )
    st.markdown("**Local or compatible endpoints**")
    st.text_input("Ollama URL", key="provider_ollama_url", placeholder="http://127.0.0.1:11434")
    st.text_input("llama.cpp URL", key="provider_llama_url", placeholder="http://127.0.0.1:8080")
    st.text_input("Custom API base URL", key="custom_api_url", placeholder="https://api.groq.com/openai/v1")
    st.text_input("Custom API key", type="password", key="custom_api_key", placeholder="optional")
    st.text_input("Custom model name", key="custom_api_model", placeholder="auto-discover if blank")
    _sync_runtime_config()

# Model discovery happens after the provider fields are synced, so a key entered
# in the UI immediately unlocks the corresponding models on the next rerun.
model_options = get_model_choices()
model_display_names = get_model_display_names(model_options)
if model_options:
    configured_model = st.session_state.get("model_select")
    if configured_model not in model_options:
        configured_model = next((m for m in model_options if m.lower() == "gpt-4.1"), model_options[0])
        st.session_state["model_select"] = configured_model
    model = st.sidebar.selectbox(
        "Model",
        model_options,
        format_func=lambda key: model_display_names.get(key, key),
        index=model_options.index(configured_model),
        key="model_select",
        help="Cloud models appear when their key is configured. Local models are discovered automatically.",
    )
    if any(m not in {"gpt-4.1", "gpt-5.2", "gpt-5.1", "gpt-5-mini", "gpt-5-nano", "claude-sonnet-4-5", "claude-sonnet-4-0", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"} for m in model_options):
        st.sidebar.caption("Local and custom models are detected automatically.")
else:
    model = None
    st.sidebar.warning("Add a provider key or local endpoint above to unlock investigations.")

st.sidebar.markdown("**Provider status**")
for provider_name, config_name, is_cloud in (
    ("OpenAI", "OPENAI_API_KEY", True),
    ("Anthropic", "ANTHROPIC_API_KEY", True),
    ("Google Gemini", "GOOGLE_API_KEY", True),
    ("OpenRouter", "OPENROUTER_API_KEY", True),
    ("Ollama", "OLLAMA_BASE_URL", False),
    ("llama.cpp", "LLAMA_CPP_BASE_URL", False),
    ("Custom API", "CUSTOM_API_BASE_URL", False),
):
    st.sidebar.caption(_provider_badge(provider_name, st.session_state.get(_PROVIDER_FIELDS[config_name][0]), is_cloud))

with st.sidebar.expander("🎛️ Investigation controls", expanded=True):
    threads = st.slider("Concurrent workers", 1, 16, 4, key="thread_slider", help="Parallel search and page-fetch workers.")
    max_results = st.slider("Results sent to ranking", 10, 100, 50, key="max_results_slider", help="Caps raw results before the LLM ranking step.")
    max_scrape = st.slider("Pages sent to extraction", 3, 20, 10, key="max_scrape_slider", help="Caps ranked results that Robin fetches for evidence.")
    save_raw = st.checkbox("Save raw evidence locally", key="save_raw", help="Raw extracted text can contain sensitive material. It is never saved unless this is checked.")

with st.sidebar.expander("🧩 Prompt settings", expanded=False):
    selected_preset_label = st.selectbox("Research domain", list(PRESET_OPTIONS), key="preset_select")
    selected_preset = PRESET_OPTIONS[selected_preset_label]
    st.text_area("Active system prompt", value=PRESET_PROMPTS[selected_preset].strip(), height=190, disabled=True, key="system_prompt_display")
    custom_instructions = st.text_area(
        "Custom focus (optional)",
        placeholder=PRESET_PLACEHOLDERS[selected_preset],
        height=95,
        key="custom_instructions",
        help="Adds a narrow focus while keeping the summary grounded in returned data.",
    )

with st.sidebar.expander("🩺 Health checks", expanded=False):
    if st.button("Test LLM connection", use_container_width=True, disabled=not model, key="health_llm_button"):
        with st.spinner("Testing selected model…"):
            st.session_state["llm_health"] = check_llm_health(model)
    llm_health = st.session_state.get("llm_health")
    if llm_health:
        if llm_health.get("status") == "up":
            st.success(f"{llm_health.get('provider', 'LLM')} connected · {llm_health.get('latency_ms')} ms")
        else:
            st.error(f"{llm_health.get('provider', 'LLM')} unavailable\n\n{llm_health.get('error', 'Unknown error')}")

    if st.button("Test Tor + search indexes", use_container_width=True, key="health_search_button"):
        with st.spinner("Checking Tor proxy…"):
            tor_result = check_tor_proxy()
        st.session_state["tor_health"] = tor_result
        if tor_result.get("status") == "up":
            with st.spinner(f"Pinging {len(SEARCH_ENGINES)} configured indexes…"):
                st.session_state["engine_health"] = check_search_engines()
    tor_health = st.session_state.get("tor_health")
    if tor_health:
        if tor_health.get("status") == "up":
            st.success(f"Tor proxy connected · {tor_health.get('latency_ms')} ms")
        else:
            st.error(f"Tor proxy unavailable\n\n{tor_health.get('error', 'Start Tor before live searches.')}")
    engine_health = st.session_state.get("engine_health")
    if engine_health:
        up = sum(1 for result in engine_health if result.get("status") == "up")
        st.info(f"{up}/{len(engine_health)} search indexes reachable")
        for result in engine_health:
            icon = "🟢" if result.get("status") == "up" else "🔴"
            suffix = f"{result.get('latency_ms')} ms" if result.get("latency_ms") else result.get("error", "unavailable")
            st.caption(f"{icon} {result.get('name', 'index')} · {suffix}")

with st.sidebar.expander("📂 Past investigations", expanded=False):
    saved_investigations = load_investigations()
    if saved_investigations:
        labels = [
            f"{_clean_text(item.get('timestamp'))[:16].replace('T', ' ')} · {_clean_text(item.get('query'))[:42]} · #{index + 1}"
            for index, item in enumerate(saved_investigations)
        ]
        selected_label = st.selectbox("Load a saved case", ["(none)"] + labels, key="saved_investigation_select")
        if selected_label != "(none)":
            selected_index = labels.index(selected_label)
            if st.button("Load investigation", use_container_width=True, key="load_investigation_button"):
                st.session_state["active_investigation"] = _investigation_from_saved(saved_investigations[selected_index])
                st.session_state["chat_history"] = []
                st.session_state["pivot_suggestions"] = []
                st.rerun()
    else:
        st.caption("No saved investigations yet.")

if st.sidebar.button("＋ New investigation", use_container_width=True, key="new_investigation_button"):
    for state_key in ("active_investigation", "chat_history", "pivot_suggestions", "pivot_query", "suggested_query"):
        st.session_state.pop(state_key, None)
    st.session_state["query_input"] = ""
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    "<div class='disclaimer'><b>Lawful-use reminder</b><br>Only investigate systems, identities, and data you are authorized to examine. Respect local law, provider terms, privacy, and institutional policy.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------------

_render_topbar(bool(model_options))
acknowledged = st.checkbox(
    "I confirm this investigation is authorized and will be used for lawful defensive or research purposes.",
    key="lawful_acknowledged",
)

hero_slot = st.empty()
if not st.session_state.get("active_investigation"):
    with hero_slot.container():
        _render_hero()

st.markdown("<div class='section-kicker'>Start with a question</div>", unsafe_allow_html=True)
st.markdown("### Run an investigation")

# Quick-query buttons render below the form. Apply their value before the form
# is instantiated on the rerun, otherwise Streamlit correctly rejects attempts
# to mutate an already-created widget key.
if "suggested_query" in st.session_state:
    st.session_state["query_input"] = st.session_state.pop("suggested_query")

query_is_ready = bool(model_options) and bool(acknowledged)
with st.container(border=True):
    with st.form("search_form", clear_on_submit=False):
        input_col, button_col = st.columns([5, 1])
        query = input_col.text_input(
            "Search query",
            placeholder="e.g. ransomware group data leak",
            label_visibility="collapsed",
            key="query_input",
        )
        run_button = button_col.form_submit_button(
            "Run search",
            type="primary",
            use_container_width=True,
            disabled=not query_is_ready,
        )

if not model_options:
    st.info("Set up at least one provider in the sidebar. You can enter credentials or a local endpoint there—no .env editing is required.")
elif not acknowledged:
    st.caption("Confirm authorized use above to enable live searches.")

st.caption(f"Live searches use {len(SEARCH_ENGINES)} configured indexes through the Tor SOCKS proxy. Use Health checks in the sidebar before the first run.")

quick_slot = st.empty()
if not st.session_state.get("active_investigation"):
    with quick_slot.container():
        st.markdown("<div class='quick-label'>Try a neutral starting point</div>", unsafe_allow_html=True)
        quick_cols = st.columns(3)
        quick_queries = [
            ("Threat actor alias", "threat actor alias"),
            ("Ransomware reporting", "ransomware group leak"),
            ("Corporate exposure", "company data exposure"),
        ]
        for col, (label, value) in zip(quick_cols, quick_queries):
            with col:
                if st.button(label, key=f"quick_{_safe_slug(value)}", use_container_width=True):
                    st.session_state["suggested_query"] = value
                    st.rerun()

# A pivot is intentionally consumed after the form is rendered, so it can use
# the same visible query bar and launch from a follow-up button.
pivot_query = st.session_state.pop("pivot_query", None)
active_query = _clean_text(pivot_query or query)
do_run = bool(active_query) and bool(model) and (bool(run_button) or pivot_query is not None)

if do_run:
    # Start a fresh case while retaining the operator's provider/settings state.
    st.session_state.pop("active_investigation", None)
    st.session_state["chat_history"] = []
    st.session_state["pivot_suggestions"] = []
    pipeline_slot = st.empty()
    _render_pipeline(pipeline_slot, active=0)

    def fail_pipeline(stage: str, exc: Exception) -> None:
        message = _clean_text(exc) or exc.__class__.__name__
        st.error(f"Could not {stage}.\n\n**Details:** {message}\n\nCheck the provider settings and the Tor health check, then try again.")

    try:
        with st.spinner("Loading selected model…"):
            llm = get_llm(model)
    except Exception as exc:
        fail_pipeline("load the selected model", exc)
        llm = None

    if llm is not None:
        try:
            _render_pipeline(pipeline_slot, active=0)
            with st.spinner("Refining your query…"):
                refined = refine_query(llm, active_query)
            refined = _clean_text(refined) or active_query

            _render_pipeline(pipeline_slot, active=1)
            with st.spinner("Searching configured indexes through Tor…"):
                raw_results = cached_search_results(refined, threads)
            raw_results = raw_results if isinstance(raw_results, list) else []
            raw_results = raw_results[:max_results]

            _render_pipeline(pipeline_slot, active=2)
            with st.spinner("Ranking the most relevant sources…"):
                filtered = filter_results(llm, refined, raw_results)
            filtered = filtered if isinstance(filtered, list) else []
            filtered = filtered[:max_scrape]

            _render_pipeline(pipeline_slot, active=3)
            with st.spinner("Extracting readable evidence…"):
                scraped = cached_scrape_multiple(filtered, threads)
            scraped = scraped if isinstance(scraped, dict) else {}

            _render_pipeline(pipeline_slot, active=4)
            with st.spinner("Writing a grounded findings report…"):
                stream_buffer = {"text": ""}
                summary_slot = st.empty()

                def emit_summary(chunk: str) -> None:
                    stream_buffer["text"] += chunk
                    summary_slot.markdown(stream_buffer["text"])

                llm.callbacks = [BufferedStreamingHandler(ui_callback=emit_summary)]
                summary = generate_summary(
                    llm,
                    active_query,
                    scraped,
                    preset=selected_preset,
                    custom_instructions=custom_instructions,
                )
                if not stream_buffer["text"].strip() and summary:
                    stream_buffer["text"] = _clean_text(summary)
                    summary_slot.markdown(stream_buffer["text"])
                summary = stream_buffer["text"] or _clean_text(summary)
                # The live stream has done its job; the organized Findings tab
                # below is the single persistent rendering of the report.
                summary_slot.empty()

            inv = {
                "query": active_query,
                "refined": refined,
                "model": model,
                "preset": selected_preset,
                "preset_label": selected_preset_label,
                "custom_instructions": custom_instructions,
                "sources": filtered,
                "scraped": scraped,
                "summary": summary,
                "results_count": len(raw_results),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "raw_saved": bool(save_raw and scraped),
            }

            try:
                filename = save_investigation(
                    query=active_query,
                    refined_query=refined,
                    model=model,
                    preset_label=selected_preset_label,
                    sources=filtered,
                    summary=summary,
                    scraped=scraped,
                    custom_instructions=custom_instructions,
                    results_count=len(raw_results),
                    save_raw=save_raw,
                )
                inv["filename"] = filename
            except OSError as exc:
                inv["filename"] = ""
                st.warning(f"Findings are ready, but the local archive could not be written: {exc}")

            # Pivot suggestions are deliberately best-effort. They never turn a
            # completed investigation into a failed one.
            with st.spinner("Finding grounded pivot ideas…"):
                try:
                    st.session_state["pivot_suggestions"] = suggest_pivots(
                        get_llm(model), active_query, scraped, preset=selected_preset
                    )
                except Exception:
                    st.session_state["pivot_suggestions"] = []

            # Replace the onboarding surface with the completed case on this
            # same run, rather than making the operator scroll past it.
            hero_slot.empty()
            quick_slot.empty()
            st.session_state["active_investigation"] = inv
            st.session_state["chat_history"] = []
            _render_pipeline(pipeline_slot, complete=True)
            st.success(
                "Investigation complete" + (f" · saved as `{inv['filename']}`" if inv.get("filename") else "")
            )
            _render_investigation(inv, pipeline_slot=pipeline_slot)
        except Exception as exc:
            fail_pipeline("complete the investigation", exc)

elif st.session_state.get("active_investigation"):
    _render_investigation(st.session_state["active_investigation"])
