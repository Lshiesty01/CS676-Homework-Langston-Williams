"""
CS676 Project 1 — Credibility-scored research chatbot.

A Streamlit chat app that answers questions using Claude, shows the sources it
used, and displays a credibility score beside each one.

Run it with:   streamlit run main.py

You should not need to change much in this file. The part you are graded on
lives in credibility.py.

PART 3 UPDATE — Stages 3 and 4 wired into the UI. Every source now shows,
alongside score_url()'s graded score:
  - a bootstrapped 90% confidence interval (Stage 4, score_with_uncertainty())
  - a Platt-calibrated P(credible) in the "Why this score?" expander
    (Stage 3, calibrated_probability())
Both are fail-open: if learned_weights.json / calibration.json /
bootstrap_weights.json are missing, these calls degrade to the point
estimate with no interval and no correction — see credibility.py's own
docstrings for that fallback behaviour. The sidebar's "Model status" panel
shows which of the three fitted models are actually loaded, so it's obvious
at a glance during a live demo whether Stage 2/3/4 are active.
"""

import os
from typing import Any, Dict, List, Tuple

import anthropic
import streamlit as st
from dotenv import load_dotenv

from credibility import (
    _CALIBRATION,
    _BOOTSTRAP_MODELS,
    _LEARNED_WEIGHTS,
    calibrated_probability,
    score_band,
    score_url,
    score_with_uncertainty,
)

load_dotenv()

# Claude Opus 5 is the most capable model. Swap to "claude-sonnet-5" or
# "claude-haiku-4-5" if you want to reduce cost while developing — note which
# one your submitted numbers used.
CHAT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# Web search can return 20+ results per turn, and every displayed source is
# scored — one API call each when the LLM layer is on. Cap what we show.
MAX_SOURCES = 6

SYSTEM_PROMPT = """You are a research assistant for a graduate data science course.

Answer using the sources available to you and cite them. Be direct and concise.
When the evidence is thin or the sources disagree, say so plainly rather than
smoothing it over. Never invent a source or a URL."""


# -----------------------------------------------------------------------------
# Optional Langfuse tracing
# -----------------------------------------------------------------------------
# Tracing is a nice-to-have, not a requirement. If the Langfuse keys are absent
# we fall back to a no-op decorator so the app still runs on a fresh clone.
# This is why you can start working before configuring anything but the API key.
try:
    from langfuse import get_client, observe

    _langfuse = get_client()
    TRACING_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
except Exception:
    TRACING_ENABLED = False
    _langfuse = None

    def observe(*_args, **_kwargs):  # type: ignore[misc]
        """No-op stand-in for @observe when Langfuse is not configured."""
        def decorator(fn):
            return fn
        return decorator


def search_serpapi(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Search Google via SerpAPI and return the organic results.

    This is optional context on top of Claude's own web search — it gives you a
    second, independently-retrieved set of URLs to score, which is useful when
    comparing how your scorer treats different kinds of source.
    """
    from serpapi import GoogleSearch

    search = GoogleSearch({"q": query, "api_key": api_key})
    return search.get_dict().get("organic_results", [])


# ---------------------------------------------------------------------------
# ⚠️  NEEDS YOUR OWN API KEY — AND SHIPPED UNVERIFIED
# ---------------------------------------------------------------------------
# REQUIRES A KEY. The chat does not work without ANTHROPIC_API_KEY in `.env`;
# the sidebar shows a red mark when it is missing. Get one at
# https://console.anthropic.com/. Calls are billed to you. The URL scorer in the
# sidebar, the tests, and evaluate.py all work without a key.
#
# VERIFIED LIVE — after a real bug was found here. The first live run returned
# ZERO sources, because this function originally read citations off the text
# blocks. `web_search_20260209` does not put them there: it returns them in
# `web_search_tool_result` blocks, and `block.citations` is None. The code below
# now reads both, and a live run yields six sources including the actual
# arXiv link for "Attention Is All You Need".
#
# The lesson is worth more than the fix: an API that returns an empty list where
# you expected data fails silently. Nothing crashed, no error was logged, the
# app just quietly showed no sources at all.
# ---------------------------------------------------------------------------
@observe()
def ask_claude(messages: List[Dict[str, str]], user: str, email: str, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Send the conversation to Claude and return the answer plus its citations.

    Where the sources come from: the `web_search_20260209` tool returns them in
    `web_search_tool_result` blocks, NOT as citation metadata on the text blocks.
    That is worth knowing — the obvious implementation reads `block.citations`,
    finds it empty, and silently shows no sources at all, which is exactly the
    bug this function was shipped with until it was run against the live API.

    :return: (answer_text, [{"url": ..., "title": ...}, ...])
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
    )

    # Claude can decline a request. Check before reading content, which is empty
    # or partial on a refusal.
    if response.stop_reason == "refusal":
        return ("I can't help with that request.", [])

    answer = ""
    citations: List[Dict[str, str]] = []
    seen: set = set()

    for block in response.content:
        if block.type == "text":
            answer += block.text
            # Some configurations attach citations directly to text blocks.
            # web_search_20260209 does NOT — see the branch below — but keep this
            # path so the app still works if that changes or you enable document
            # citations.
            for citation in getattr(block, "citations", None) or []:
                url = getattr(citation, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"url": url, "title": getattr(citation, "title", "") or url})

        elif block.type == "web_search_tool_result":
            # This is where the sources actually are. Each successful result block
            # holds a list of `web_search_result` items with .url and .title.
            # On failure `.content` is a single error object rather than a list,
            # so check the type before iterating.
            results = getattr(block, "content", None)
            if not isinstance(results, list):
                continue
            for item in results:
                url = getattr(item, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"url": url, "title": getattr(item, "title", "") or url})

    # A single turn can return twenty-odd results across several searches, and the
    # app scores every one of them — which with the LLM layer on is one API call
    # each. Cap it: the first few are the ones the model actually leaned on.
    citations = citations[:MAX_SOURCES]

    if TRACING_ENABLED and _langfuse is not None:
        _langfuse.update_current_trace(
            input=messages[-1]["content"] if messages else "",
            output=answer,
            user_id=user,
            session_id=session_id,
            tags=["cs676", "project-1"],
            metadata={"email": email, "citations": len(citations)},
        )

    return answer, citations


def render_source(index: int, title: str, url: str, snippet: str = "") -> None:
    """
    Render one source as a labelled row with a coloured credibility chip.

    The chip is the visible payoff of your work in credibility.py — a reader
    should be able to judge a source at a glance without reading the URL.

    PART 3: also surfaces Stage 4's bootstrapped confidence interval next to
    the chip, and Stage 3's calibrated probability inside the expander.
    score_url() is called once here and cached; the two calls below reuse
    that cache entry (see credibility.py's _CACHE), so this costs no extra
    network or LLM calls beyond what the original single-score version did.
    """
    result = score_url(url)
    label, colour = score_band(result["score"])
    interval = score_with_uncertainty(url)
    calibrated = calibrated_probability(url)

    ci_badge = ""
    if interval["n_bootstrap"] > 0:
        ci_badge = f"  &nbsp; *(90% CI: {interval['low']:.2f}–{interval['high']:.2f})*"

    st.markdown(
        f"**{index}. [{title}]({url})** &nbsp; "
        f":{colour}[**● {result['score']:.2f} {label}**]{ci_badge}"
    )
    if snippet:
        st.caption(snippet)
    with st.expander("Why this score?"):
        st.write(result["explanation"])
        if interval["n_bootstrap"] > 0:
            st.caption(
                f"**Uncertainty (Stage 4):** 90% confidence interval "
                f"{interval['low']:.2f}–{interval['high']:.2f}, from "
                f"{interval['n_bootstrap']} bootstrap-resampled models. "
                "Narrower means this source's features resemble the training "
                "data closely; wider means the model is extrapolating."
            )
        else:
            st.caption(
                "Uncertainty (Stage 4) unavailable — run `bootstrap_uncertainty.py` "
                "to enable confidence intervals."
            )
        st.caption(
            f"**Calibrated P(credible) (Stage 3):** {calibrated:.2f} — "
            "Platt-scaled against a binary credible/not-credible split. This "
            "can disagree with the graded score above, especially mid-range "
            "sources; see the technique report, Section 3.4/5.5, for why "
            "score_url() itself is left uncalibrated."
        )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CS676 — Credibility Chatbot", page_icon="🔍")
st.title("🔍 Credibility-Scored Research Assistant")

with st.sidebar:
    st.subheader("Session")
    user = st.text_input("Name", value="student")
    email = st.text_input("Email", value="student@pace.edu")
    session_id = f"{user}_{email}"

    st.divider()
    use_serpapi = st.checkbox("Also search with SerpAPI", value=False)

    st.divider()
    st.caption("**Status**")
    st.caption(("✅" if os.getenv("ANTHROPIC_API_KEY") else "❌") + " Anthropic API key")
    st.caption(("✅" if os.getenv("SERPAPI_API_KEY") else "⬜") + " SerpAPI key (optional)")
    st.caption(("✅" if TRACING_ENABLED else "⬜") + " Langfuse tracing (optional)")

    # PART 3: which fitted models credibility.py actually loaded at import
    # time. All three are optional and fail open (see credibility.py), so
    # this is purely informational — it tells a grader mid-demo whether
    # Stage 2/3/4 are live without them having to read a JSON file.
    st.divider()
    st.caption("**Model status**")
    st.caption(("✅" if _LEARNED_WEIGHTS else "⬜")
               + " Stage 2 — learned weights (train_weights.py)")
    st.caption(("✅" if _CALIBRATION else "⬜")
               + " Stage 3 — calibration (calibration.py)")
    st.caption(("✅" if _BOOTSTRAP_MODELS else "⬜")
               + f" Stage 4 — bootstrap uncertainty ({len(_BOOTSTRAP_MODELS)} models)")

    st.divider()
    st.caption("Score any URL directly:")
    probe = st.text_input("URL", placeholder="https://arxiv.org/abs/1706.03762")
    if probe:
        probe_result = score_url(probe)
        probe_label, probe_colour = score_band(probe_result["score"])
        probe_interval = score_with_uncertainty(probe)
        probe_calibrated = calibrated_probability(probe)

        st.markdown(f":{probe_colour}[**{probe_result['score']:.2f} — {probe_label}**]")
        if probe_interval["n_bootstrap"] > 0:
            st.caption(
                f"90% CI: {probe_interval['low']:.2f}–{probe_interval['high']:.2f} "
                f"(Stage 4, {probe_interval['n_bootstrap']} models)"
            )
        st.caption(f"Calibrated P(credible): {probe_calibrated:.2f} (Stage 3)")
        st.caption(probe_result["explanation"])

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning("No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env` and add your key. "
               "The URL scorer in the sidebar still works without one.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for i, source in enumerate(message.get("sources", []), 1):
            render_source(i, source["title"], source["url"], source.get("snippet", ""))

if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the request separately from the stored history. Search context is
    # useful for this turn only — writing it back into session_state would
    # re-send it on every later turn and inflate the conversation.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    serp_sources: List[Dict[str, str]] = []

    if use_serpapi and os.getenv("SERPAPI_API_KEY"):
        try:
            results = search_serpapi(prompt, os.getenv("SERPAPI_API_KEY"))[:5]
            if results:
                context = "\n\nSearch results for reference:\n"
                for r in results:
                    title = r.get("title", "Untitled")
                    link = r.get("link", "")
                    snippet = r.get("snippet", "")
                    serp_sources.append({"title": title, "url": link, "snippet": snippet})
                    context += f"- {title} ({link})\n  {snippet}\n"
                api_messages[-1] = {"role": "user", "content": prompt + context}
        except Exception as e:
            st.warning(f"SerpAPI search failed: {e}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, citations = ask_claude(api_messages, user, email, session_id)
            except Exception as e:
                answer, citations = f"Error: {e}", []

        st.markdown(answer)

        # Merge Claude's own citations with any SerpAPI results, dropping dupes.
        sources: List[Dict[str, str]] = []
        seen_urls: set = set()
        for source in citations + serp_sources:
            if source["url"] and source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                sources.append(source)

        if sources:
            st.divider()
            st.caption(f"**{len(sources)} source(s), scored by `credibility.score_url`**")
            for i, source in enumerate(sources, 1):
                render_source(i, source["title"], source["url"], source.get("snippet", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
