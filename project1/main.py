"""
CS676 Project 1 — Credibility-scored research chatbot.

A Streamlit chat app that answers questions using Claude, shows the sources it
used, and displays a credibility score beside each one.

Run it with:   uv run streamlit run main.py

There is also a Gradio version of this same app in app.py. Both share their
model call and citation handling via chat_backend.py.

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
from typing import Any, Dict, List

import streamlit as st

from chat_backend import (
    MAX_SOURCES,
    TRACING_ENABLED,
    ask_claude,
    merge_sources,
    search_serpapi,
)
from credibility import (
    _CALIBRATION,
    _BOOTSTRAP_MODELS,
    _LEARNED_WEIGHTS,
    calibrated_probability,
    score_band,
    score_url,
    score_with_uncertainty,
)


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
