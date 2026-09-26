"""
app.py — the Gradio front end for the credibility-scored research chatbot.

Same app as main.py, different UI framework. Both import their model call,
search helper, and citation extraction from chat_backend.py, so there is exactly
one copy of the logic and a fix in one place reaches both.

    uv run python app.py          # http://localhost:7860

This is the file a Hugging Face **Gradio** Space runs, which is why it is called
`app.py` — Spaces looks for that name by default. Gradio Spaces run free on
ZeroGPU, which is the reason this front end exists alongside the Streamlit one:
a Streamlit app on Spaces needs the Docker SDK, and creating a Docker Space
requires a paid plan. See the bonus section of README.md.

You should not need to change much in this file. The part you are graded on
lives in credibility.py.

PART 3 UPDATE — Stages 3 and 4 wired into this front end too, mirroring
main.py's render_source(): every source now shows a bootstrapped 90%
confidence interval (Stage 4) next to its chip, and a Platt-calibrated
P(credible) (Stage 3) inside the "Why this score?" details block. Both are
fail-open, same as in main.py — if learned_weights.json / calibration.json /
bootstrap_weights.json are missing, these calls degrade to the point estimate
with no interval and no correction.
"""

import os
from typing import Any, Dict, List, Tuple

import gradio as gr

from chat_backend import MAX_SOURCES, ask_claude, merge_sources, search_serpapi
from credibility import (
    _CALIBRATION,
    _BOOTSTRAP_MODELS,
    _LEARNED_WEIGHTS,
    calibrated_probability,
    score_band,
    score_url,
    score_with_uncertainty,
)

# Gradio has no `st.markdown(":green[...]")` equivalent, so the chips are plain
# HTML. These are the three bands from credibility.score_band(), mapped to
# colours that stay legible on both the light and dark Gradio themes.
BAND_COLOURS = {
    "green": "#1a7f37",
    "orange": "#bf8700",
    "red": "#cf222e",
}


def _chip(score: float, ci_text: str = "") -> str:
    """Render one credibility score as a coloured inline badge.

    PART 3: takes an optional pre-formatted CI string (Stage 4) to show right
    next to the chip, mirroring main.py's ci_badge.
    """
    label, colour = score_band(score)
    badge = (
        f'<span style="background:{BAND_COLOURS.get(colour, "#57606a")};color:#fff;'
        f'padding:2px 8px;border-radius:10px;font-size:0.85em;font-weight:600;'
        f'white-space:nowrap">● {score:.2f} {label}</span>'
    )
    if ci_text:
        badge += (
            f'<span style="color:#57606a;font-size:0.85em;margin-left:6px">'
            f"{ci_text}</span>"
        )
    return badge


def _uncertainty_html(interval: Dict[str, Any]) -> str:
    """Stage 4: the bootstrapped confidence interval, or a note that it's not
    available yet — same fail-open message main.py shows."""
    if interval["n_bootstrap"] > 0:
        return (
            "<p style='margin:.2em 0;color:#57606a;font-size:.9em'>"
            "<b>Uncertainty (Stage 4):</b> 90% confidence interval "
            f"{interval['low']:.2f}–{interval['high']:.2f}, from "
            f"{interval['n_bootstrap']} bootstrap-resampled models. Narrower means "
            "this source's features resemble the training data closely; wider "
            "means the model is extrapolating.</p>"
        )
    return (
        "<p style='margin:.2em 0;color:#57606a;font-size:.9em'>"
        "Uncertainty (Stage 4) unavailable — run <code>bootstrap_uncertainty.py</code> "
        "to enable confidence intervals.</p>"
    )


def _calibration_html(calibrated: float) -> str:
    """Stage 3: the Platt-calibrated probability, with the same caveat
    main.py gives about it disagreeing with the graded score."""
    return (
        "<p style='margin:.2em 0;color:#57606a;font-size:.9em'>"
        f"<b>Calibrated P(credible) (Stage 3):</b> {calibrated:.2f} — Platt-scaled "
        "against a binary credible/not-credible split. This can disagree with the "
        "graded score above, especially mid-range sources; see the technique "
        "report, Section 3.4/5.5, for why score_url() itself is left "
        "uncalibrated.</p>"
    )


def render_sources(sources: List[Dict[str, str]]) -> str:
    """
    Turn the scored sources into one HTML block for the panel below the chat.

    This is the visible payoff of your work in credibility.py — a reader should
    be able to judge a source at a glance without reading the URL. Each score
    comes with its explanation, because a number nobody can interrogate is not
    much better than no number at all.

    PART 3: also surfaces Stage 4's bootstrapped confidence interval next to
    the chip, and Stage 3's calibrated probability inside the details block —
    the same information main.py shows, adapted to Gradio's HTML rendering.
    score_url() is called once here and cached; the two extra calls below
    reuse that cache entry (see credibility.py's _CACHE), so this costs no
    extra network or LLM calls beyond what the original single-score version
    did.
    """
    if not sources:
        return "_No sources for this answer yet._"

    parts = [f"**{len(sources)} source(s), scored by `credibility.score_url`**\n"]
    for i, source in enumerate(sources, 1):
        result = score_url(source["url"])
        interval = score_with_uncertainty(source["url"])
        calibrated = calibrated_probability(source["url"])
        title = source.get("title") or source["url"]

        ci_text = ""
        if interval["n_bootstrap"] > 0:
            ci_text = f"(90% CI: {interval['low']:.2f}–{interval['high']:.2f})"

        parts.append(
            f'<p style="margin:.6em 0 .2em"><b>{i}.</b> '
            f'<a href="{source["url"]}" target="_blank">{title}</a> &nbsp; '
            f"{_chip(result['score'], ci_text)}</p>"
        )
        if source.get("snippet"):
            parts.append(
                f'<p style="margin:.1em 0;color:#57606a;font-size:.9em">'
                f'{source["snippet"]}</p>'
            )
        parts.append(
            "<details><summary>Why this score?</summary>"
            f"<p style='margin:.4em 0'>{result['explanation']}</p>"
            f"{_uncertainty_html(interval)}"
            f"{_calibration_html(calibrated)}"
            "</details>"
        )
    return "\n".join(parts)


def score_one_url(url: str) -> str:
    """The standalone URL scorer. Works with no API key, same as in main.py.

    PART 3: now also shows the Stage 4 interval and Stage 3 calibrated
    probability, matching main.py's sidebar probe.
    """
    if not url or not url.strip():
        return ""
    clean = url.strip()
    result = score_url(clean)
    interval = score_with_uncertainty(clean)
    calibrated = calibrated_probability(clean)

    ci_text = ""
    if interval["n_bootstrap"] > 0:
        ci_text = f"(90% CI: {interval['low']:.2f}–{interval['high']:.2f})"

    return (
        f"{_chip(result['score'], ci_text)}"
        f"<p style='margin:.5em 0'>{result['explanation']}</p>"
        f"{_calibration_html(calibrated)}"
    )


def status_markdown() -> str:
    """Which keys were found. Mirrors the Streamlit sidebar's status block."""
    from chat_backend import TRACING_ENABLED

    return "\n".join(
        [
            ("✅" if os.getenv("ANTHROPIC_API_KEY") else "❌") + " Anthropic API key",
            ("✅" if os.getenv("SERPAPI_API_KEY") else "⬜") + " SerpAPI key (optional)",
            ("✅" if TRACING_ENABLED else "⬜") + " Langfuse tracing (optional)",
        ]
    )


def model_status_markdown() -> str:
    """
    PART 3: which fitted models credibility.py actually loaded at import time.
    Mirrors main.py's "Model status" panel — purely informational, so it's
    obvious at a glance during a live demo whether Stage 2/3/4 are active.
    """
    return "\n".join(
        [
            ("✅" if _LEARNED_WEIGHTS else "⬜") + " Stage 2 — learned weights (train_weights.py)",
            ("✅" if _CALIBRATION else "⬜") + " Stage 3 — calibration (calibration.py)",
            ("✅" if _BOOTSTRAP_MODELS else "⬜")
            + f" Stage 4 — bootstrap uncertainty ({len(_BOOTSTRAP_MODELS)} models)",
        ]
    )


def respond(
    message: str,
    history: List[Dict[str, str]],
    user: str,
    email: str,
    use_serpapi: bool,
) -> Tuple[List[Dict[str, str]], str, str]:
    """
    Handle one chat turn.

    :param history: Gradio `type="messages"` history — [{"role", "content"}, ...]
    :return: (updated history, sources HTML, cleared textbox)
    """
    if not message or not message.strip():
        return history, "", ""

    history = list(history) + [{"role": "user", "content": message}]

    if not os.getenv("ANTHROPIC_API_KEY"):
        history.append(
            {
                "role": "assistant",
                "content": (
                    "No `ANTHROPIC_API_KEY` found. Copy `.env.example` to `.env` and add "
                    "your key — on a Hugging Face Space, add it under Settings → "
                    "Variables and secrets. The URL scorer below still works without one."
                ),
            }
        )
        return history, "", ""

    # Build the request separately from the displayed history. Search context is
    # useful for this turn only — writing it back into the history would re-send
    # it on every later turn and inflate the conversation.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    serp_sources: List[Dict[str, str]] = []

    if use_serpapi and os.getenv("SERPAPI_API_KEY"):
        try:
            for r in search_serpapi(message, os.getenv("SERPAPI_API_KEY"))[:5]:
                serp_sources.append(
                    {
                        "title": r.get("title", "Untitled"),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                )
            if serp_sources:
                context = "\n\nSearch results for reference:\n" + "".join(
                    f"- {s['title']} ({s['url']})\n  {s['snippet']}\n" for s in serp_sources
                )
                api_messages[-1] = {"role": "user", "content": message + context}
        except Exception as exc:  # noqa: BLE001 — a failed search shouldn't kill the turn
            serp_sources = []
            gr.Warning(f"SerpAPI search failed: {exc}")

    try:
        answer, citations = ask_claude(api_messages, user, email, f"{user}_{email}")
    except Exception as exc:  # noqa: BLE001 — show the error rather than a blank screen
        history.append({"role": "assistant", "content": f"Error: {exc}"})
        return history, "", ""

    sources = merge_sources(citations, serp_sources)
    history.append({"role": "assistant", "content": answer})
    return history, render_sources(sources), ""


with gr.Blocks(title="CS676 — Credibility Chatbot") as demo:
    gr.Markdown("# 🔍 Credibility-Scored Research Assistant")
    gr.Markdown(
        "Ask a research question. The answer cites its sources, and every source "
        "carries a credibility score from `credibility.score_url` — the function "
        "you are graded on."
    )

    with gr.Row():
        with gr.Column(scale=3):
            # Gradio 6 takes [{"role": ..., "content": ...}] natively; the
            # `type="messages"` argument that Gradio 4/5 needed was removed.
            chatbot = gr.Chatbot(height=430, label="Conversation")
            with gr.Row():
                box = gr.Textbox(
                    placeholder="Ask a research question...",
                    show_label=False,
                    scale=8,
                    submit_btn=True,
                )
                clear = gr.Button("Clear", scale=1)
            sources_panel = gr.Markdown("_No sources yet._", label="Sources")

        with gr.Column(scale=1):
            gr.Markdown("### Session")
            user_box = gr.Textbox(value="student", label="Name")
            email_box = gr.Textbox(value="student@pace.edu", label="Email")
            serp_toggle = gr.Checkbox(value=False, label="Also search with SerpAPI")

            gr.Markdown("### Status")
            gr.Markdown(status_markdown())

            # PART 3: same "Model status" panel main.py's sidebar has.
            gr.Markdown("### Model status")
            gr.Markdown(model_status_markdown())

            gr.Markdown("### Score any URL")
            gr.Markdown("_Works without an API key._")
            probe = gr.Textbox(
                placeholder="https://arxiv.org/abs/1706.03762",
                show_label=False,
                submit_btn=True,
            )
            probe_out = gr.Markdown()

    inputs = [box, chatbot, user_box, email_box, serp_toggle]
    outputs = [chatbot, sources_panel, box]
    box.submit(respond, inputs, outputs)

    probe.submit(score_one_url, probe, probe_out)
    clear.click(lambda: ([], "_No sources yet._", ""), None, outputs)


if __name__ == "__main__":
    # 0.0.0.0:7860 is what Hugging Face Spaces expects. Locally it just means the
    # app is reachable at http://localhost:7860.
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
