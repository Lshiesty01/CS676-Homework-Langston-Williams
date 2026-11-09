"""
app.py — a Gradio web front end for PersonaForge (the Hugging Face Space entry point).

This file is deliberately thin: every decision is made inside the `personaforge`
package, exactly as in the terminal app. The UI only

    * keeps one Orchestrator per browser session (in gr.State),
    * streams each line of dialogue to the chat as it is produced, and
    * shows the persona files and the latest transcript.

Model selection, in order:

    1. a key pasted into the "Your Anthropic API key" box (used for this
       session only, never stored or logged),
    2. the ANTHROPIC_API_KEY secret set on the Space,
    3. otherwise offline mode, with scripted replies.

Each session writes to its own temporary folder, so visitors to a public Space
never see or overwrite each other's personas.

Run locally:   uv run --group web app.py      then open http://localhost:7860
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
from pathlib import Path

# On a Space the package is shipped as src/personaforge rather than installed.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from personaforge import ClaudeLLM, Orchestrator, list_personas  # noqa: E402
from personaforge.llm import DemoLLM  # noqa: E402

load_dotenv()

MODEL = os.getenv("PERSONAFORGE_MODEL", "claude-haiku-4-5")  # cheap default for a public demo

EXAMPLES = [
    "Create a skeptical data scientist named Ada and an overconfident startup founder",
    "Have them argue about whether the model is ready to ship, for 8 turns",
    "Review that conversation",
    "Who do we have?",
]


def _new_orchestrator(api_key: str) -> tuple[Orchestrator, str]:
    """Build an orchestrator for this session and describe the mode it is in."""
    persona_dir = Path(tempfile.mkdtemp(prefix="personaforge-"))
    if api_key.strip():
        llm = ClaudeLLM(model=MODEL)
        import anthropic

        llm._client = anthropic.Anthropic(api_key=api_key.strip(), max_retries=2)
        return Orchestrator(llm, persona_dir), f"Live — your key, {MODEL}"
    if os.getenv("ANTHROPIC_API_KEY"):
        return Orchestrator(ClaudeLLM(model=MODEL), persona_dir), f"Live — {MODEL}"
    return Orchestrator(DemoLLM(), persona_dir), "Offline — scripted replies (add a key for real personas)"


def _roster(orch: Orchestrator | None) -> str:
    if orch is None:
        return "_No personas yet._"
    specs = list_personas(orch.persona_dir)
    if not specs:
        return "_No personas yet._"
    return "\n".join(f"- **{s.name}** ({s.role}) — {s.summary}" for s in specs)


def _latest_transcript(orch: Orchestrator | None):
    if orch is None or not orch.transcript_dir.exists():
        return None
    files = sorted(orch.transcript_dir.glob("*.md"))
    return str(files[-1]) if files else None


def respond(message: str, history: list, orch: Orchestrator | None, mode: str, api_key: str):
    """
    Handle one user message, streaming dialogue lines into the chat.

    The orchestrator runs in a worker thread and pushes each conversation turn
    onto a queue; this generator drains the queue and yields the updated chat,
    so the user watches the conversation unfold instead of waiting for it.
    """
    history = list(history or [])
    if not message.strip():
        yield history, orch, mode, _roster(orch), _latest_transcript(orch), ""
        return
    if orch is None:
        orch, mode = _new_orchestrator(api_key or "")

    history.append({"role": "user", "content": message})
    events: queue.Queue = queue.Queue()
    outcome: dict = {}

    def on_start(topic: str) -> None:
        events.put(("start", topic))

    def on_turn(turn) -> None:
        events.put(("turn", turn))

    def work() -> None:
        try:
            outcome["result"] = orch.handle(message, on_turn=on_turn, on_start=on_start)
        except Exception as exc:  # noqa: BLE001 — show it in the chat, keep the app alive
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            events.put(("done", None))

    threading.Thread(target=work, daemon=True).start()

    while True:
        kind, payload = events.get()
        if kind == "done":
            break
        if kind == "start":
            history.append({"role": "assistant", "content": f"*── conversation: {payload} ──*"})
        else:
            history.append({"role": "assistant", "content": f"**{payload.speaker}:** {payload.text}"})
        yield history, orch, mode, _roster(orch), _latest_transcript(orch), ""

    if "error" in outcome:
        history.append({"role": "assistant", "content": f"⚠️ {outcome['error']}"})
    else:
        result = outcome["result"]
        reply = result.reply
        if result.verdict is not None and result.verdict.to_markdown() not in reply:
            reply += "\n\n" + result.verdict.to_markdown()
        history.append({"role": "assistant", "content": reply})
    yield history, orch, mode, _roster(orch), _latest_transcript(orch), ""


def reset(api_key: str):
    """Start a fresh session (new folder, new history), e.g. after entering a key."""
    orch, mode = _new_orchestrator(api_key or "")
    return [], orch, mode, _roster(orch), None


with gr.Blocks(title="PersonaForge") as demo:
    gr.Markdown(
        "# PersonaForge\n"
        "Invent characters, then make them talk to each other. "
        "Ask in plain language — Claude decides which tools to use."
    )
    orch_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520, label="Stage")
            box = gr.Textbox(placeholder="e.g. make two doctors and a nurse and have them argue", show_label=False)
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=2):
            mode_box = gr.Textbox(label="Mode", interactive=False)
            key_box = gr.Textbox(
                label="Your Anthropic API key (optional)",
                type="password",
                placeholder="sk-ant-… used for this session only",
            )
            new_btn = gr.Button("New session")
            roster = gr.Markdown("_No personas yet._", label="Personas")
            transcript_file = gr.File(label="Latest transcript")

    outputs = [chat, orch_state, mode_box, roster, transcript_file]
    box.submit(respond, [box, chat, orch_state, mode_box, key_box], outputs + [box])
    new_btn.click(reset, [key_box], outputs)
    demo.load(reset, [key_box], outputs)


if __name__ == "__main__":
    # Spaces expects 0.0.0.0:7860; binding to localhost builds fine and then times out.
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
