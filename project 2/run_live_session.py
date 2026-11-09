"""
run_live_session.py — run a scripted live session and record everything the report needs.

    uv run run_live_session.py                     # claude-haiku-4-5, the cheap default
    uv run run_live_session.py --model claude-sonnet-4-5

Each scenario below is typed into the orchestrator exactly as a user would type
it. The script records, for every step: what was typed, which tools Claude chose
and with what arguments, the stage manager's reply, any conversation and
observer verdict, errors, wall time, and API calls/tokens used.

Output: `live_session/live_session_log.md` (one file to read or share) plus the
persona files and transcripts the session produced, in the same folder.

Cost: roughly 50-70 API calls. On claude-haiku-4-5 that is well under a dollar.
This file is a reporting harness, not part of the package; it only uses the
package's public API.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from personaforge import ClaudeLLM, LLMError, Orchestrator, list_personas

# Each scenario tests something specific; the label says what, for the report.
SCENARIOS = [
    ("Free phrasing, two personas in one sentence",
     "make a skeptical data scientist named Ada and a startup founder named Leo "
     "who thinks the churn model is ready to ship"),
    ("Conversation with turn count and review in one request",
     "have Ada and Leo argue about whether to ship the churn model, 8 turns, and review it afterwards"),
    ("Follow-up that relies on session memory",
     "now add a compliance officer and have all three talk about it for 9 turns"),
    ("Vague request — should ask a clarifying question",
     "create a persona"),
    ("Roster question in natural language",
     "who do we have so far?"),
    ("The README's medical scenario, created and run in one message",
     "create a patient with chronic back pain who distrusts doctors and a blunt internist "
     "running forty minutes late, then have them talk about the MRI results for 10 turns and review it"),
    ("Ambiguous reference — two plausible matches",
     "create another doctor, a cautious pediatrician, then have the doctor talk to Ada about bias in medical data"),
    ("Unknown persona",
     "have Gandalf and Leo talk about the roadmap"),
]


@dataclass
class RecordingLLM:
    """Wraps ClaudeLLM to note every tool call the orchestrator's model makes."""

    inner: ClaudeLLM
    tool_log: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system, messages, max_tokens=2048):
        return self.inner.complete(system, messages, max_tokens)

    def complete_with_tools(self, system, messages, tools, max_tokens=2048):
        reply = self.inner.complete_with_tools(system, messages, tools, max_tokens)
        for call in reply.tool_calls:
            self.tool_log.append({"name": call.name, "input": call.input})
        return reply


def fmt_args(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def main() -> int:
    # Windows consoles default to cp1252, which cannot print the › and — used below.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Record a live PersonaForge session for the report.")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--out", default="live_session")
    args = parser.parse_args()
    load_dotenv()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    persona_dir = out / "personas"

    llm = RecordingLLM(ClaudeLLM(model=args.model))
    orch = Orchestrator(llm, persona_dir)
    log: List[str] = [
        "# PersonaForge live session log",
        "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Model (orchestrator, personas, and observer): `{args.model}`",
        "",
    ]
    totals_before = llm.inner.usage()

    for n, (label, text) in enumerate(SCENARIOS, 1):
        print(f"\n[{n}/{len(SCENARIOS)}] {label}\n  you › {text}")
        llm.tool_log.clear()
        before = llm.inner.usage()
        start = time.time()
        lines: List[str] = []

        def on_turn(turn):
            print(f"    {turn.speaker}: {turn.text[:100]}")

        try:
            result = orch.handle(text, on_turn=on_turn)
            error = None
        except Exception as exc:  # noqa: BLE001 — record it and keep going
            result, error = None, f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - start
        after = llm.inner.usage()
        used = {k: after[k] - before[k] for k in after}

        lines += [f"## {n}. {label}", "", f"**You:** {text}", ""]
        lines.append("**Tools Claude chose:** " + (
            "; ".join(f"`{c['name']}({fmt_args(c['input'])})`" for c in llm.tool_log) or "none"))
        lines.append("")
        if error:
            lines += [f"**Error:** {error}", ""]
        else:
            lines += [f"**Stage manager:** {result.reply}", ""]
            for event in result.events:
                lines.append(f"- {event}")
            lines.append("")
            if result.conversation is not None:
                c = result.conversation
                lines += ["", "<details><summary>Transcript</summary>", "", c.transcript(), "", "</details>", ""]
            if result.verdict is not None:
                lines += [result.verdict.to_markdown(), ""]
        lines += [f"_Cost: {used['calls']} API calls, {used['input_tokens']} input + "
                  f"{used['output_tokens']} output tokens, {elapsed:.1f}s_", ""]
        log += lines
        print(f"  stage › {(result.reply if result else error)[:160]}")
        print(f"  ({used['calls']} calls, {elapsed:.1f}s)")

        reply = error or result.reply
        if "API key was rejected" in reply or "credit balance" in reply.lower():
            print("\nStopping early: fix the API key or billing, then run again.")
            break

    # Personas as authored, so the report can comment on their quality.
    log += ["## Personas authored this session", ""]
    for spec in list_personas(persona_dir):
        log += [f"### {spec.name} ({spec.role})", "", spec.body, ""]

    total = llm.inner.usage()
    total = {k: total[k] - totals_before[k] for k in total}
    log += ["## Totals", "",
            f"- API calls: {total['calls']}",
            f"- Input tokens: {total['input_tokens']}",
            f"- Output tokens: {total['output_tokens']}", ""]

    out.mkdir(parents=True, exist_ok=True)
    path = out / "live_session_log.md"
    path.write_text("\n".join(log), encoding="utf-8")
    print(f"\nWrote {path}  —  {total['calls']} calls, "
          f"{total['input_tokens']} in / {total['output_tokens']} out tokens")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LLMError as exc:
        print(f"\n{exc}")
        sys.exit(1)
