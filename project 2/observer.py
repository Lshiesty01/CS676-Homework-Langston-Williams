"""
observer.py — a third agent that watches a conversation and judges it.

Nobody inside a conversation can tell whether it went well: each persona is busy
being itself. The observer reads the finished transcript, together with the
persona files of everyone in it, and returns a structured `Verdict`:

    summary        two or three sentences on what happened
    goal_met       did the conversation do what its topic set out to do?
    in_character   did every participant stay in voice?
    scores         1-5 on relevance, consistency, and resolution
    issues         specific problems, quoting the line where possible

This is the AI-judge piece of the course rubric. It costs one extra API call per
judged conversation and is only run when asked.

The model is asked for JSON. Following the lesson of the frontmatter bug in
persona.py, the parser is strict about what matters (the fields have the right
types) and generous about everything else (code fences, prose around the JSON,
out-of-range scores, missing keys).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .conversation import Conversation
from .llm import LLM
from .persona import PersonaSpec

SCORE_KEYS = ("relevance", "consistency", "resolution")

_OBSERVER_SYSTEM = """You are an impartial observer evaluating a simulated conversation
between role-played characters. You are not one of the characters.

You will receive each character's specification and the full transcript. Judge it
and reply with ONLY a JSON object, no prose and no code fences, of this shape:

{
  "summary": "<two or three sentences on what happened>",
  "goal_met": <true if the conversation achieved what its topic set out to do>,
  "in_character": <true if every character stayed consistent with their spec>,
  "scores": {
    "relevance": <1-5, did they stay on the topic>,
    "consistency": <1-5, did each speaker match their personality and voice>,
    "resolution": <1-5, did the conversation get somewhere>
  },
  "issues": ["<specific problem, quoting the line if possible>", ...]
}

Be specific and honest. An empty issues list is fine if there were none."""


@dataclass
class Verdict:
    """The observer's judgment of one conversation."""

    summary: str
    goal_met: Optional[bool] = None
    in_character: Optional[bool] = None
    scores: Dict[str, int] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    parsed: bool = True   # False when the model's reply could not be read as JSON

    def overall(self) -> Optional[float]:
        """Mean of the available scores, or None if there are none."""
        if not self.scores:
            return None
        return round(sum(self.scores.values()) / len(self.scores), 2)

    def to_markdown(self) -> str:
        """Render for the terminal, the web UI, or appending to a transcript."""

        def yes_no(value: Optional[bool]) -> str:
            return "unknown" if value is None else ("yes" if value else "no")

        lines = ["## Observer verdict", "", self.summary, ""]
        lines.append(f"- Goal met: {yes_no(self.goal_met)}")
        lines.append(f"- Stayed in character: {yes_no(self.in_character)}")
        if self.scores:
            score_text = ", ".join(f"{k} {v}/5" for k, v in self.scores.items())
            lines.append(f"- Scores: {score_text} (overall {self.overall()})")
        if self.issues:
            lines.append("- Issues:")
            lines.extend(f"  - {issue}" for issue in self.issues)
        if not self.parsed:
            lines.append("- _(The observer's reply was not valid JSON; showing it as the summary.)_")
        return "\n".join(lines)


def _extract_json(text: str) -> Optional[dict]:
    """Find and decode the first JSON object in `text`, tolerating fences and prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no"}:
        return False
    return None


def parse_verdict(text: str) -> Verdict:
    """
    Turn the observer model's reply into a `Verdict`.

    Never raises. Unreadable output becomes a Verdict whose summary is the raw
    reply and whose `parsed` flag is False, so the caller can still show it.
    """
    data = _extract_json(text)
    if data is None:
        return Verdict(summary=text.strip() or "(empty reply)", parsed=False)

    scores: Dict[str, int] = {}
    raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    for key in SCORE_KEYS:
        try:
            scores[key] = max(1, min(5, int(round(float(raw_scores[key])))))
        except (KeyError, TypeError, ValueError):
            continue  # a missing score is left out, not invented

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]

    return Verdict(
        summary=str(data.get("summary") or "").strip() or "(no summary given)",
        goal_met=_as_bool(data.get("goal_met")),
        in_character=_as_bool(data.get("in_character")),
        scores=scores,
        issues=[str(i) for i in issues if str(i).strip()],
    )


def observe(conversation: Conversation, personas: Sequence[PersonaSpec], llm: LLM) -> Verdict:
    """
    Ask the model to judge a finished conversation.

    :param conversation: the dialogue to judge
    :param personas:     the specs of the participants, so the judge can check
                         consistency against what each character was meant to be
    :param llm:          any LLM; one call is made
    """
    if not conversation.turns:
        return Verdict(summary="Nothing was said, so there is nothing to judge.", parsed=False)

    specs = "\n\n".join(f"### {p.name} ({p.role})\n{p.body}" for p in personas)
    prompt = (
        f"Topic: {conversation.topic}\n\n"
        f"## Character specifications\n\n{specs or '(not available)'}\n\n"
        f"## Transcript\n\n{conversation.transcript()}"
    )
    reply = llm.complete(_OBSERVER_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=1024)
    return parse_verdict(reply)
