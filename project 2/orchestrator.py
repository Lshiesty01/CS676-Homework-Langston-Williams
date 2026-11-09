"""
orchestrator.py — the agent you talk to.

You type "make two doctors and a nurse and have them argue about the
diagnosis", and something has to turn that into actions. That something is now
Claude itself, using tool calling:

    1. The user's message goes to the model together with a list of TOOLS
       (create_persona, list_personas, show_persona, start_conversation,
       review_conversation).
    2. The model either answers in prose — including asking a clarifying
       question — or asks for one or more tools to be run.
    3. We run them, send the results back as `tool_result` blocks, and repeat
       until the model answers in prose or MAX_TOOL_ROUNDS is reached.

The model chooses, so multi-step requests, unusual phrasings, and follow-ups
like "now have the nurse and the second doctor talk" all work without a single
new pattern.

The old regular-expression router is kept as `_handle_with_regex`, used only
when the LLM cannot do tool calls — that is, in `--offline` mode and in the
original tests. It is a fallback, not the design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent import agent_from_spec
from .conversation import TRANSCRIPT_SUBDIR, Conversation, Turn, run_conversation, save_transcript
from .llm import LLM, LLMError, ToolCall, supports_tools
from .observer import Verdict, observe
from .persona import (
    DEFAULT_PERSONA_DIR,
    PersonaSpec,
    find_persona,
    generate_persona,
    list_personas,
    match_personas,
    save_persona,
)

# Safety valves for the agent loop.
MAX_TOOL_ROUNDS = 8          # model↔tool round trips per user message
MAX_PERSONAS_PER_MESSAGE = 6  # create_persona calls per user message (each is an API call)
MIN_TURNS, MAX_TURNS, DEFAULT_TURNS = 2, 20, 6
HISTORY_USER_TURNS = 12       # how many past user messages the orchestrator remembers
TRANSCRIPT_PREVIEW_CHARS = 4000  # how much of a conversation is shown back to the model

# Used by the regex fallback's chat path. The name is kept because the offline
# stub recognises the phrase "stage manager" and the original tests import it.
_STAGE_MANAGER_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. You help them
decide who to create and what those characters should discuss.

Be brief — two or three sentences. When the user seems ready to act, remind them they
can say things like "create a persona doctor", "list personas", or "have them talk
about the test results"."""

_TOOL_SYSTEM = """You are the stage manager of PersonaForge, a workshop where the user builds
fictional characters (personas) and has them talk to each other.

You act through tools:
- create_persona: author and save one new character. Call it once per character;
  "two doctors and a nurse" is three calls with three distinct descriptions.
- list_personas / show_persona: see who exists and what they are like.
- start_conversation: put two or more existing personas in a scene.
- review_conversation: have an impartial observer judge the most recent conversation.

Guidelines:
- If a request is too vague to act on (e.g. "create a persona" with no hint of who),
  ask ONE short clarifying question instead of inventing something.
- If several personas could match a name, ask which one the user meant.
- Use the personas that already exist when the user refers to them; do not re-create them.
- Every persona and every conversation turn costs an API call. Do not start a
  conversation the user did not ask for, and do not exceed what they asked.
- After tools run, reply to the user in two or three sentences. Do not repeat the
  whole conversation back — the user has already watched it.

Personas currently saved:
{roster}"""

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "create_persona",
        "description": (
            "Author a new fictional character from a short description and save it as a "
            "markdown file. Returns the character's name, role, and summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Who to create, in a sentence: role, situation, temperament, and any "
                        "name the user asked for. E.g. 'a skeptical data scientist named Ada'."
                    ),
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_personas",
        "description": "List every saved persona with its name, role, and one-line summary.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_persona",
        "description": "Show the full character sheet of one saved persona.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name, first name, or role of the persona."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "start_conversation",
        "description": (
            "Run a conversation between two or more saved personas and save the transcript. "
            "Participants speak in the order given. Returns the transcript."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Names (or unambiguous first names / roles) of the personas, first speaker first.",
                },
                "topic": {
                    "type": "string",
                    "description": "The scene or subject, e.g. 'the MRI results' or 'arguing about the diagnosis'.",
                },
                "turns": {
                    "type": "integer",
                    "minimum": MIN_TURNS,
                    "maximum": MAX_TURNS,
                    "description": f"Maximum lines of dialogue in total. Default {DEFAULT_TURNS}.",
                },
                "review": {
                    "type": "boolean",
                    "description": "Also have the observer judge the conversation afterwards. Default false.",
                },
            },
            "required": ["participants", "topic"],
        },
    },
    {
        "name": "review_conversation",
        "description": (
            "Have an impartial observer judge the most recent conversation: summary, whether "
            "its goal was met, whether everyone stayed in character, and 1-5 scores."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


@dataclass
class Result:
    """What the orchestrator did, in a form the UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None   # the last conversation run, if any
    verdict: Optional[Verdict] = None             # the last observer verdict, if any


class ToolInputError(ValueError):
    """A tool was called with arguments we cannot act on. Shown to the model."""


class Orchestrator:
    """Turns what the user typed into persona actions."""

    def __init__(
        self,
        llm: LLM,
        persona_dir: Path = DEFAULT_PERSONA_DIR,
        use_tools: Optional[bool] = None,
    ) -> None:
        """
        :param llm:         the model. Tool use is enabled if it supports it.
        :param persona_dir: where persona files are read and written.
        :param use_tools:   force the routing mode; by default, tools are used
                            whenever the model offers `complete_with_tools`.
        """
        self.llm = llm
        self.persona_dir = Path(persona_dir)
        self.use_tools = supports_tools(llm) if use_tools is None else use_tools
        self.history: List[Dict[str, Any]] = []
        self.last_conversation: Optional[Conversation] = None
        self.last_participants: List[PersonaSpec] = []

    @property
    def transcript_dir(self) -> Path:
        return self.persona_dir / TRANSCRIPT_SUBDIR

    # -- actions shared by both routers ---------------------------------------

    def _create(self, description: str) -> Tuple[PersonaSpec, Path]:
        spec = generate_persona(description, self.llm)
        return spec, save_persona(spec, self.persona_dir)

    def _resolve(self, names: List[str]) -> List[PersonaSpec]:
        """Map names to personas, raising ToolInputError on a miss or an ambiguity."""
        specs: List[PersonaSpec] = []
        for name in names:
            matches = match_personas(name, self.persona_dir)
            if not matches:
                raise ToolInputError(f"I couldn't find a persona matching '{name}'.")
            if len(matches) > 1:
                options = ", ".join(f"{m.name} ({m.role})" for m in matches)
                raise ToolInputError(f"'{name}' is ambiguous — it could be {options}.")
            if any(m.name == matches[0].name for m in specs):
                raise ToolInputError(f"{matches[0].name} is listed twice.")
            specs.append(matches[0])
        return specs

    def _run(
        self,
        specs: List[PersonaSpec],
        topic: str,
        turns: int,
        on_turn: Optional[Callable[[Turn], None]],
        on_start: Optional[Callable[[str], None]],
    ) -> Tuple[Conversation, Path]:
        agents = [agent_from_spec(s, self.llm) for s in specs]
        if on_start is not None:
            on_start(topic)
        conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)
        path = save_transcript(conversation, self.transcript_dir)
        self.last_conversation, self.last_participants = conversation, specs
        return conversation, path

    def _review(self) -> Verdict:
        if self.last_conversation is None:
            raise ToolInputError("There is no conversation to review yet.")
        return observe(self.last_conversation, self.last_participants, self.llm)

    def _roster(self) -> str:
        specs = list_personas(self.persona_dir)
        if not specs:
            return "(none yet)"
        return "\n".join(f"- {s.name} ({s.role}) — {s.summary}" for s in specs)

    # -- entry point ----------------------------------------------------------

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Work out what the user meant and do it."""
        if not text.strip():
            return Result(reply="Say what you'd like to do — for example, create a persona.")
        if self.use_tools:
            return self._handle_with_tools(text, on_turn, on_start)
        return self._handle_with_regex(text, on_turn, on_start)

    # -- tool-use router ------------------------------------------------------

    def _handle_with_tools(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]],
        on_start: Optional[Callable[[str], None]],
    ) -> Result:
        """The agent loop: model → tools → model, until the model answers in prose."""
        result = Result(reply="")
        messages: List[Dict[str, Any]] = self.history + [{"role": "user", "content": text}]
        budget = {"create_persona": MAX_PERSONAS_PER_MESSAGE}
        final_text: Optional[str] = None

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                system = _TOOL_SYSTEM.format(roster=self._roster())
                reply = self.llm.complete_with_tools(system, messages, TOOLS, max_tokens=1024)

                # The API rejects an assistant turn with no content blocks.
                content = reply.content or [{"type": "text", "text": "(no reply)"}]
                messages.append({"role": "assistant", "content": content})

                if not reply.wants_tools:
                    final_text = reply.text
                    break

                tool_results = []
                for call in reply.tool_calls:
                    output, is_error = self._run_tool(call, result, budget, on_turn, on_start)
                    block = {"type": "tool_result", "tool_use_id": call.id, "content": output}
                    if is_error:
                        block["is_error"] = True
                    tool_results.append(block)
                messages.append({"role": "user", "content": tool_results})
        except LLMError as exc:
            # Don't keep a half-finished exchange: it may end in an unanswered
            # tool_use, which the API would reject on the next message.
            result.reply = f"The model call failed: {exc}"
            return result

        if final_text is None:
            final_text = (
                f"I stopped after {MAX_TOOL_ROUNDS} steps without finishing. "
                "Here is what got done; try a smaller request."
            )
        result.reply = final_text or "Done."
        self.history = _trim_history(messages, HISTORY_USER_TURNS)
        return result

    def _run_tool(
        self,
        call: ToolCall,
        result: Result,
        budget: Dict[str, int],
        on_turn: Optional[Callable[[Turn], None]],
        on_start: Optional[Callable[[str], None]],
    ) -> Tuple[str, bool]:
        """
        Run one tool call. Returns (text for the model, is_error).

        Bad arguments and missing personas are reported back to the model as
        errors so it can correct itself or ask the user — they never crash the
        loop. API failures (LLMError) propagate to `_handle_with_tools`.
        """
        args = call.input or {}
        try:
            if call.name == "create_persona":
                description = str(args.get("description", "")).strip()
                if not description:
                    raise ToolInputError("description is empty — ask the user who to create.")
                if budget["create_persona"] <= 0:
                    raise ToolInputError(
                        f"Limit of {MAX_PERSONAS_PER_MESSAGE} new personas per message reached."
                    )
                budget["create_persona"] -= 1
                spec, path = self._create(description)
                result.events.append(f"wrote {path}")
                return f"Created {spec.name} ({spec.role}) — {spec.summary}. Saved to {path.name}.", False

            if call.name == "list_personas":
                return self._roster(), False

            if call.name == "show_persona":
                spec = self._resolve([str(args.get("name", ""))])[0]
                return spec.to_markdown(), False

            if call.name == "start_conversation":
                names = args.get("participants") or []
                if not isinstance(names, list) or len(names) < 2:
                    raise ToolInputError("participants must list at least two personas.")
                topic = str(args.get("topic", "")).strip() or "an initial conversation"
                turns = _clamp_turns(args.get("turns", DEFAULT_TURNS))
                specs = self._resolve([str(n) for n in names])

                conversation, path = self._run(specs, topic, turns, on_turn, on_start)
                result.conversation = conversation
                result.events.append(f"wrote {path}")

                summary = _conversation_summary(conversation)
                if args.get("review"):
                    result.verdict = self._review()
                    summary += "\n\n" + result.verdict.to_markdown()
                return summary, False

            if call.name == "review_conversation":
                result.verdict = self._review()
                return result.verdict.to_markdown(), False

            return f"Unknown tool '{call.name}'.", True
        except ToolInputError as exc:
            return str(exc), True

    # -- regex fallback (offline mode) ---------------------------------------

    def _handle_with_regex(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]],
        on_start: Optional[Callable[[str], None]],
    ) -> Result:
        """
        The original pattern router, for models without tool support.

        Patterns are tried in order and the first match wins. It recognises a
        handful of fixed phrasings; anything else is passed to the model as chat.
        """
        lowered = text.strip().lower()

        create = re.search(
            r"(?:create|make|add|build|generate)\s+(?:a\s+|an\s+|new\s+)*persona\s*(?:for|of|who is|that is|:)?\s*(.*)",
            lowered,
        )
        if create:
            description = create.group(1).strip()
            if not description:
                return Result(reply="Tell me what kind of persona — for example, 'create a persona doctor'.")
            spec, path = self._create(description)
            return Result(reply=f"Created **{spec.name}** ({spec.role}) — {spec.summary}", events=[f"wrote {path}"])

        if re.search(r"\b(list|show|what|who)\b.*\bpersona", lowered) or lowered in {"list", "ls"}:
            return self._list_result()

        if re.search(r"\b(review|judge|evaluate|score)\b", lowered):
            try:
                verdict = self._review()
            except ToolInputError as exc:
                return Result(reply=str(exc))
            return Result(reply=verdict.to_markdown(), verdict=verdict)

        if re.search(r"\b(?:talk|conversation|converse|chat|discuss|speak)\b", lowered):
            topic_match = re.search(r"\babout\s+(.*)$", lowered)
            topic = topic_match.group(1).strip() if topic_match else "an initial consultation"
            turns_match = re.search(r"\b(\d+)\s+turns?\b", lowered)
            turns = _clamp_turns(turns_match.group(1)) if turns_match else DEFAULT_TURNS
            topic = re.sub(r"\s*\b(?:for\s+)?\d+\s+turns?\b", "", topic).strip() or topic

            who_match = re.search(r"\bbetween\s+(.*?)(?:\s+about\b|$)", lowered)
            who = None
            if who_match:
                who = [w.strip() for w in re.split(r"\s+and\s+|,", who_match.group(1)) if w.strip()]
            return self._regex_converse(topic, who, turns, on_turn, on_start)

        return self._regex_chat(text)

    def _list_result(self) -> Result:
        specs = list_personas(self.persona_dir)
        if not specs:
            return Result(reply="No personas yet. Try: create a persona patient")
        lines = [f"- **{s.name}** ({s.role}) — {s.summary}" for s in specs]
        return Result(reply=f"{len(specs)} persona(s) in `{self.persona_dir}`:\n" + "\n".join(lines))

    def _regex_converse(self, topic, who, turns, on_turn, on_start) -> Result:
        if who:
            specs = []
            for name in who:
                found = find_persona(name, self.persona_dir)
                if found is None:
                    return Result(reply=f"I couldn't find a persona matching '{name}'.")
                specs.append(found)
        else:
            specs = list_personas(self.persona_dir)[:2]

        if len(specs) < 2:
            return Result(reply="I need at least two personas before they can talk. Create another one first.")

        conversation, path = self._run(specs, topic, turns, on_turn, on_start)
        reply = f"{len(conversation.turns)} turns between {' and '.join(s.name for s in specs)}."
        if conversation.error:
            reply += f" Stopped early: {conversation.error}"
        return Result(reply=reply, events=[f"wrote {path}"], conversation=conversation)

    def _regex_chat(self, text: str) -> Result:
        self.history.append({"role": "user", "content": text})
        reply = self.llm.complete(_STAGE_MANAGER_SYSTEM, self.history, max_tokens=512)
        self.history.append({"role": "assistant", "content": reply})
        self.history = _trim_history(self.history, HISTORY_USER_TURNS)
        return Result(reply=reply)


# -- helpers -------------------------------------------------------------------


def _clamp_turns(value: Any) -> int:
    """Coerce a model- or user-supplied turn count into the allowed range."""
    try:
        turns = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TURNS
    return max(MIN_TURNS, min(MAX_TURNS, turns))


def _conversation_summary(conversation: Conversation) -> str:
    """What the model is told after a conversation: outcome plus a capped transcript."""
    status = f"{len(conversation.turns)} turns between {', '.join(conversation.participants)}."
    if conversation.ended_by:
        status += f" {conversation.ended_by} ended it naturally."
    if conversation.error:
        status += f" It stopped early because {conversation.error}."
    transcript = conversation.transcript()
    if len(transcript) > TRANSCRIPT_PREVIEW_CHARS:
        transcript = transcript[:TRANSCRIPT_PREVIEW_CHARS] + "\n…(truncated)"
    return f"{status}\n\n{transcript}"


def _trim_history(messages: List[Dict[str, Any]], keep_user_turns: int) -> List[Dict[str, Any]]:
    """
    Keep only the last `keep_user_turns` user messages and everything after them.

    The cut is only ever made at a message the *user typed* (string content),
    never at a tool_result, so a tool_use is never separated from its result —
    the API rejects a history where that happens.
    """
    typed = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str)]
    if len(typed) <= keep_user_turns:
        return messages
    return messages[typed[-keep_user_turns]:]
