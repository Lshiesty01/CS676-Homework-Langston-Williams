"""
llm.py — the thin layer between a persona and a model.

Everything in this package that needs to "think" goes through the `LLM`
protocol below. There are two families of implementation:

    ClaudeLLM        real Claude calls. Needs ANTHROPIC_API_KEY.
    ScriptedLLM      returns canned strings. No key, no network, no cost.
    ScriptedToolLLM  returns canned *tool-use* turns, for testing the
                     orchestrator's agent loop without a key.

That split is deliberate and it is the reason this project is testable. The
unit tests in `tests/` inject a scripted model, so they run in milliseconds for
free, and `main.py --offline` uses one too so you can see the whole app work
before you have configured anything.

Two protocols live here:

    LLM       complete(system, messages) -> str
    ToolLLM   LLM + complete_with_tools(system, messages, tools) -> ToolReply

Persona agents only ever need `LLM`. The orchestrator uses `ToolLLM` when the
model offers it and falls back to its regex router when it does not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

# Claude Opus 5 is the most capable model. While you are iterating you may want
# "claude-haiku-4-5" — a persona conversation makes one call per turn, and those
# add up. Say which model produced your submitted transcripts.
DEFAULT_MODEL = "claude-opus-5"


class LLM(Protocol):
    """Anything that can turn a system prompt plus a transcript into a reply."""

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        ...


class LLMError(RuntimeError):
    """
    A model call failed in a way the caller should report, not crash on.

    Raised for network errors, rate limits, and bad credentials. The message is
    written for the person at the keyboard: it says what went wrong and what to
    try next.
    """


# -----------------------------------------------------------------------------
# Tool use
# -----------------------------------------------------------------------------
# The orchestrator hands the model a set of tools and lets it choose. That needs
# a richer reply than a string: the model may return some text, some tool
# calls, or both. These two small types are the package's own representation of
# that reply, so nothing above this file depends on the anthropic SDK's classes
# — and so a scripted fake can produce exactly the same shapes in tests.


@dataclass
class ToolCall:
    """One request from the model to run a tool."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolReply:
    """
    One model turn in a tool-use loop.

    :param text:        any prose the model wrote this turn
    :param tool_calls:  tools it wants run, in order (empty when it is done)
    :param content:     the assistant content blocks as plain dicts, ready to
                        append to the message history unchanged. The API needs
                        the assistant turn echoed back exactly before the
                        matching tool_result blocks.
    """

    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    content: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ToolLLM(LLM, Protocol):
    """An LLM that can also take part in a tool-use loop."""

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolReply:
        ...


def supports_tools(llm: Any) -> bool:
    """True if `llm` can drive the tool-use orchestrator."""
    return callable(getattr(llm, "complete_with_tools", None))


# =============================================================================
# ⚠️  NEEDS YOUR OWN API KEY
# =============================================================================
# ClaudeLLM is every real model call in this package. Without ANTHROPIC_API_KEY
# it raises with instructions rather than failing quietly. Calls are billed to
# you: a conversation costs one call per turn, plus one per persona created,
# plus one or more orchestrator calls per thing you type.
#
# You do not need a key to work on this project: `uv run pytest` and
# `uv run main.py --offline` exercise the whole package for free.
# =============================================================================


@dataclass
class ClaudeLLM:
    """
    Real Claude.

    `complete()` is all a persona needs: "here is who you are, here is the
    conversation so far, what do you say next". `complete_with_tools()` is what
    the orchestrator uses to let the model choose an action.

    The client is created lazily so importing this module never requires a key.
    """

    model: str = DEFAULT_MODEL
    _client: Any = field(default=None, repr=False)
    # Running totals, so a session can report what it cost (see run_live_session.py).
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def usage(self) -> Dict[str, int]:
        """Calls and tokens used by this instance so far."""
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "or run in offline mode: uv run main.py --offline"
                )
            # The SDK retries rate limits and 5xx errors itself; two is enough
            # to ride out a blip without making a dead network feel frozen.
            self._client = anthropic.Anthropic(max_retries=2)
        return self._client

    def _create(self, **kwargs: Any) -> Any:
        """Make one API call, translating SDK exceptions into `LLMError`."""
        import anthropic

        try:
            response = self._get_client().messages.create(model=self.model, **kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                f"Your API key was rejected ({exc.message}). Check ANTHROPIC_API_KEY in .env, "
                "and that no older ANTHROPIC_API_KEY is set in your system environment."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the API. Wait a minute, or use --model claude-haiku-4-5.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model '{self.model}' was not found. Try --model claude-haiku-4-5.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the API. Check your connection, or use --offline.") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"The API returned an error: {exc}") from exc

        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return response

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Send one request and return the concatenated text of the reply."""
        response = self._create(max_tokens=max_tokens, system=system, messages=messages)

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return "[declined to respond]"

        return "".join(block.text for block in response.content if block.type == "text").strip()

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolReply:
        """Send one request with tools attached and return a `ToolReply`."""
        response = self._create(max_tokens=max_tokens, system=system, messages=messages, tools=tools)
        return _tool_reply_from_blocks(response.content)


def _tool_reply_from_blocks(blocks: List[Any]) -> ToolReply:
    """
    Convert SDK content blocks into a `ToolReply`.

    Only text and tool_use blocks are kept. Anything else (thinking blocks, for
    instance) is dropped from the echoed history — the orchestrator does not
    enable those features, so dropping is safe and keeps the history small.
    """
    text_parts: List[str] = []
    calls: List[ToolCall] = []
    content: List[Dict[str, Any]] = []
    for block in blocks:
        if block.type == "text":
            text_parts.append(block.text)
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))
            content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input or {})})
    return ToolReply(text="".join(text_parts).strip(), tool_calls=calls, content=content)


# =============================================================================
# Scripted models — no key, no network, no cost
# =============================================================================


@dataclass
class ScriptedLLM:
    """
    A fake model that replays a fixed list of replies, then loops.

    Used by the tests and by `--offline`. It records every call it received in
    `.calls`, which makes it easy to assert that a persona's markdown actually
    reached the system prompt.
    """

    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


# What a scripted tool turn may be written as, for readable tests:
#   "plain text"                                  -> a final answer
#   ("create_persona", {"description": "nurse"})  -> one tool call
#   [("list_personas", {}), ("create_persona", {...})]  -> several calls
ScriptedTurn = Union[str, tuple, list]


@dataclass
class ScriptedToolLLM:
    """
    A fake tool-using model for testing the orchestrator's agent loop.

    `tool_turns` is consumed in order by `complete_with_tools`. When it runs out,
    the model answers "(done)". Plain `complete()` calls — from persona
    authoring and persona dialogue — are served from `replies`, exactly like
    `ScriptedLLM`, so one fake can drive a whole session.
    """

    tool_turns: List[ScriptedTurn] = field(default_factory=list)
    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_seen: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolReply:
        # Deep-ish copy: the orchestrator mutates its history list afterwards.
        self.tool_calls_seen.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        turn: ScriptedTurn = self.tool_turns.pop(0) if self.tool_turns else "(done)"

        if isinstance(turn, str):
            return ToolReply(text=turn, content=[{"type": "text", "text": turn}])

        pairs = [turn] if isinstance(turn, tuple) else list(turn)
        calls, content = [], []
        for i, (name, args) in enumerate(pairs):
            call_id = f"toolu_{len(self.tool_calls_seen)}_{i}"
            calls.append(ToolCall(id=call_id, name=name, input=dict(args)))
            content.append({"type": "tool_use", "id": call_id, "name": name, "input": dict(args)})
        return ToolReply(text="", tool_calls=calls, content=content)


_DEMO_DIALOGUE = [
    "I understand. Could you tell me more about when the symptoms started?",
    "That has been going on for about three weeks now, mostly in the mornings.",
    "Have you noticed anything that makes it better or worse?",
    "Coffee seems to make it worse, and lying down helps a little.",
    "Thank you, that is useful. I would like to run a couple of tests.",
    "Whatever you think is best, doctor. I just want to feel normal again.",
]

_DEMO_PERSONA = """---
name: {name}
role: {role}
summary: a demo persona generated in offline mode
---

# {name}

## Background
This persona was produced by the offline stub, not by a model. Run without
`--offline` to get a real character written by Claude.

## Personality
- Placeholder
- Placeholder

## How they speak
- In whatever the scripted stub happens to return
"""

_DEMO_VERDICT = """{"summary": "Offline demo: the scripted dialogue covered symptoms and next steps.",
"goal_met": true, "in_character": true,
"scores": {"relevance": 4, "consistency": 4, "resolution": 3},
"issues": ["offline mode — these scores are canned, not a real judgment"]}"""


@dataclass
class DemoLLM:
    """
    The offline stub used by `--offline`.

    A single ScriptedLLM cannot serve every job: asked to author a persona it
    would return a line of dialogue, and you would get a character named
    "Unnamed". So this one looks at the system prompt and answers in the shape
    the caller expects — persona markdown for the author prompt, a JSON verdict
    for the observer, dialogue for everything else.

    It deliberately does NOT implement `complete_with_tools`, so offline mode
    exercises the orchestrator's regex fallback.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        request = messages[-1]["content"] if messages else ""

        # The persona-author prompt is the only one that asks for frontmatter.
        if "role: <one lowercase word" in system:
            description = request.replace("Write a persona for:", "").strip()
            words = [w for w in description.split() if w.isalpha()]
            role = words[0].lower() if words else "person"
            return _DEMO_PERSONA.format(name=description[:40].title() or "Demo Persona", role=role)

        if "impartial observer" in system.lower():
            return _DEMO_VERDICT

        if "stage manager" in system.lower():
            return "Offline mode — I'm a stub. Try: create a persona doctor"

        return _DEMO_DIALOGUE[(len(self.calls) - 1) % len(_DEMO_DIALOGUE)]


def default_llm(offline: bool = False, model: Optional[str] = None) -> LLM:
    """Pick an implementation. Central so you only change this in one place."""
    if offline:
        return DemoLLM()
    return ClaudeLLM(model=model or DEFAULT_MODEL)
