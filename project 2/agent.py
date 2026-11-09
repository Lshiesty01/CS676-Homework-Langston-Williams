"""
agent.py — markdown file + model = an agent with its own mind.

This is the smallest interesting idea in the package. A `PersonaAgent` holds a
`PersonaSpec` (which came from a `.md` file) and an `LLM`. When you ask it to
speak, it builds a system prompt out of that markdown and sends the transcript
so far. The persona file *is* the agent's brain — nothing else distinguishes
the doctor from the patient.

Because the transcript is passed in rather than stored, agents are stateless
between turns. That is a design decision with consequences; see YOUR TASK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from .llm import LLM
from .persona import PersonaSpec

if TYPE_CHECKING:  # imported for type hints only; conversation.py imports this module
    from .conversation import Turn

# An agent appends this to a line to say the conversation has reached its
# natural end. `run_conversation` strips it and stops the loop.
END_MARKER = "[END]"

# How many recent lines an agent sees. Every turn re-sends the transcript, so
# cost grows quadratically with length; capping the window keeps a long
# conversation affordable. The opening scene is always kept.
DEFAULT_WINDOW = 20

# Wrapped around the persona markdown to put the model in character and keep it
# from narrating. Without the last two rules models tend to write stage
# directions and answer for everyone in the room.
_IN_CHARACTER_TEMPLATE = """You are role-playing a character in a simulated conversation.

Everything between the markers below defines who you are. Inhabit it completely:
speak in this person's voice, hold their opinions, want what they want.

=== BEGIN CHARACTER ===
{persona}
=== END CHARACTER ===

Rules:
- Speak only as {name}. Never speak for anyone else.
- Reply with dialogue only. No stage directions, no asterisks, no narration.
- Stay in character even if the conversation goes somewhere unexpected.
- Keep replies to a few sentences unless the moment genuinely calls for more.
- You are {name}. Do not mention that you are an AI or a simulation.
- If the conversation has genuinely reached its natural end — the question is
  answered, a decision is made, or someone is leaving — finish your line with
  {end_marker}. Do not use it just because you have little to add."""


@dataclass
class PersonaAgent:
    """One character in a conversation."""

    spec: PersonaSpec
    llm: LLM

    @property
    def name(self) -> str:
        return self.spec.name

    def system_prompt(self) -> str:
        """The persona markdown, wrapped in the in-character instructions."""
        return _IN_CHARACTER_TEMPLATE.format(
            persona=self.spec.body, name=self.spec.name, end_marker=END_MARKER
        )

    def respond(
        self,
        transcript: List["Turn"],
        max_tokens: int = 1024,
        window: int = DEFAULT_WINDOW,
    ) -> str:
        """
        Produce this character's next line.

        The transcript is rewritten from this agent's point of view: its own past
        lines become `assistant` turns and everyone else's become `user` turns.
        That is what makes the model feel like a participant rather than an
        observer reading a script.

        Only the first line (the scene) and the last `window` lines are sent.
        """
        if len(transcript) > window + 1:
            transcript = [transcript[0]] + list(transcript[-window:])

        messages: List[dict] = []
        for turn in transcript:
            if turn.speaker == self.name:
                messages.append({"role": "assistant", "content": turn.text})
            else:
                messages.append({"role": "user", "content": f"{turn.speaker}: {turn.text}"})

        # The API requires the first message to be from the user, and consecutive
        # same-role turns get merged, so an empty or assistant-first transcript
        # needs an opener.
        if not messages or messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "(begin the conversation)"})

        return self.llm.complete(self.system_prompt(), messages, max_tokens=max_tokens)


def agent_from_spec(spec: PersonaSpec, llm: LLM) -> PersonaAgent:
    """Convenience constructor, so callers don't import the dataclass directly."""
    return PersonaAgent(spec=spec, llm=llm)


# =============================================================================
# STATUS OF THE ORIGINAL TASK LIST
# =============================================================================
#
#  DONE  An agent can end the conversation (END_MARKER, honoured by the loop).
#  DONE  Transcript re-sending is capped at DEFAULT_WINDOW lines, so cost stops
#        growing quadratically on long conversations.
#  OPEN  Memory across conversations, private vs. public knowledge, and an
#        in-loop character-break check. The observer (observer.py) detects
#        character breaks after the fact instead; see REPORT.md for why.
