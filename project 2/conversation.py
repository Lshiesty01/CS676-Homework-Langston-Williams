"""
conversation.py — put two or more agents in a room and let them talk.

Agents take turns in the order given. Two things stop a conversation:

    * the turn budget runs out, or
    * an agent signals the conversation is finished by ending its line with
      END_MARKER, which the in-character prompt tells it about. The marker is
      stripped before anyone sees the line.

A failed model call no longer throws away everything said so far: the loop
stops, records the error on the `Conversation`, and returns what it has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .agent import END_MARKER, PersonaAgent

# Transcripts live in their own subfolder. The persona directory is globbed for
# `*.md` to find personas, so a transcript saved next to them used to be loaded
# back as a persona called "Unnamed".
TRANSCRIPT_SUBDIR = "conversations"


@dataclass
class Turn:
    """One line of dialogue."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A completed (or interrupted) dialogue, plus the setup that produced it."""

    topic: str
    participants: List[str]
    turns: List[Turn] = field(default_factory=list)
    ended_by: Optional[str] = None   # the agent that closed it early, if any
    error: Optional[str] = None      # set when a model call failed mid-conversation

    def transcript(self) -> str:
        """Render as markdown, suitable for saving or printing."""
        header = f"# Conversation: {self.topic}\n\nParticipants: {', '.join(self.participants)}\n\n"
        body = "\n\n".join(f"**{t.speaker}:** {t.text}" for t in self.turns)
        footer = ""
        if self.ended_by:
            footer += f"\n\n_{self.ended_by} ended the conversation._"
        if self.error:
            footer += f"\n\n_Stopped early: {self.error}_"
        return header + body + footer


def _strip_end_marker(text: str) -> Tuple[str, bool]:
    """Remove END_MARKER from a line and report whether it was there."""
    if END_MARKER not in text:
        return text, False
    return text.replace(END_MARKER, "").strip(), True


def run_conversation(
    agents: List[PersonaAgent],
    topic: str,
    turns: int = 6,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Conversation:
    """
    Run a turn-taking conversation between the given agents.

    :param agents:  two or more PersonaAgents. They speak in list order.
    :param topic:   seeds the opening; each agent sees it as the situation.
    :param turns:   the maximum number of lines of dialogue. Agents may end it
                    sooner, but only once everyone has spoken at least once.
    :param on_turn: called after each line, so a UI can print as it goes rather
                    than waiting for the whole conversation to finish.
    """
    if len(agents) < 2:
        raise ValueError("a conversation needs at least two agents")
    if turns < 1:
        raise ValueError("turns must be at least 1")

    conversation = Conversation(topic=topic, participants=[a.name for a in agents])

    # The topic is injected as a scene-setting turn from a narrator. Every agent
    # sees it as context but nobody has to answer it directly.
    history: List[Turn] = [Turn(speaker="Scene", text=topic)]

    for i in range(turns):
        speaker = agents[i % len(agents)]
        try:
            raw = speaker.respond(history)
        except Exception as exc:  # noqa: BLE001 — keep the partial transcript
            conversation.error = f"{speaker.name}'s turn failed ({type(exc).__name__}: {exc})"
            break

        text, wants_to_end = _strip_end_marker(raw)
        # An agent that replies with only the marker still took its turn.
        turn = Turn(speaker=speaker.name, text=text or "…")
        history.append(turn)
        conversation.turns.append(turn)

        if on_turn is not None:
            on_turn(turn)

        # Ending is honoured only after every participant has had a say, so a
        # terse first speaker cannot close the scene before it starts.
        if wants_to_end and i + 1 >= len(agents):
            conversation.ended_by = speaker.name
            break

    return conversation


def _slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "conversation"


def save_transcript(conversation: Conversation, directory, filename: Optional[str] = None) -> Path:
    """
    Write the transcript to `directory` and return the path.

    Without an explicit `filename`, the name is a timestamp plus the topic, e.g.
    `20261020-141502-the-mri-results.md`, and a numeric suffix is added if that
    file already exists — so each conversation keeps its own file instead of
    overwriting the last one. An explicit `filename` is written as given.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        path = directory / filename
    else:
        stem = f"{datetime.now():%Y%m%d-%H%M%S}-{_slugify(conversation.topic)}"
        path = directory / f"{stem}.md"
        counter = 2
        while path.exists():
            path = directory / f"{stem}-{counter}.md"
            counter += 1

    path.write_text(conversation.transcript(), encoding="utf-8")
    return path
