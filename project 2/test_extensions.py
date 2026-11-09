"""
Tests for the observer and the robustness changes:

    * parse_verdict is generous about format and strict about types
    * a failing agent no longer loses the conversation
    * agents can end a conversation, but not before everyone has spoken
    * transcripts and personas are never silently overwritten
    * persona lookup reports ambiguity instead of guessing
"""

from types import SimpleNamespace

import pytest

from personaforge import (
    END_MARKER,
    Conversation,
    Orchestrator,
    PersonaSpec,
    Turn,
    agent_from_spec,
    match_personas,
    observe,
    parse_verdict,
    run_conversation,
    save_persona,
    save_transcript,
)
from personaforge.llm import DemoLLM, ScriptedLLM, _tool_reply_from_blocks


def spec(name, role="x"):
    return PersonaSpec(name=name, role=role, summary="s", body=f"{name} body")


# -- observer ----------------------------------------------------------------------

GOOD = '{"summary": "They agreed.", "goal_met": true, "in_character": false, "scores": {"relevance": 5, "consistency": 2, "resolution": 4}, "issues": ["Bob said he is an AI"]}'


def test_parse_clean_json():
    v = parse_verdict(GOOD)
    assert v.parsed and v.goal_met is True and v.in_character is False
    assert v.scores == {"relevance": 5, "consistency": 2, "resolution": 4}
    assert v.overall() == 3.67
    assert v.issues == ["Bob said he is an AI"]


def test_parse_tolerates_fences_and_prose():
    v = parse_verdict(f"Here is my verdict:\n```json\n{GOOD}\n```\nHope that helps.")
    assert v.parsed and v.summary == "They agreed."


def test_parse_clamps_and_coerces_scores():
    v = parse_verdict('{"summary": "s", "scores": {"relevance": 9, "consistency": "3", "resolution": 0}}')
    assert v.scores == {"relevance": 5, "consistency": 3, "resolution": 1}


def test_parse_leaves_missing_fields_unknown():
    v = parse_verdict('{"summary": "s", "goal_met": "maybe"}')
    assert v.goal_met is None and v.in_character is None and v.scores == {}


def test_parse_garbage_never_raises():
    v = parse_verdict("I'd rather not grade this.")
    assert v.parsed is False and "rather not" in v.summary
    assert "not valid JSON" in v.to_markdown()


def test_observe_sends_specs_and_transcript():
    llm = ScriptedLLM(replies=[GOOD])
    convo = Conversation(topic="TOPIC_MARK", participants=["Alice"], turns=[Turn("Alice", "LINE_MARK")])
    observe(convo, [spec("Alice")], llm)

    sent = llm.calls[0]["messages"][0]["content"]
    assert "TOPIC_MARK" in sent and "LINE_MARK" in sent and "Alice body" in sent


def test_observe_empty_conversation_skips_the_call():
    llm = ScriptedLLM()
    v = observe(Conversation(topic="x", participants=[]), [], llm)
    assert llm.calls == [] and v.parsed is False


def test_offline_review_works_through_regex_router(tmp_path):
    save_persona(spec("Alice Adams", "doctor"), tmp_path)
    save_persona(spec("Bob Brown", "patient"), tmp_path)
    orch = Orchestrator(DemoLLM(), tmp_path)
    orch.handle("have them talk about the results")
    result = orch.handle("review the conversation")

    assert result.verdict is not None and result.verdict.parsed


# -- conversation robustness --------------------------------------------------------


class FlakyLLM(ScriptedLLM):
    """Answers normally, then raises on the Nth call."""

    def __init__(self, fail_on):
        super().__init__(replies=["fine"])
        self.fail_on = fail_on

    def complete(self, system, messages, max_tokens=2048):
        if len(self.calls) + 1 == self.fail_on:
            self.calls.append({})
            raise RuntimeError("network blip")
        return super().complete(system, messages, max_tokens)


def test_failed_turn_keeps_partial_transcript():
    llm = FlakyLLM(fail_on=3)
    a, b = agent_from_spec(spec("Alice"), llm), agent_from_spec(spec("Bob"), llm)
    convo = run_conversation([a, b], topic="x", turns=6)

    assert len(convo.turns) == 2
    assert "Alice" in convo.error and "network blip" in convo.error
    assert "Stopped early" in convo.transcript()


def test_agent_can_end_the_conversation():
    llm = ScriptedLLM(replies=["Hello.", f"Goodbye. {END_MARKER}"])
    a, b = agent_from_spec(spec("Alice"), llm), agent_from_spec(spec("Bob"), llm)
    convo = run_conversation([a, b], topic="x", turns=10)

    assert len(convo.turns) == 2
    assert convo.ended_by == "Bob"
    assert END_MARKER not in convo.turns[-1].text


def test_end_is_ignored_until_everyone_has_spoken():
    llm = ScriptedLLM(replies=[f"Done already. {END_MARKER}", "Wait, I haven't spoken!"])
    a, b = agent_from_spec(spec("Alice"), llm), agent_from_spec(spec("Bob"), llm)
    convo = run_conversation([a, b], topic="x", turns=3)

    assert len(convo.turns) == 3 and convo.ended_by == "Alice"   # honoured on turn 3


def test_system_prompt_mentions_end_marker():
    assert END_MARKER in agent_from_spec(spec("Alice"), ScriptedLLM()).system_prompt()


def test_window_caps_what_is_resent():
    llm = ScriptedLLM()
    agent = agent_from_spec(spec("Alice"), llm)
    transcript = [Turn("Scene", "SCENE")] + [Turn("Bob", f"line {i}") for i in range(50)]
    agent.respond(transcript, window=5)

    sent = llm.calls[0]["messages"]
    assert len(sent) == 6 and "SCENE" in sent[0]["content"] and "line 49" in sent[-1]["content"]


def test_zero_turns_is_rejected():
    a, b = agent_from_spec(spec("A"), ScriptedLLM()), agent_from_spec(spec("B"), ScriptedLLM())
    with pytest.raises(ValueError):
        run_conversation([a, b], topic="x", turns=0)


# -- no silent overwrites --------------------------------------------------------------


def test_transcripts_get_unique_names(tmp_path):
    convo = Conversation(topic="Same Topic", participants=["A"], turns=[Turn("A", "hi")])
    paths = {save_transcript(convo, tmp_path) for _ in range(3)}
    assert len(paths) == 3 and all("same-topic" in p.name for p in paths)


def test_saving_a_duplicate_persona_keeps_both(tmp_path):
    first = save_persona(spec("Maria Delgado"), tmp_path)
    second = save_persona(spec("Maria Delgado"), tmp_path)
    assert first.name == "maria-delgado.md" and second.name == "maria-delgado-2.md"


def test_overwrite_flag_replaces(tmp_path):
    save_persona(spec("Maria Delgado"), tmp_path)
    save_persona(spec("Maria Delgado"), tmp_path, overwrite=True)
    assert len(list(tmp_path.glob("*.md"))) == 1


# -- persona matching ------------------------------------------------------------------


def test_match_prefers_exact_word_over_substring(tmp_path):
    save_persona(spec("Dorothy Vance", "patient"), tmp_path)
    save_persona(spec("Sam Doc", "doctor"), tmp_path)
    assert [s.name for s in match_personas("doc", tmp_path)] == ["Sam Doc"]


def test_match_returns_all_ambiguous_candidates(tmp_path):
    save_persona(spec("Ann Lee", "doctor"), tmp_path)
    save_persona(spec("Raj Rao", "doctor"), tmp_path)
    assert len(match_personas("doctor", tmp_path)) == 2


def test_match_blank_is_empty(tmp_path):
    save_persona(spec("Ann Lee"), tmp_path)
    assert match_personas("  ", tmp_path) == []


# -- SDK block conversion ----------------------------------------------------------------


def test_sdk_blocks_become_a_tool_reply():
    blocks = [
        SimpleNamespace(type="text", text="Let me do that."),
        SimpleNamespace(type="tool_use", id="toolu_1", name="list_personas", input={}),
        SimpleNamespace(type="thinking", thinking="ignored"),
    ]
    reply = _tool_reply_from_blocks(blocks)

    assert reply.text == "Let me do that." and reply.wants_tools
    assert reply.tool_calls[0].name == "list_personas"
    assert [b["type"] for b in reply.content] == ["text", "tool_use"]


def test_claude_llm_counts_calls_and_tokens(monkeypatch):
    """Usage totals feed the cost section of the report; check them without a network."""
    from personaforge import ClaudeLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )
    llm = ClaudeLLM(model="m")
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    llm.complete("s", [{"role": "user", "content": "x"}])
    llm.complete("s", [{"role": "user", "content": "y"}])

    assert llm.usage() == {"calls": 2, "input_tokens": 20, "output_tokens": 6}
