"""
Tests for the tool-use orchestrator.

`ScriptedToolLLM` plays the part of Claude: each entry in `tool_turns` is one
model turn (a tool call, several tool calls, or a final text answer). That lets
us test the agent loop — dispatch, tool_result plumbing, error reporting,
history — without a key or a network.
"""

from personaforge import Orchestrator, PersonaSpec, ScriptedToolLLM, save_persona
from personaforge.llm import LLMError, ScriptedLLM
from personaforge.orchestrator import MAX_PERSONAS_PER_MESSAGE, MAX_TOOL_ROUNDS, TOOLS, _trim_history

PERSONA_MD = """---
name: {name}
role: {role}
summary: a summary
---

# {name}
Body text.
"""


def seed(tmp_path, *people):
    for name, role in people:
        save_persona(PersonaSpec(name=name, role=role, summary="s", body="b"), tmp_path)


def tool_results(llm, call_index):
    """The tool_result blocks sent back to the model on its `call_index`-th call."""
    return llm.tool_calls_seen[call_index]["messages"][-1]["content"]


# -- routing mode ---------------------------------------------------------------


def test_tool_capable_llm_uses_tool_router(tmp_path):
    assert Orchestrator(ScriptedToolLLM(), tmp_path).use_tools is True


def test_plain_llm_falls_back_to_regex(tmp_path):
    assert Orchestrator(ScriptedLLM(), tmp_path).use_tools is False


def test_every_tool_schema_is_well_formed():
    for tool in TOOLS:
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"
        for required in tool["input_schema"].get("required", []):
            assert required in tool["input_schema"]["properties"]


# -- the loop -------------------------------------------------------------------


def test_plain_answer_needs_no_tools(tmp_path):
    llm = ScriptedToolLLM(tool_turns=["What kind of person should I create?"])
    result = Orchestrator(llm, tmp_path).handle("create a persona")

    assert result.reply == "What kind of person should I create?"   # a clarifying question
    assert len(llm.tool_calls_seen) == 1


def test_create_persona_tool_writes_file_and_reports_back(tmp_path):
    llm = ScriptedToolLLM(
        tool_turns=[("create_persona", {"description": "a skeptical data scientist named Ada"}), "Made Ada."],
        replies=[PERSONA_MD.format(name="Ada Okafor", role="scientist")],
    )
    result = Orchestrator(llm, tmp_path).handle("create a skeptical data scientist named Ada")

    assert (tmp_path / "ada-okafor.md").exists()
    assert result.reply == "Made Ada."
    sent = tool_results(llm, 1)[0]
    assert sent["type"] == "tool_result" and "Ada Okafor" in sent["content"]
    assert "is_error" not in sent


def test_several_tools_in_one_turn(tmp_path):
    """'Two doctors and a nurse' is three create calls the regex router could never express."""
    llm = ScriptedToolLLM(
        tool_turns=[
            [
                ("create_persona", {"description": "doctor one"}),
                ("create_persona", {"description": "doctor two"}),
                ("create_persona", {"description": "a nurse"}),
            ],
            "Three created.",
        ],
        replies=[
            PERSONA_MD.format(name="Ann Lee", role="doctor"),
            PERSONA_MD.format(name="Raj Rao", role="doctor"),
            PERSONA_MD.format(name="Kim Fox", role="nurse"),
        ],
    )
    result = Orchestrator(llm, tmp_path).handle("make two doctors and a nurse")

    assert len(list(tmp_path.glob("*.md"))) == 3
    assert len(tool_results(llm, 1)) == 3
    assert len(result.events) == 3


def test_create_then_converse_across_rounds(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"))
    llm = ScriptedToolLLM(
        tool_turns=[
            ("create_persona", {"description": "a patient"}),
            ("start_conversation", {"participants": ["Alice", "Bob"], "topic": "the scan", "turns": 4}),
            "They talked.",
        ],
        replies=[PERSONA_MD.format(name="Bob Brown", role="patient"), "a line"],
    )
    seen = []
    result = Orchestrator(llm, tmp_path).handle("add a patient and have them discuss the scan", on_turn=seen.append)

    assert result.conversation is not None
    assert result.conversation.participants == ["Alice Adams", "Bob Brown"]
    assert len(result.conversation.turns) == 4 and len(seen) == 4
    assert "the scan" in tool_results(llm, 2)[0]["content"]


def test_turns_are_clamped(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    llm = ScriptedToolLLM(
        tool_turns=[("start_conversation", {"participants": ["Alice", "Bob"], "topic": "x", "turns": 500}), "ok"],
        replies=["line"],
    )
    result = Orchestrator(llm, tmp_path).handle("talk forever")
    assert len(result.conversation.turns) == 20


def test_transcript_goes_to_conversations_subfolder(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    llm = ScriptedToolLLM(
        tool_turns=[("start_conversation", {"participants": ["Alice", "Bob"], "topic": "x", "turns": 2}), "ok"],
    )
    Orchestrator(llm, tmp_path).handle("talk")

    assert len(list((tmp_path / "conversations").glob("*.md"))) == 1
    assert len(list(tmp_path.glob("*.md"))) == 2   # only the two personas


# -- errors go back to the model, not to the user as a crash --------------------


def test_unknown_participant_is_a_tool_error(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    llm = ScriptedToolLLM(
        tool_turns=[("start_conversation", {"participants": ["Zaphod", "Bob"], "topic": "x"}), "Who is Zaphod?"],
    )
    result = Orchestrator(llm, tmp_path).handle("have Zaphod and Bob talk")

    sent = tool_results(llm, 1)[0]
    assert sent["is_error"] is True and "Zaphod" in sent["content"]
    assert result.reply == "Who is Zaphod?" and result.conversation is None


def test_ambiguous_name_lists_candidates(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Raj Rao", "doctor"), ("Kim Fox", "nurse"))
    llm = ScriptedToolLLM(
        tool_turns=[("start_conversation", {"participants": ["doctor", "Kim"], "topic": "x"}), "Which doctor?"],
    )
    Orchestrator(llm, tmp_path).handle("have the doctor and Kim talk")

    sent = tool_results(llm, 1)[0]
    assert sent["is_error"] is True
    assert "Ann Lee" in sent["content"] and "Raj Rao" in sent["content"]


def test_empty_description_is_a_tool_error(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[("create_persona", {"description": "  "}), "Who should I create?"])
    Orchestrator(llm, tmp_path).handle("create one")

    assert tool_results(llm, 1)[0]["is_error"] is True
    assert not list(tmp_path.glob("*.md"))


def test_unknown_tool_is_a_tool_error(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[("delete_everything", {}), "Sorry."])
    Orchestrator(llm, tmp_path).handle("x")
    assert tool_results(llm, 1)[0]["is_error"] is True


def test_persona_budget_per_message(tmp_path):
    many = [("create_persona", {"description": f"person {i}"}) for i in range(MAX_PERSONAS_PER_MESSAGE + 2)]
    llm = ScriptedToolLLM(tool_turns=[many, "ok"], replies=[PERSONA_MD.format(name="P", role="x")])
    Orchestrator(llm, tmp_path).handle("make a crowd")

    errors = [b for b in tool_results(llm, 1) if b.get("is_error")]
    assert len(errors) == 2


def test_runaway_loop_stops(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[("list_personas", {})] * (MAX_TOOL_ROUNDS + 5))
    result = Orchestrator(llm, tmp_path).handle("loop")

    assert len(llm.tool_calls_seen) == MAX_TOOL_ROUNDS
    assert "stopped" in result.reply.lower()


def test_api_failure_is_reported_and_history_untouched(tmp_path):
    class FailingLLM(ScriptedToolLLM):
        def complete_with_tools(self, *args, **kwargs):
            raise LLMError("Could not reach the API.")

    orch = Orchestrator(FailingLLM(), tmp_path)
    result = orch.handle("hello")

    assert "Could not reach the API" in result.reply
    assert orch.history == []


# -- memory within a session ------------------------------------------------------


def test_history_carries_into_the_next_message(tmp_path):
    llm = ScriptedToolLLM(tool_turns=["first answer", "second answer"])
    orch = Orchestrator(llm, tmp_path)
    orch.handle("first question")
    orch.handle("second question")

    second_call = llm.tool_calls_seen[1]["messages"]
    assert [m["content"] for m in second_call if isinstance(m["content"], str)][:1] == ["first question"]


def test_roster_is_in_the_system_prompt(tmp_path):
    seed(tmp_path, ("Maria Delgado", "patient"))
    llm = ScriptedToolLLM(tool_turns=["hi"])
    Orchestrator(llm, tmp_path).handle("who's here?")
    assert "Maria Delgado" in llm.tool_calls_seen[0]["system"]


def test_trim_history_never_orphans_a_tool_result():
    history = []
    for i in range(5):
        history += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "r"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ]
    trimmed = _trim_history(history, keep_user_turns=2)

    assert trimmed[0] == {"role": "user", "content": "q3"}
    assert len(trimmed) == 8


# -- observer through the orchestrator --------------------------------------------


def test_review_tool_judges_last_conversation(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    verdict_json = '{"summary": "fine", "goal_met": true, "in_character": true, "scores": {"relevance": 5, "consistency": 4, "resolution": 3}, "issues": []}'
    llm = ScriptedToolLLM(
        tool_turns=[
            ("start_conversation", {"participants": ["Alice", "Bob"], "topic": "x", "turns": 2}),
            ("review_conversation", {}),
            "Reviewed.",
        ],
        replies=["line", "line", verdict_json],
    )
    result = Orchestrator(llm, tmp_path).handle("talk then judge it")

    assert result.verdict is not None
    assert result.verdict.scores == {"relevance": 5, "consistency": 4, "resolution": 3}


def test_review_without_a_conversation_is_a_tool_error(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[("review_conversation", {}), "Nothing to review."])
    Orchestrator(llm, tmp_path).handle("judge it")
    assert tool_results(llm, 1)[0]["is_error"] is True
