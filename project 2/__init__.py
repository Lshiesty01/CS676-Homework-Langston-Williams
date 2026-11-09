"""
PersonaForge — an agent-to-agent persona simulation package.

The idea in four lines:

    spec  = generate_persona("a patient with chronic back pain", llm)
    path  = save_persona(spec)              # writes temp/<name>.md
    agent = agent_from_spec(spec, llm)      # markdown + model = a mind
    convo = run_conversation([doctor, patient], topic="the test results")

And two more for the parts added in this version:

    verdict = observe(convo, [doctor_spec, patient_spec], llm)   # an AI judge
    Orchestrator(ClaudeLLM()).handle("make two nurses and have them argue")

A persona is a markdown file. An agent is that file plus a model. A conversation
is agents taking turns. The orchestrator is Claude choosing among tools.

Versioning follows semver: 0.2.0 adds tool use and the observer; the 0.1 API
still works, with one behaviour change — transcripts are saved under
`<persona_dir>/conversations/` with a unique name, not as `conversation.md`.
"""

from .agent import END_MARKER, PersonaAgent, agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import (
    LLM,
    ClaudeLLM,
    LLMError,
    ScriptedLLM,
    ScriptedToolLLM,
    ToolCall,
    ToolLLM,
    ToolReply,
    default_llm,
    supports_tools,
)
from .observer import Verdict, observe, parse_verdict
from .orchestrator import TOOLS, Orchestrator, Result
from .persona import (
    PersonaSpec,
    find_persona,
    generate_persona,
    list_personas,
    load_persona,
    match_personas,
    save_persona,
)

__version__ = "0.2.0"

__all__ = [
    # personas
    "PersonaSpec",
    "find_persona",
    "generate_persona",
    "list_personas",
    "load_persona",
    "match_personas",
    "save_persona",
    # agents and conversations
    "END_MARKER",
    "PersonaAgent",
    "agent_from_spec",
    "Conversation",
    "Turn",
    "run_conversation",
    "save_transcript",
    # models
    "LLM",
    "ToolLLM",
    "ClaudeLLM",
    "LLMError",
    "ScriptedLLM",
    "ScriptedToolLLM",
    "ToolCall",
    "ToolReply",
    "default_llm",
    "supports_tools",
    # observer
    "Verdict",
    "observe",
    "parse_verdict",
    # orchestrator
    "TOOLS",
    "Orchestrator",
    "Result",
]
