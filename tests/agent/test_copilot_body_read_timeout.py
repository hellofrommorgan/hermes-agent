"""Capture real SDK requests through Hermes's conversation retry path."""

import copy
import json
from pathlib import Path

import httpx
import pytest

from run_agent import AIAgent
from tools.tool_result_storage import extract_persisted_path


BODY_ERROR = {"error": {"code": "user_request_timeout", "message":
    "Timed out reading request body. Try again, or use a smaller request size."}}
LARGE_TEXT = "evidence line with Unicode: café 🌊\n" * 600
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read saved output",
    "parameters": {"type": "object", "properties": {}}}}]


def _success(output=None):
    response = {"id": "resp_ok", "object": "response", "created_at": 1,
        "model": "gpt-6-astra", "status": "completed", "output": [{"id": "msg_ok", "type": "message",
        "role": "assistant", "status": "completed", "content": [{"type": "output_text",
        "text": "done", "annotations": []}]}],
        "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22}}
    if output is not None:
        response["output"] = output
    events = [{"type": "response.output_item.done", "output_index": i, "item": item}
        for i, item in enumerate(response["output"])] + [{"type": "response.completed", "response": response}]
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
        content="".join("data: " + json.dumps(event) + "\n\n" for event in events))


def _agent(monkeypatch, handler, *, provider="copilot", session_db=None):
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: copy.deepcopy(TOOLS))
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", staticmethod(
        lambda *a, **kw: httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0)
    agent = AIAgent(model="gpt-6-astra", provider=provider, api_mode="codex_responses",
        base_url="https://api.enterprise.githubcopilot.com" if provider == "copilot" else "https://api.openai.com/v1",
        api_key="test-token", quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], max_iterations=4, session_db=session_db, session_id="body-read-session")
    agent.compression_enabled = False
    agent._cached_system_prompt = "Keep this exact system instruction."
    agent._use_prompt_caching = False
    agent._disable_streaming = True  # Responses still uses the real SSE dispatch.
    agent._cleanup_task_resources = lambda *_: None
    agent._save_trajectory = lambda *a, **kw: None
    def forbidden(*a, **kw):
        pytest.fail("body-read recovery must not invoke compaction, fallback or credential rotation")
    agent._compress_context = forbidden
    agent._try_activate_fallback = forbidden
    agent._try_recover_with_credential_pool = forbidden
    return agent


def _history(text=LARGE_TEXT):
    return [{"role": "user", "content": "Keep the original user request."},
        {"role": "assistant", "content": "Inspecting evidence", "tool_calls": [{"id": "call_evidence",
            "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_evidence", "content": text}]


@pytest.mark.parametrize("outcome", ["success", "repeat", "no_text", "write_failure", "retry_500", "retry_connection", "retry_stream", "no_reader", "unavailable_backend"])
def test_exact_body_timeout_reduces_once_or_stops(monkeypatch, tmp_path, outcome):
    captures = []

    def handler(request):
        captures.append(request.content)
        if len(captures) > 1 and outcome == "retry_500":
            return httpx.Response(500, json={"error": {"message": "fixture failure"}})
        if len(captures) > 1 and outcome == "retry_connection":
            raise httpx.ReadError("fixture lost connection")
        if len(captures) > 1 and outcome == "retry_stream":
            class BrokenStream(httpx.SyncByteStream):
                def __iter__(self):
                    yield b'data: {"type":"response.created"}\n\n'
                    raise httpx.ReadError("fixture stream interrupted")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=BrokenStream())
        if len(captures) == 1 or outcome != "success":
            return httpx.Response(408, json=BODY_ERROR)
        return _success()

    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    history = _history("short" if outcome == "no_text" else LARGE_TEXT)
    db.create_session("body-read-session", source="cli")
    db.append_messages_batch("body-read-session", history)
    agent = _agent(monkeypatch, handler, session_db=db)
    original = copy.deepcopy(history)
    if outcome == "write_failure":
        def fail(*a, **kw):
            raise OSError("fixture disk full")
        monkeypatch.setattr("utils.atomic_write_text", fail)
    if outcome == "no_reader":
        agent.tools = []
    if outcome == "unavailable_backend":
        monkeypatch.setattr("tools.terminal_tool.get_active_env", lambda *a: None)
        monkeypatch.setattr("tools.terminal_tool._get_env_config", lambda: {"env_type": "docker"})
    result = agent.run_conversation("Continue with all evidence", conversation_history=history,
        task_id="body-timeout-test")
    has_retry = outcome not in {"no_text", "write_failure", "no_reader", "unavailable_backend"}
    assert len(captures) == (2 if has_retry else 1)
    assert [{k: v for k, v in m.items() if k != "_db_persisted"} for m in history] == original
    assert next(m for m in result["messages"] if m.get("role") == "tool")["content"] == original[-1]["content"]
    durable = db.get_messages_as_conversation(agent.session_id)
    assert next(m for m in durable if m.get("role") == "tool")["content"] == original[-1]["content"]
    if has_retry:
        first, second = map(json.loads, captures)
        original_output = next(i["output"] for i in first["input"] if i.get("type") == "function_call_output")
        assert len(captures[1]) < len(captures[0])
        outputs = [i for i in second["input"] if i.get("type") == "function_call_output"]
        path = extract_persisted_path(outputs[0]["output"])
        assert path and Path(path).read_text() == original_output
        from tools.file_tools import read_file_tool
        retrieved = json.loads(read_file_tool(path, offset=1, limit=2, task_id="body-timeout-test"))
        assert "café 🌊" in retrieved["content"]
        outputs[0]["output"] = original_output
        assert first == second  # Every non-tool field survives exactly.
    assert bool(result.get("completed")) == (outcome == "success")
    if outcome != "success":
        assert result["failure_retryable"] is False
        assert "Copilot" in result["final_response"]


@pytest.mark.parametrize("provider,error", [("copilot", {"error": {"code": "timeout", "message": "Request timed out"}}),
    ("openai", BODY_ERROR), ("copilot", {"error": None}), ("copilot", {"error": "Request timed out"}),
    ("copilot", {"error": {"code": "user_request_timeout", "message": "Other timeout"}})])
def test_non_target_timeout_keeps_ordinary_retry(monkeypatch, provider, error):
    captures = []
    def handler(request):
        captures.append(request.content)
        return httpx.Response(408, json=error) if len(captures) == 1 else _success()
    agent = _agent(monkeypatch, handler, provider=provider)
    result = agent.run_conversation("continue", conversation_history=_history())
    assert result["completed"]
    assert len(captures) == 2
    assert captures[0] == captures[1]


@pytest.mark.parametrize("stateful", ["previous_response_id", "conversation", "item_reference"])
def test_stateful_requests_keep_ordinary_retry(monkeypatch, stateful):
    captures = []
    def handler(request):
        captures.append(request.content)
        return httpx.Response(408, json=BODY_ERROR) if len(captures) == 1 else _success()
    agent = _agent(monkeypatch, handler)
    build = agent._build_api_kwargs
    def with_state(*a, **kw):
        payload = build(*a, **kw)
        if stateful == "item_reference":
            payload["extra_body"] = {"input": [{"type": "item_reference", "id": "msg_before"}]}
        else:
            payload["extra_body"] = {stateful: "resp_before" if stateful == "previous_response_id" else "conv_before"}
        return payload
    agent._build_api_kwargs = with_state
    assert agent.run_conversation("continue", conversation_history=_history())["completed"]
    assert len(captures) == 2 and captures[0] == captures[1]


@pytest.mark.parametrize("streaming", [False, True])
def test_current_turn_results_and_turn_wide_retry_budget(monkeypatch, streaming):
    captures, executed = [], []
    def tool_response(call_id):
        return _success([{"type": "function_call", "id": "fc_" + call_id, "call_id": call_id,
            "name": "read_file", "arguments": "{}", "status": "completed"}])
    def handler(request):
        captures.append(request.content)
        n = len(captures)
        if n in {1, 3}:
            return tool_response("call_a" if n == 1 else "call_b")
        if n in {2, 4, 5}:
            return httpx.Response(408, json=BODY_ERROR)
        return _success()
    agent = _agent(monkeypatch, handler)
    agent._disable_streaming = not streaming
    agent.tool_delay = 0
    def execute(name, args, *a, **kw):
        executed.append(name)
        return LARGE_TEXT
    monkeypatch.setattr("model_tools.handle_function_call", execute)
    result = agent.run_conversation("Read evidence twice", task_id="current-tools")
    assert len(captures) == 4 and executed == ["read_file", "read_file"]
    assert result["failed"] and not result["failure_retryable"]
    original, reduced = json.loads(captures[1]), json.loads(captures[2])
    assert len(captures[2]) < len(captures[1])
    original_output = next(i["output"] for i in original["input"] if i.get("type") == "function_call_output")
    reduced_output = next(i["output"] for i in reduced["input"] if i.get("type") == "function_call_output")
    assert Path(extract_persisted_path(reduced_output)).read_text() == original_output
    # The same cached agent starts a NEW logical user turn with a fresh one-shot allowance.
    resumed = agent.run_conversation("Continue", conversation_history=result["messages"], task_id="current-tools")
    assert resumed["completed"] and len(captures) == 6
    assert executed == ["read_file", "read_file"]


@pytest.mark.parametrize("tool_blocks", [False, True])
def test_wire_projection_preserves_native_reasoning_and_attachments(monkeypatch, tool_blocks):
    captures = []
    def handler(request):
        captures.append(request.content)
        return httpx.Response(408, json=BODY_ERROR) if len(captures) == 1 else _success()
    agent = _agent(monkeypatch, handler)
    history = _history()
    history[0]["content"] = [{"type": "text", "text": "Preserve user text"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]
    history[1]["codex_reasoning_items"] = [{"type": "reasoning", "encrypted_content": "opaque-signed-content",
        "summary": [{"type": "summary_text", "text": "Preserve reasoning"}]}]
    if tool_blocks:
        history[-1]["content"] = [{"type": "text", "text": LARGE_TEXT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,dG9vbA=="}}]
    result = agent.run_conversation("continue", conversation_history=history)
    assert result["completed"] and len(captures) == 2
    first, second = map(json.loads, captures)
    assert any(i.get("type") == "reasoning" and i.get("encrypted_content") == "opaque-signed-content"
        for i in first["input"])
    assert "data:image/png;base64,aW1hZ2U=" in captures[0].decode()
    if tool_blocks:
        assert "data:image/png;base64,dG9vbA==" in captures[0].decode()
    from agent.copilot_body_read_recovery import _text_slots
    for (before, before_key), (after, after_key) in zip(_text_slots(first), _text_slots(second)):
        if before[before_key] != after[after_key]:
            assert Path(extract_persisted_path(after[after_key])).read_text() == before[before_key]
            after[after_key] = before[before_key]
    assert first == second


def test_archives_follow_profile_scope_a_b_a(monkeypatch, tmp_path):
    from agent import secret_scope
    from agent.copilot_body_read_recovery import _archive_text
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope = secret_scope.set_secret_scope({})
    paths = []
    try:
        for name in ("a", "b", "a"):
            home = tmp_path / name
            home.mkdir(exist_ok=True)
            (home / "config.yaml").write_text("terminal:\n  backend: local\n")
            token = set_hermes_home_override(home)
            try:
                path = Path(_archive_text(LARGE_TEXT, f"profile-{name}"))
                assert path.is_relative_to(home)
                assert path.read_text() == LARGE_TEXT
                paths.append(path)
            finally:
                reset_hermes_home_override(token)
    finally:
        secret_scope.reset_secret_scope(scope)
    assert paths[0].parent == paths[2].parent != paths[1].parent
    assert len(set(paths)) == 3
