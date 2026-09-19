"""One artifact-backed retry for Copilot's explicit request-body read failure.

This is a wire projection, not transcript compaction. The session retains the
original results; only tool-result text in this turn's outgoing requests changes.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import uuid

from utils import base_url_host_matches

_BODY_READ_MESSAGE = "Timed out reading request body. Try again, or use a smaller request size."
# Eligibility floor, NOT a claimed provider request-size limit.
_MIN_RESULT_CHARS = 8000


class CopilotBodyReadStopped(Exception):
    """Must bypass generic retry, credential rotation, fallback and LLM compaction."""


def _full_input(body):
    return (
        isinstance(body, dict) and isinstance(body.get("input"), list)
        and not body.get("previous_response_id") and not body.get("conversation")
        and not any(i.get("type") in {"item_reference", "compaction"}
                    for i in body["input"] if isinstance(i, dict))
    )


def _text_slots(body):
    """Yield only tool output text; never interpret strings as JSON/media envelopes."""
    for item in body.get("input", []):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        if isinstance(output, str):
            yield item, "output"
        elif isinstance(output, list):
            for block in output:
                if isinstance(block, dict) and block.get("type") in {"input_text", "text"}:
                    if isinstance(block.get("text"), str):
                        yield block, "text"


def _archive_text(text, task_id):
    """Commit and verify a profile-local archive before advertising a readable path."""
    from tools.spill_safety import ensure_spill_dir
    from tools.tool_result_storage import (
        get_spillover_dir, _is_host_side_env, _sandbox_visible_spillover_path,
    )
    from tools.terminal_tool import _get_env_config, get_active_env
    from utils import atomic_write_text

    env = get_active_env(task_id)
    if env is None and _get_env_config().get("env_type", "local") != "local":
        raise OSError("no active terminal backend can verify the archive")
    directory = ensure_spill_dir(get_spillover_dir(), private=False)
    path = directory / f"copilot-body-{uuid.uuid4().hex}.txt"
    atomic_write_text(path, text, mode=0o600, fsync_dir=True)
    data = text.encode("utf-8")
    if path.read_bytes() != data:
        raise OSError("archive read-back differs from the original text")
    if _is_host_side_env(env):
        return str(path)
    visible = _sandbox_visible_spillover_path(str(path), env)
    if visible is None:
        raise OSError("archive is not readable in the terminal backend")
    # A mount/sync probe alone cannot establish that the backend sees the same bytes.
    with tempfile.TemporaryDirectory(prefix="hermes-body-check-") as tmp:
        fetched = Path(tmp) / "result.txt"
        env.fetch_file(visible, fetched, max_bytes=len(data))
        if fetched.read_bytes() != data:
            raise OSError("terminal backend archive differs from the original text")
    return visible


@dataclass
class CopilotBodyReadRecovery:
    attempted: bool = False
    retry_in_flight: bool = False
    replacements: dict[str, str] = field(default_factory=dict)

    def project(self, kwargs):
        if not self.replacements or not _full_input(kwargs):
            return kwargs
        projected = deepcopy(kwargs)
        for container, key in _text_slots(projected):
            container[key] = self.replacements.get(container[key], container[key])
        return projected

    def open_stream(self, agent, kwargs, send):
        """``send`` is the actual zero-SDK-retry Responses request boundary."""
        effective = {**kwargs, **(kwargs.get("extra_body") or {})}
        if not (
            agent.api_mode == "codex_responses"
            and base_url_host_matches(str(agent.base_url or ""), "githubcopilot.com")
            and _full_input(effective)
        ):
            return send(kwargs)
        kwargs = self.project(kwargs)
        try:
            return send(kwargs)
        except Exception as exc:
            body = getattr(exc, "body", None)
            error = body.get("error", body) if isinstance(body, dict) else {}
            target = (
                getattr(exc, "status_code", None) == 408
                and isinstance(error, dict)
                and error.get("code") == "user_request_timeout"
                and error.get("message") == _BODY_READ_MESSAGE
                and _full_input(kwargs)
            )
            if not target:
                raise
            if self.attempted:
                raise CopilotBodyReadStopped("The one reduced retry for this user turn was already used.") from exc
            self.attempted = True
            try:
                reduced = self._reduce(agent, kwargs, exc)
            except Exception as reduction_error:
                raise CopilotBodyReadStopped(
                    "No safe, readable tool-result archive could reduce this request."
                ) from reduction_error
            self.retry_in_flight = True
            try:
                return send(reduced)
            except Exception as retry_error:
                raise CopilotBodyReadStopped("The reduced request also failed; it was not retried again.") from retry_error

    def _reduce(self, agent, kwargs, error):
        from tools.tool_result_storage import _build_persisted_message, generate_preview, PERSISTED_OUTPUT_TAG

        # Use the SDK's actual serialized failing body, including extra_body overrides.
        original_bytes = error.response.request.content
        original = json.loads(original_bytes)
        if (
            not _full_input(original) or original["input"] != kwargs.get("input")
            or "input" in (kwargs.get("extra_body") or {})
        ):
            raise ValueError("wire input differs from the replayable full input")
        if not any(t.get("function", {}).get("name") == "read_file" for t in agent.tools or []):
            raise ValueError("read_file is unavailable for artifact retrieval")
        replacements = {}
        for container, key in _text_slots(original):
            text = container[key]
            if len(text) < _MIN_RESULT_CHARS or PERSISTED_OUTPUT_TAG in text or text in replacements:
                continue
            path = _archive_text(text, agent._current_task_id)
            preview, more = generate_preview(text)
            replacements[text] = _build_persisted_message(preview, more, len(text), path)
        candidate = deepcopy(original)
        for container, key in _text_slots(candidate):
            container[key] = replacements.get(container[key], container[key])
        # HTTPX JSON encoding: compare bytes, not tokens/chars. No fixed payload cap.
        encoded = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if not replacements or len(encoded) >= len(original_bytes):
            raise ValueError("no serialized reduction")
        # Publish only after ALL archives succeed. The canonical messages never mutate.
        self.replacements.update(replacements)
        return self.project(kwargs)
