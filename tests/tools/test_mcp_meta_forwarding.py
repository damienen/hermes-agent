"""MCP session-meta forwarding patch: the call-tool handler forwards trusted session context to the MCP server
via request ``_meta`` (``hermes.sender`` + ``hermes.media``), read from the gateway's task-local
ContextVars in the SYNC handler body (before the cross-loop dispatch). The model never supplies these —
they ride ``_meta``, not the tool ``arguments`` — so sender identity + media paths are unforgeable.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool
from tools.thread_context import mark_sandbox_call, sandbox_call
from gateway.session_context import set_session_vars, clear_session_vars


class _FakeContentBlock:
    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeCallToolResult:
    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _install_lock_and_run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_install_lock_and_run())
    finally:
        loop.close()


@pytest.fixture
def fake_session():
    """Register a fake MCP server whose call_tool records its kwargs, and run _call inline."""
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=_FakeCallToolResult(content=[_FakeContentBlock("ok")])
    )
    fake_server = SimpleNamespace(session=session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield session


def _invoke(session):
    handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
    raw = handler({"q": "x"})
    assert json.loads(raw) == {"result": "ok"}
    return session.call_tool.call_args


def test_forwards_sender_and_media_in_meta(fake_session):
    tokens = set_session_vars(user_id="40751", media_paths=["/cache/a.jpg", "/cache/b.png"])
    try:
        call = _invoke(fake_session)
    finally:
        clear_session_vars(tokens)
    # tool arguments are exactly what the model supplied — identity/media are NOT in there
    assert call.kwargs["arguments"] == {"q": "x"}
    # sender + media ride _meta, sourced from trusted gateway ContextVars
    assert call.kwargs["meta"] == {
        "hermes.sender": "40751",
        "hermes.media": ["/cache/a.jpg", "/cache/b.png"],
    }


def test_sender_only_when_no_media(fake_session):
    tokens = set_session_vars(user_id="40751")
    try:
        call = _invoke(fake_session)
    finally:
        clear_session_vars(tokens)
    assert call.kwargs["meta"] == {"hermes.sender": "40751"}
    assert "hermes.media" not in call.kwargs["meta"]


def test_no_meta_when_no_session_context(fake_session):
    # CLI/cron path: no gateway session vars → nothing trusted to forward → call_tool gets NO meta kwarg,
    # i.e. exactly the upstream call signature (zero behavior change without a session).
    clear_session_vars([])  # force vars to "" so there is no os.environ fallback
    call = _invoke(fake_session)
    assert "meta" not in call.kwargs


def test_sandbox_flag_forwarded_only_from_sandbox_dispatch(fake_session):
    """A call dispatched on behalf of an execute_code script carries hermes.sandbox; a direct call does not."""
    tokens = set_session_vars(user_id="40751")
    try:
        direct = _invoke(fake_session)
        with mark_sandbox_call():
            sandboxed = _invoke(fake_session)
    finally:
        clear_session_vars(tokens)
    assert "hermes.sandbox" not in direct.kwargs["meta"]
    assert sandboxed.kwargs["meta"] == {"hermes.sender": "40751", "hermes.sandbox": True}
    # the marker is scoped: it resets when the block exits
    assert sandbox_call.get() is False


def test_sandbox_flag_never_comes_from_tool_arguments(fake_session):
    """A model cannot forge the flag by naming it in the arguments — it is trusted metadata only."""
    tokens = set_session_vars(user_id="40751")
    try:
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        handler({"q": "x", "hermes.sandbox": True})
    finally:
        clear_session_vars(tokens)
    call = fake_session.call_tool.call_args
    assert call.kwargs["arguments"] == {"q": "x", "hermes.sandbox": True}  # stays a plain argument
    assert "hermes.sandbox" not in call.kwargs["meta"]


def test_sandbox_flag_alone_without_session_is_still_forwarded(fake_session):
    """CLI code mode has no gateway session; the flag still rides _meta on its own."""
    clear_session_vars([])
    with mark_sandbox_call():
        call = _invoke(fake_session)
    assert call.kwargs["meta"] == {"hermes.sandbox": True}
