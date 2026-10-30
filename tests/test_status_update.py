"""Unit tests for herald — all offline, no API keys required."""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from herald.app import ReviewResult, ReviewScreen, edit_in_editor, extract_color, publish_validated_draft
from herald.config import (
    DEFAULT_MAX_TOKENS,
    MAX_CONFIGURED_MAX_TOKENS,
    UserPrefs,
    _cache_key,
    find_scripts_dir,
    find_skills_dir,
    load_prefs,
)
from herald.gather import (
    _field_to_text,
    _prune_issue,
    is_significant,
    load_current_statuses,
    batch_draft,
)
from herald.jira import _normalized_adf, get_jira_auth, update_jira_status
from herald.llm import (
    TokenUsage,
    TruncatedResponseError,
    _first_text,
    build_system_prompt,
    call_with_truncation_retry,
    draft_ryg,
    request_changes_api,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_draft_state(tmp_path, monkeypatch):
    """Keep batch_draft's draft cache out of the user's global state dir.

    Draft state is keyed by the gathered output path, and pytest recycles
    ``pytest-N`` basetemps across sessions, so tests reusing a path could load
    a stale draft from a previous run and never exercise the code under test.
    """
    from herald import gather
    monkeypatch.setattr(gather, "_get_state_dir", lambda: tmp_path / "state")


# ─── build_system_prompt ─────────────────────────────────────────────────────

def test_build_system_prompt_includes_skill_docs(tmp_path):
    activity = "# Activity Analysis\n\nGreen means on-track."
    formatting = "# Formatting\n\n* Color Status: Green"
    (tmp_path / "activity-analysis.md").write_text(activity)
    (tmp_path / "formatting.md").write_text(formatting)
    prefs = UserPrefs(writing_rules=["No fractions"])

    with patch("herald.llm.find_skills_dir", return_value=tmp_path):
        prompt = build_system_prompt(prefs)

    assert activity in prompt
    assert formatting in prompt
    assert "No fractions" in prompt
    # Canonical docs must come before user rules
    assert prompt.index(activity) < prompt.index("No fractions")
    assert "output starts with `* Color Status:`" in prompt


def test_build_system_prompt_no_user_rules(tmp_path):
    (tmp_path / "activity-analysis.md").write_text("analysis")
    (tmp_path / "formatting.md").write_text("formatting")
    prefs = UserPrefs()

    with patch("herald.llm.find_skills_dir", return_value=tmp_path):
        prompt = build_system_prompt(prefs)

    assert "Additional writing rules" not in prompt


def test_build_system_prompt_missing_files_silently_skipped(tmp_path):
    prefs = UserPrefs()
    with patch("herald.llm.find_skills_dir", return_value=tmp_path):
        prompt = build_system_prompt(prefs)
    assert "You are a Jira status analyst" in prompt


# ─── stop_reason handling ────────────────────────────────────────────────────

def test_first_text_skips_thinking_blocks():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(thinking="reasoning about the evidence"),
            SimpleNamespace(text="* Color Status: Green\nUpdate"),
        ],
    )
    assert _first_text(response) == "* Color Status: Green\nUpdate"


def test_first_text_falls_back_to_thinking_without_a_text_block():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(thinking="reasoning only")],
    )
    assert _first_text(response) == "reasoning only"


def test_first_text_raises_when_response_hits_max_tokens():
    response = SimpleNamespace(
        stop_reason="max_tokens",
        content=[
            SimpleNamespace(thinking="long reasoning trace"),
            SimpleNamespace(text="partial output"),
        ],
    )
    with pytest.raises(TruncatedResponseError, match="max_tokens=8192"):
        _first_text(response, 8192)


def test_call_with_truncation_retry_doubles_cap_and_notifies():
    caps = []
    notices = []

    async def call(max_tokens):
        caps.append(max_tokens)
        if len(caps) == 1:
            raise TruncatedResponseError("truncated")
        return "complete draft"

    result = asyncio.run(call_with_truncation_retry(call, 4096, notices.append))

    assert result == "complete draft"
    assert caps == [4096, 8192]
    assert notices == [8192]


def test_call_with_truncation_retry_caps_at_configured_max():
    caps = []

    async def call(max_tokens):
        caps.append(max_tokens)
        raise TruncatedResponseError("truncated")

    with pytest.raises(TruncatedResponseError):
        asyncio.run(call_with_truncation_retry(call, MAX_CONFIGURED_MAX_TOKENS))
    assert caps == [MAX_CONFIGURED_MAX_TOKENS, MAX_CONFIGURED_MAX_TOKENS]


def test_call_with_truncation_retry_propagates_second_truncation():
    caps = []

    async def call(max_tokens):
        caps.append(max_tokens)
        raise TruncatedResponseError("truncated")

    with pytest.raises(TruncatedResponseError):
        asyncio.run(call_with_truncation_retry(call, 4096))
    assert caps == [4096, 8192]


def test_draft_ryg_sends_configured_max_tokens_and_raises_on_truncation():
    captured = {}

    class Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                stop_reason="max_tokens",
                content=[SimpleNamespace(thinking="reasoning only")],
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=8192,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0,
                ),
            )

    usage = TokenUsage()
    with pytest.raises(TruncatedResponseError):
        asyncio.run(draft_ryg(
            SimpleNamespace(messages=Messages()), "model", "system", "summary", usage,
            max_tokens=4096,
        ))
    assert captured["max_tokens"] == 4096
    assert usage.calls == 1
    assert usage.output_tokens == 8192


def test_request_changes_api_sends_configured_max_tokens_and_raises_on_truncation():
    captured = {}

    class Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                stop_reason="max_tokens",
                content=[SimpleNamespace(thinking="reasoning only")],
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=4096,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0,
                ),
            )

    usage = TokenUsage()
    with pytest.raises(TruncatedResponseError):
        asyncio.run(request_changes_api(
            SimpleNamespace(messages=Messages()), "model", "draft", "summary",
            "instruction", usage, max_tokens=4096,
        ))
    assert captured["max_tokens"] == 4096
    assert usage.calls == 1


# ─── load_prefs ──────────────────────────────────────────────────────────────

def test_load_prefs_defaults_when_file_missing(tmp_path):
    prefs = load_prefs(tmp_path / "nonexistent.toml")
    assert prefs.model == "claude-sonnet-5"
    assert prefs.max_tokens == DEFAULT_MAX_TOKENS
    assert prefs.writing_rules == []
    assert prefs.jira_url == "https://redhat.atlassian.net"
    assert prefs.profiles == {}


def test_load_prefs_from_toml(tmp_path):
    content = """
[llm]
model = "claude-haiku-4-5"
max_tokens = 4096
writing_rules = ["No fractions", "No percentages"]

[jira]
url = "https://example.atlassian.net"
skills_dir = "/custom/skills"

[profiles.myprofile]
project = "TEST"
"""
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text(content)

    prefs = load_prefs(prefs_file)

    assert prefs.model == "claude-haiku-4-5"
    assert prefs.max_tokens == 4096
    assert "No fractions" in prefs.writing_rules
    assert prefs.jira_url == "https://example.atlassian.net"
    assert prefs.skills_dir == "/custom/skills"
    assert "myprofile" in prefs.profiles


def test_cache_key_separates_distinct_custom_jql():
    base = dict(
        project=None, component=None, label=None,
        assignee=None, exclude=None, days=7,
        start_date=None, end_date=None,
    )
    first = _cache_key(argparse.Namespace(jql="project = A", **base))
    second = _cache_key(argparse.Namespace(jql="project = B", **base))
    none_jql = _cache_key(argparse.Namespace(jql=None, **base))

    assert len({first, second, none_jql}) == 3


def test_cache_key_accepts_jql_only_invocation():
    """JQL-only runs have project=None; the key must still be derivable."""
    key = _cache_key(argparse.Namespace(jql="project = A", project=None))

    assert isinstance(key, str) and len(key) == 16


def test_load_prefs_partial_fills_defaults(tmp_path):
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text("[llm]\nmodel = \"claude-opus-5\"\n")
    prefs = load_prefs(prefs_file)
    assert prefs.model == "claude-opus-5"
    assert prefs.max_tokens == DEFAULT_MAX_TOKENS
    assert prefs.writing_rules == []


@pytest.mark.parametrize("value", ["8192", 0, -1, True, 1.5, 1_000_001])
def test_load_prefs_rejects_invalid_max_tokens(tmp_path, value):
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text(f"[llm]\nmax_tokens = {json.dumps(value)}\n")
    assert load_prefs(prefs_file).max_tokens == DEFAULT_MAX_TOKENS


# ─── TokenUsage.cost ─────────────────────────────────────────────────────────

def test_token_usage_cost_arithmetic():
    usage = TokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        cache_write_tokens=2_000,
        cache_read_tokens=4_000,
        calls=5,
    )
    pricing = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}
    expected = (
        1_000 * 3.0 / 1_000_000
        + 500 * 15.0 / 1_000_000
        + 2_000 * 3.75 / 1_000_000
        + 4_000 * 0.30 / 1_000_000
    )
    assert abs(usage.cost(pricing) - expected) < 1e-10


def test_token_usage_cost_zero():
    pricing = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}
    assert TokenUsage().cost(pricing) == 0.0


# ─── get_jira_auth ───────────────────────────────────────────────────────────

def test_get_jira_auth_produces_correct_base64():
    with patch.dict(os.environ, {"JIRA_USERNAME": "user@example.com", "JIRA_API_TOKEN": "tok"}):
        headers = get_jira_auth()
    expected = base64.b64encode(b"user@example.com:tok").decode()
    assert headers["Authorization"] == f"Basic {expected}"
    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/json"


def test_get_jira_auth_raises_when_env_missing():
    with patch.dict(os.environ, {"JIRA_USERNAME": "", "JIRA_API_TOKEN": ""}):
        with pytest.raises(ValueError, match="JIRA_USERNAME and JIRA_API_TOKEN"):
            get_jira_auth()


class _Response:
    def __init__(self, status, data=None, text=""):
        self.status = status
        self.data = data
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def text(self):
        return self._text

    async def json(self):
        return self.data


class _Session:
    def __init__(self, put_status=204, get_status=200, field=None, error=""):
        self.put_status = put_status
        self.get_status = get_status
        self.field = field
        self.error = error

    def put(self, *args, **kwargs):
        if self.error:
            raise OSError(self.error)
        return _Response(self.put_status, text=self.error)

    def get(self, *args, **kwargs):
        return _Response(self.get_status, {"fields": {"customfield_10814": self.field}})


def test_jira_403_is_rejected():
    result = asyncio.run(update_jira_status(_Session(put_status=403), "https://jira", {}, "ABC-1", "draft"))
    assert not result.ok
    assert result.state == "rejected"
    assert "permission" in result.error


def test_jira_network_failure_is_unconfirmed():
    result = asyncio.run(update_jira_status(
        _Session(error="connection reset"), "https://jira", {}, "ABC-1", "draft"
    ))
    assert not result.ok
    assert result.state == "unconfirmed"
    assert "connection reset" in result.error


def test_jira_matching_adf_readback_confirms_publication():
    from herald.jira import _text_to_adf
    expected = _text_to_adf("* Color Status: Green\nDone")
    result = asyncio.run(update_jira_status(_Session(field=expected), "https://jira", {}, "ABC-1", "* Color Status: Green\nDone"))
    assert result.ok
    assert result.state == "published"


def test_jira_old_or_different_content_is_unconfirmed():
    result = asyncio.run(update_jira_status(
        _Session(field={"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "old"}]}]}),
        "https://jira", {}, "ABC-1", "new",
    ))
    assert not result.ok
    assert result.state == "unconfirmed"
    assert "does not match" in result.error


def test_adf_normalization_ignores_version_and_mark_order():
    left = {"type": "doc", "version": 1, "content": [{"type": "text", "text": "x", "marks": [{"type": "strong"}, {"type": "em"}]}]}
    right = {"version": 99, "content": [{"marks": [{"type": "em"}, {"type": "strong"}], "text": "x", "type": "text"}], "type": "doc"}
    assert _normalized_adf(left) == _normalized_adf(right)


def test_publication_failure_retry_publishes_once(monkeypatch):
    from herald import app as app_module
    from herald.jira import JiraUpdateResult
    draft = "* Color Status: Green\nImplementation complete."
    review = SimpleNamespace(
        results={"ABC-1": ReviewResult("ABC-1", "approved", "Green", draft)},
        _publication_tasks={"ABC-1": object()},
        _aiohttp_session=SimpleNamespace(closed=False), jira_url="https://jira",
        auth_headers={}, _mark_action=lambda *args: None,
        app=SimpleNamespace(notify=lambda *args, **kwargs: None),
    )
    outcomes = [
        JiraUpdateResult(False, "permission denied", "rejected"),
        JiraUpdateResult(True),
    ]

    async def fake_update(*args):
        return outcomes.pop(0)

    monkeypatch.setattr(app_module, "update_jira_status", fake_update)
    asyncio.run(ReviewScreen._do_jira_update(review, "ABC-1", draft))
    assert review.results["ABC-1"].publication_state == "failed"
    assert review.results["ABC-1"].publication_error == "permission denied"

    review.results["ABC-1"].publication_state = "pending"
    review._publication_tasks["ABC-1"] = object()
    asyncio.run(ReviewScreen._do_jira_update(review, "ABC-1", draft))
    assert review.results["ABC-1"].publication_state == "published"
    assert [r for r in review.results.values() if r.publication_state == "published"] == [review.results["ABC-1"]]


def test_quit_waits_for_pending_publication():
    result = ReviewResult("ABC-1", "approved", "Green", "draft", publication_state="pending")
    review = SimpleNamespace(
        _publication_tasks={"ABC-1": object()}, _aiohttp_session=None,
        results={"ABC-1": result}, app=SimpleNamespace(notify=lambda *args, **kwargs: None),
        dismissed=None,
    )
    review.dismiss = lambda results: setattr(review, "dismissed", results)

    async def finish_write():
        await asyncio.sleep(0.02)
        result.publication_state = "published"
        review._publication_tasks.clear()

    async def run_quit():
        asyncio.create_task(finish_write())
        await ReviewScreen.action_quit_review(review)

    asyncio.run(run_quit())
    assert review.dismissed == [result]
    assert result.publication_state == "published"


# ─── extract_color ───────────────────────────────────────────────────────────

def test_extract_color_green():
    assert extract_color("* Color Status: Green\nSome text") == "Green"


def test_extract_color_yellow():
    assert extract_color("* Color Status: Yellow\nAt risk") == "Yellow"


def test_extract_color_red():
    assert extract_color("* Color Status: Red\nBlocked") == "Red"


def test_extract_color_unknown_when_missing_or_ambiguous():
    assert extract_color("No color line here") == "Unknown"
    assert extract_color("* Color Status: Red\n* Color Status: Green\nUpdate") == "Unknown"


def test_extract_color_case_sensitive():
    assert extract_color("* Color Status: green\nUpdate details") == "Green"


def test_invalid_drafts_never_call_jira_writer():
    calls = []

    async def writer(*args):
        calls.append(args)

    for text in ("", "Not a status update", "* Color Status: Green", "* Color Status: Blue\nUpdate"):
        result, error = asyncio.run(publish_validated_draft(writer, None, "url", {}, "ABC-1", text))
        assert result is None
        assert error
    assert calls == []


def test_valid_draft_remains_publishable():
    async def writer(*args):
        return "written"

    result, error = asyncio.run(publish_validated_draft(
        writer, None, "url", {}, "ABC-1", "* Color Status: Yellow\nUpdate details"
    ))
    assert result == "written"
    assert error is None


@pytest.mark.parametrize("draft", ["", "Not a status update", "* Color Status: Green"])
def test_invalid_draft_does_not_start_publication(draft):
    table = MagicMock()
    review = SimpleNamespace(
        _current_key=lambda: "ABC-1", drafts={"ABC-1": draft},
        _publication_tasks={}, results={}, _col_keys=(0, 1, 2, 3),
        query_one=lambda *args: table,
        app=SimpleNamespace(notify=MagicMock()),
        _start_jira_update=MagicMock(), _advance=MagicMock(),
    )
    ReviewScreen.action_approve(review)
    review._start_jira_update.assert_not_called()
    review._advance.assert_not_called()
    assert review.results == {}
    review.app.notify.assert_called_once()


def test_invalid_worker_draft_clears_pending_publication(monkeypatch):
    from herald import app as app_module

    writer = MagicMock()
    monkeypatch.setattr(app_module, "update_jira_status", writer)
    review = SimpleNamespace(
        results={"ABC-1": ReviewResult("ABC-1", "approved", "Unknown", "")},
        _publication_tasks={"ABC-1": object()},
        _aiohttp_session=SimpleNamespace(closed=False), jira_url="https://jira",
        auth_headers={}, _mark_action=MagicMock(),
        app=SimpleNamespace(notify=MagicMock()),
    )
    asyncio.run(ReviewScreen._do_jira_update(review, "ABC-1", ""))
    writer.assert_not_called()
    assert review._publication_tasks == {}
    assert review.results["ABC-1"].publication_state == "failed"
    assert review.results["ABC-1"].publication_error
    review._mark_action.assert_called_once_with("ABC-1", "Invalid draft")


def test_batch_draft_distinguishes_quiet_skip_and_missing_data(tmp_path):
    from herald import gather

    issues = [{"key": "QUIET-1"}, {"key": "MISSING-1"}]
    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "QUIET-1.json").write_text(json.dumps({"issue": {}, "prs": []}))
    usage = TokenUsage()
    drafts, summaries, failed, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, issues, usage, only_significant=True
    ))
    assert drafts == {}
    assert outcomes == {"QUIET-1": "quiet_skip", "MISSING-1": "missing_data"}
    assert failed == ["MISSING-1"]


def test_batch_draft_rejects_malformed_llm_output(tmp_path, monkeypatch):
    from herald import gather

    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "ABC-1.json").write_text(json.dumps({"issue": {"key": "ABC-1"}}))

    async def malformed(*args, **kwargs):
        return "No status here"

    monkeypatch.setattr(gather, "draft_ryg", malformed)
    drafts, summaries, failed, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, [{"key": "ABC-1"}], TokenUsage()
    ))
    assert drafts == {}
    assert failed == ["ABC-1"]
    assert outcomes == {"ABC-1": "invalid_draft"}


def test_batch_draft_classifies_llm_failure_separately(tmp_path, monkeypatch):
    from herald import gather

    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "ABC-1.json").write_text(json.dumps({"issue": {"key": "ABC-1"}}))

    async def broken(*args, **kwargs):
        raise RuntimeError("offline")

    async def no_wait(*args):
        return None

    monkeypatch.setattr(gather, "draft_ryg", broken)
    monkeypatch.setattr(gather.asyncio, "sleep", no_wait)
    drafts, summaries, failed, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, [{"key": "ABC-1"}], TokenUsage()
    ))
    assert drafts == {}
    assert failed == ["ABC-1"]
    assert outcomes == {"ABC-1": "draft_failed"}


def test_batch_draft_classifies_truncation_as_draft_failure(tmp_path, monkeypatch):
    from herald import gather

    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "ABC-1.json").write_text(json.dumps({"issue": {"key": "ABC-1"}}))
    caps = []

    async def truncated(client, model, prompt, summary, usage, *, max_tokens):
        caps.append(max_tokens)
        raise TruncatedResponseError("Response was truncated at max_tokens=8192")

    monkeypatch.setattr(gather, "draft_ryg", truncated)
    drafts, summaries, failed, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, [{"key": "ABC-1"}], TokenUsage()
    ))
    assert caps == [DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS * 2]
    assert drafts == {}
    assert failed == ["ABC-1"]
    assert outcomes == {"ABC-1": "draft_failed"}


def test_batch_draft_retries_truncation_with_doubled_cap(tmp_path, monkeypatch):
    from herald import gather

    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "ABC-1.json").write_text(json.dumps({"issue": {"key": "ABC-1"}}))
    caps = []

    async def truncated_then_ok(client, model, prompt, summary, usage, *, max_tokens):
        caps.append(max_tokens)
        if len(caps) == 1:
            raise TruncatedResponseError("truncated")
        return "* Color Status: Green\nUpdate"

    monkeypatch.setattr(gather, "draft_ryg", truncated_then_ok)
    drafts, _, failed, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, [{"key": "ABC-1"}], TokenUsage(),
        max_tokens=4096,
    ))
    assert caps == [4096, 8192]
    assert failed == []
    assert outcomes == {"ABC-1": "drafted"}
    assert drafts["ABC-1"].startswith("* Color Status: Green")


def test_batch_draft_retries_truncation_after_prior_failures(tmp_path, monkeypatch):
    from herald import gather

    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "ABC-1.json").write_text(json.dumps({"issue": {"key": "ABC-1"}}))
    caps = []

    async def fail_then_truncate(client, model, prompt, summary, usage, *, max_tokens):
        caps.append(max_tokens)
        if len(caps) == 1:
            raise RuntimeError("transient")
        if len(caps) == 2:
            raise TruncatedResponseError("truncated")
        return "* Color Status: Green\nUpdate"

    async def no_wait(*args):
        return None

    monkeypatch.setattr(gather, "draft_ryg", fail_then_truncate)
    monkeypatch.setattr(gather.asyncio, "sleep", no_wait)
    drafts, _, failed, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, [{"key": "ABC-1"}], TokenUsage(),
        max_tokens=4096,
    ))
    assert caps == [4096, 4096, 8192]
    assert failed == []
    assert outcomes == {"ABC-1": "drafted"}


def test_batch_draft_forwards_max_tokens(tmp_path, monkeypatch):
    from herald import gather

    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    (issue_dir / "ABC-1.json").write_text(json.dumps({"issue": {"key": "ABC-1"}}))
    seen = {}

    async def capture(client, model, prompt, summary, usage, *, max_tokens):
        seen["max_tokens"] = max_tokens
        return "* Color Status: Green\nUpdate"

    monkeypatch.setattr(gather, "draft_ryg", capture)
    drafts, _, _, outcomes = asyncio.run(batch_draft(
        None, "model", "prompt", tmp_path, [{"key": "ABC-1"}], TokenUsage(),
        max_tokens=4096,
    ))
    assert seen["max_tokens"] == 4096
    assert outcomes == {"ABC-1": "drafted"}


# ─── truncation retry in the review TUI ──────────────────────────────────────

def _review_shim(**overrides):
    values = dict(
        client=None,
        model="model",
        usage=TokenUsage(),
        max_tokens=4096,
        drafts={"ABC-1": "old draft"},
        summaries={},
        outcomes={},
        results={},
        date_dir=None,
        _save_drafts=MagicMock(),
        _update_draft_panel=MagicMock(),
        _col_keys=(0, 1, 2, 3),
        query_one=lambda *args: MagicMock(),
        app=SimpleNamespace(notify=MagicMock()),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_request_changes_retries_truncation_with_doubled_cap(monkeypatch):
    from herald import app as app_module

    caps = []

    async def fake(client, model, draft, summary, instruction, usage, *, max_tokens):
        caps.append(max_tokens)
        if len(caps) == 1:
            raise TruncatedResponseError("truncated")
        return "* Color Status: Green\nUpdate"

    monkeypatch.setattr(app_module, "request_changes_api", fake)
    review = _review_shim()
    asyncio.run(ReviewScreen._do_request_changes(
        review, "ABC-1", "old draft", "summary", "change it"
    ))

    assert caps == [4096, 8192]
    assert review.outcomes == {"ABC-1": "drafted"}
    assert review.drafts["ABC-1"].startswith("* Color Status: Green")
    review.app.notify.assert_called_once()


def test_request_changes_reports_failure_after_second_truncation(monkeypatch):
    from herald import app as app_module

    async def always_truncated(client, model, draft, summary, instruction, usage, *, max_tokens):
        raise TruncatedResponseError("truncated")

    monkeypatch.setattr(app_module, "request_changes_api", always_truncated)
    review = _review_shim()
    asyncio.run(ReviewScreen._do_request_changes(
        review, "ABC-1", "old draft", "summary", "change it"
    ))

    assert review.outcomes == {}
    assert review.drafts["ABC-1"] == "old draft"
    titles = [call.kwargs.get("title") for call in review.app.notify.call_args_list]
    assert "Revision failed" in titles


def test_refresh_retries_truncation_with_doubled_cap(tmp_path, monkeypatch):
    from herald import app as app_module

    (tmp_path / "issues").mkdir()
    (tmp_path / "issues" / "ABC-1.json").write_text(
        json.dumps({"issue": {"key": "ABC-1"}})
    )
    caps = []

    async def fake_gather(*args, **kwargs):
        return tmp_path, {}

    async def fake_draft(client, model, prompt, summary, usage, *, max_tokens):
        caps.append(max_tokens)
        if len(caps) == 1:
            raise TruncatedResponseError("truncated")
        return "* Color Status: Green\nUpdate"

    monkeypatch.setattr(app_module, "run_gather", fake_gather)
    monkeypatch.setattr(app_module, "draft_ryg", fake_draft)
    monkeypatch.setattr(app_module, "_prune_issue", lambda data, manifest: {"key": "ABC-1"})
    monkeypatch.setattr(app_module, "save_draft_input", lambda *args: None)

    review = _review_shim(
        scripts_dir=tmp_path,
        args=SimpleNamespace(output_dir=str(tmp_path / "out")),
        system_prompt="system",
        drafts={},
    )
    asyncio.run(ReviewScreen._do_refresh(review, "ABC-1"))

    assert caps == [4096, 8192]
    assert review.outcomes == {"ABC-1": "drafted"}
    assert review.drafts["ABC-1"].startswith("* Color Status: Green")


def test_refresh_reports_failure_after_second_truncation(tmp_path, monkeypatch):
    from herald import app as app_module

    (tmp_path / "issues").mkdir()
    (tmp_path / "issues" / "ABC-1.json").write_text(
        json.dumps({"issue": {"key": "ABC-1"}})
    )

    async def fake_gather(*args, **kwargs):
        return tmp_path, {}

    async def always_truncated(client, model, prompt, summary, usage, *, max_tokens):
        raise TruncatedResponseError("truncated")

    monkeypatch.setattr(app_module, "run_gather", fake_gather)
    monkeypatch.setattr(app_module, "draft_ryg", always_truncated)
    monkeypatch.setattr(app_module, "_prune_issue", lambda data, manifest: {"key": "ABC-1"})

    review = _review_shim(
        scripts_dir=tmp_path,
        args=SimpleNamespace(output_dir=str(tmp_path / "out")),
        system_prompt="system",
        drafts={},
    )
    asyncio.run(ReviewScreen._do_refresh(review, "ABC-1"))

    assert review.outcomes == {"ABC-1": "draft_failed"}
    titles = [call.kwargs.get("title") for call in review.app.notify.call_args_list]
    assert "Refresh failed" in titles


# ─── find_scripts_dir / find_skills_dir ──────────────────────────────────────

def test_find_skills_dir_uses_prefs_skills_dir(tmp_path):
    fake_skills = tmp_path / "status-analysis"
    fake_skills.mkdir()
    prefs = UserPrefs(skills_dir=str(fake_skills))
    assert find_skills_dir(prefs) == fake_skills


def test_find_skills_dir_prefs_missing_path_raises(tmp_path):
    prefs = UserPrefs(skills_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError, match="skills_dir from prefs not found"):
        find_skills_dir(prefs)


def test_find_skills_dir_env_var(tmp_path):
    fake_skills = tmp_path / "status-analysis"
    fake_skills.mkdir()
    with patch.dict(os.environ, {"HERALD_SKILLS_DIR": str(fake_skills)}):
        assert find_skills_dir() == fake_skills


def test_find_skills_dir_env_var_missing_raises(tmp_path):
    with patch.dict(os.environ, {"HERALD_SKILLS_DIR": str(tmp_path / "gone")}):
        with pytest.raises(FileNotFoundError, match="HERALD_SKILLS_DIR not found"):
            find_skills_dir()


def test_find_skills_dir_repo_relative_fallback(tmp_path):
    # When no prefs/env/cached copy is available, the repo-relative heuristic
    # (step 4) is used. Patched so the test is offline and independent of where
    # the package happens to be checked out.
    fake_skills = tmp_path / "skills" / "status-analysis"
    fake_skills.mkdir(parents=True)
    with patch("herald.config._get_data_dir", return_value=tmp_path / "data"), \
         patch("herald.config._find_repo_skills_dir", return_value=fake_skills):
        assert find_skills_dir() == fake_skills


def test_find_skills_dir_fetches_when_all_else_missing(tmp_path):
    # Final fallback (step 5) delegates to _fetch_ai_helpers. Mocked so the
    # test never performs a real network clone.
    fetched = tmp_path / "fetched" / "status-analysis"
    with patch("herald.config._get_data_dir", return_value=tmp_path / "data"), \
         patch("herald.config._find_repo_skills_dir", return_value=None), \
         patch("herald.config._fetch_ai_helpers", return_value=fetched) as mock_fetch:
        assert find_skills_dir() == fetched
    mock_fetch.assert_called_once()


def test_find_scripts_dir_returns_scripts_subdir(tmp_path):
    fake_skills = tmp_path / "status-analysis"
    (fake_skills / "scripts").mkdir(parents=True)
    prefs = UserPrefs(skills_dir=str(fake_skills))
    assert find_scripts_dir(prefs).name == "scripts"
    assert find_scripts_dir(prefs).parent == fake_skills


# ─── edit_in_editor ──────────────────────────────────────────────────────────

def test_edit_in_editor_returns_edited_text():
    expected = "This is the edited draft."

    def fake_run(*args, **kwargs):
        tmpfile = args[0][1]
        with open(tmpfile, "w") as f:
            f.write(expected + "\n")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"EDITOR": "vi"}):
            result = edit_in_editor("original draft")

    assert result == expected


def test_edit_in_editor_strips_comment_lines():
    def fake_run(*args, **kwargs):
        tmpfile = args[0][1]
        with open(tmpfile, "w") as f:
            f.write("Actual content\n# This comment should be stripped\n")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"EDITOR": "vi"}):
            result = edit_in_editor("original")

    assert result == "Actual content"
    assert "comment" not in result


def test_edit_in_editor_returns_none_on_empty_file():
    def fake_run(*args, **kwargs):
        tmpfile = args[0][1]
        with open(tmpfile, "w") as f:
            f.write("")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"EDITOR": "vi"}):
            result = edit_in_editor("original")

    assert result is None


# ─── _prune_issue / is_significant ───────────────────────────────────────────

def _make_issue_data(**kwargs):
    """Minimal per-issue JSON dict for testing."""
    base = {
        "issue": {
            "key": "ISSUE-1",
            "summary": "Test issue",
            "status": "In Progress",
            "assignee": {"name": "Alice", "email": "alice@example.com"},
            "current_status_summary": None,
            "last_status_summary_update": None,
        },
        "descendants": {"total": 0, "completion_pct": 0.0, "by_status": {}, "updated_in_range": []},
        "changelog_in_range": [],
        "comments_in_range": [],
        "prs": [],
    }
    base.update(kwargs)
    return base


def test_prune_issue_includes_key_fields():
    data = _make_issue_data()
    pruned = _prune_issue(data)
    assert pruned["key"] == "ISSUE-1"
    assert pruned["summary"] == "Test issue"
    assert pruned["status"] == "In Progress"
    assert pruned["assignee"] == "Alice"
    assert pruned["reporting_window"]["status"] == "unknown"
    assert pruned["gather_freshness"] == "unknown"
    assert "descendants" in pruned["carry_forward_context"]
    assert "current_status_summary" in pruned["carry_forward_context"]
    assert pruned["events_in_window"] == []
    assert pruned["undated_activity"] == []
    assert "pull_requests" in pruned["carry_forward_context"]


def test_prune_issue_converts_adf_current_status():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "Green - on track"}]}
    ]}
    data = _make_issue_data()
    data["issue"]["current_status_summary"] = adf
    pruned = _prune_issue(data)
    current_summary = pruned["carry_forward_context"]["current_status_summary"]
    assert "Green - on track" in current_summary
    assert isinstance(current_summary, str)


def test_prune_issue_strips_bot_comments():
    data = _make_issue_data()
    data["comments_in_range"] = [
        {"date": "2026-08-01T00:00:00", "author_name": "bot", "body": "automated", "is_bot": True},
        {"date": "2026-08-01T00:00:00", "author_name": "human", "body": "real comment", "is_bot": False},
    ]
    pruned = _prune_issue(data)
    assert len(pruned["undated_activity"]) == 1
    assert pruned["undated_activity"][0]["author"] == "human"


def test_prune_issue_pr_drops_raw_arrays():
    data = _make_issue_data()
    data["prs"] = [{
        "number": 42,
        "title": "Fix bug",
        "state": "MERGED",
        "is_draft": False,
        "merged_at": "2026-08-01T12:00:00Z",
        "dates": {"merged_at": "2026-08-01T12:00:00Z"},
        "activity_summary": {"commits_in_range": 3, "reviews_in_range": 1, "review_comments_in_range": 2},
        "commits_in_range": [{"sha": "abc123", "author_name": "Alice"}],
        "reviews_in_range": [{"author_name": "Bob"}],
    }]
    pruned = _prune_issue(data)
    pr = pruned["carry_forward_context"]["pull_requests"][0]
    assert pr["number"] == 42
    assert pr["state"] == "MERGED"
    assert "commits_in_range" not in pr
    assert "reviews_in_range" not in pr
    assert pruned["events_in_window"] == []
    assert "sha" not in str(pruned)


def test_prune_issue_classifies_events_using_half_open_window():
    data = _make_issue_data(events_in_window=[
        {"type": "pr_review", "occurred_at": "2026-08-01T00:00:00Z", "text": "reviewed"},
        {"type": "pr_commit", "occurred_at": "2026-08-08T00:00:00Z", "text": "boundary"},
        {"type": "jira_comment", "text": "missing timestamp"},
    ])
    manifest = {
        "reporting_window": {
            "start": "2026-08-01T00:00:00Z",
            "end_exclusive": "2026-08-08T00:00:00Z",
            "timezone": "Europe/Prague",
        },
        "issue_gathered_at": {"ISSUE-1": "2026-08-08T00:05:00Z"},
    }

    pruned = _prune_issue(data, manifest)

    assert pruned["reporting_window"]["status"] == "known"
    assert pruned["gathered_at"] == "2026-08-08T00:05:00Z"
    assert [event["type"] for event in pruned["events_in_window"]] == ["pr_review"]
    assert pruned["carry_forward_context"]["older_activity"][0]["type"] == "pr_commit"
    assert pruned["undated_activity"][0]["type"] == "jira_comment"


def test_prune_issue_keeps_recent_review_separate_from_old_merge():
    data = _make_issue_data(
        events_in_window=[{
            "type": "pr_review",
            "occurred_at": "2026-08-04T12:00:00Z",
            "pull_request": 42,
        }],
    )
    data["prs"] = [{
        "number": 42,
        "title": "Fix bug",
        "state": "MERGED",
        "merged_at": "2026-07-15T12:00:00Z",
    }]
    manifest = {
        "reporting_window": {
            "start": "2026-08-01T00:00:00Z",
            "end_exclusive": "2026-08-08T00:00:00Z",
            "timezone": "Europe/Prague",
        },
        "issue_gathered_at": {"ISSUE-1": "2026-08-08T00:05:00Z"},
    }

    pruned = _prune_issue(data, manifest)

    assert pruned["events_in_window"][0]["type"] == "pr_review"
    assert pruned["carry_forward_context"]["pull_requests"][0]["merged_at"] == "2026-07-15T12:00:00Z"
    assert len(pruned["events_in_window"]) == 1


def test_is_significant_active_pr():
    data = _make_issue_data()
    data["prs"] = [{"state": "OPEN", "activity_summary": {"commits_in_range": 2, "reviews_in_range": 0}}]
    assert is_significant(data) is True


def test_is_significant_no_activity():
    data = _make_issue_data()
    assert is_significant(data) is False


def test_field_to_text_plain_string():
    assert _field_to_text("hello") == "hello"


def test_field_to_text_adf_dict():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "status text"}]}
    ]}
    result = _field_to_text(adf)
    assert "status text" in result


# ─── load_current_statuses ───────────────────────────────────────────────────

def test_load_current_statuses_reads_from_json(tmp_path):
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "ISSUE-1.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-1", "current_status_summary": "Green - on track"}
    }))
    (issues_dir / "ISSUE-2.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-2", "current_status_summary": ""}
    }))

    statuses = load_current_statuses(tmp_path, ["ISSUE-1", "ISSUE-2", "ISSUE-3"])

    assert statuses["ISSUE-1"] == "Green - on track"
    assert statuses["ISSUE-2"] == ""
    assert statuses["ISSUE-3"] == ""  # missing file → empty string


def test_load_current_statuses_handles_missing_field(tmp_path):
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "ISSUE-1.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-1"}  # no current_status_summary key
    }))

    statuses = load_current_statuses(tmp_path, ["ISSUE-1"])
    assert statuses["ISSUE-1"] == ""


def test_load_current_statuses_converts_adf_to_markdown(tmp_path):
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Green - on track"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (issues_dir / "ISSUE-1.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-1", "current_status_summary": adf}
    }))

    statuses = load_current_statuses(tmp_path, ["ISSUE-1"])
    assert "Green - on track" in statuses["ISSUE-1"]
    assert isinstance(statuses["ISSUE-1"], str)


# ─── Profile CLI override ────────────────────────────────────────────────────

def test_profile_values_fill_missing_cli_args():
    prefs = UserPrefs(
        profiles={"my-profile": {"component": "Profile Component", "label": "profile-label"}}
    )
    args = argparse.Namespace(
        project="TEST",
        component="CLI Component",  # explicitly set — must NOT be overridden
        label=None,                  # not set — profile should fill this in
        profile="my-profile",
    )

    if args.profile and args.profile in prefs.profiles:
        for key, value in prefs.profiles[args.profile].items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    assert args.component == "CLI Component"   # CLI wins
    assert args.label == "profile-label"       # profile fills missing


def test_profile_does_not_override_explicitly_set_args():
    prefs = UserPrefs(
        profiles={"p": {"component": "Profile", "label": "plabel", "model": "claude-haiku-4-5"}}
    )
    args = argparse.Namespace(
        project="TEST",
        component="My Component",
        label="my-label",
        model=None,
        profile="p",
    )

    if args.profile and args.profile in prefs.profiles:
        for key, value in prefs.profiles[args.profile].items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    assert args.component == "My Component"
    assert args.label == "my-label"
    assert args.model == "claude-haiku-4-5"  # profile fills model (was None)
