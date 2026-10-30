"""Data gathering subprocess, issue JSON processing, draft persistence, and batch drafting."""

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyadf

from .config import (
    DEFAULT_MAX_TOKENS, _find_cached_gather, _get_state_dir, _save_cache_meta,
)
from .llm import (
    TokenUsage, TruncatedResponseError, call_with_truncation_retry, draft_ryg,
)
from .validation import validate_draft


def _manifest_gathered_at(manifest: dict) -> Optional[str]:
    """Return the gather timestamp when the manifest records one."""
    value = manifest.get("gathered_at")
    return value if _parse_instant(value) else None


def _parse_instant(value) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return instant if instant.tzinfo is not None else None


def _github_repository(url: Optional[str]) -> Optional[str]:
    """Extract owner/repo from a canonical GitHub pull request URL."""
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.hostname == "github.com"
        and len(parts) >= 4
        and parts[2] == "pull"
        and parts[0]
        and parts[1]
    ):
        return f"{parts[0]}/{parts[1]}"
    return None


def _temporal_context(manifest: Optional[dict], issue_key: Optional[str] = None) -> dict:
    """Build an explicit temporal envelope, marking legacy metadata unknown."""
    manifest = manifest or {}
    window = manifest.get("reporting_window")
    if not isinstance(window, dict):
        window = {}
    start = window.get("start")
    end = window.get("end_exclusive")
    timezone_name = window.get("timezone")
    start_instant = _parse_instant(start)
    end_instant = _parse_instant(end)
    try:
        ZoneInfo(timezone_name)
        valid_timezone = True
    except (ZoneInfoNotFoundError, TypeError, ValueError):
        valid_timezone = False
    known = bool(
        start_instant and end_instant and start_instant < end_instant and valid_timezone
    )
    gathered_at = None
    issue_timestamps = manifest.get("issue_gathered_at")
    if isinstance(issue_timestamps, dict) and issue_key:
        gathered_at = issue_timestamps.get(issue_key)
    if not gathered_at and not isinstance(issue_timestamps, dict):
        gathered_at = _manifest_gathered_at(manifest)
    gathered_at = gathered_at if _parse_instant(gathered_at) else None
    return {
        "reporting_window": {
            "start": start if known else None,
            "end_exclusive": end if known else None,
            "timezone": timezone_name if known else None,
            "status": "known" if known else "unknown",
        },
        "gathered_at": gathered_at,
        "gather_freshness": "known" if gathered_at else "unknown",
    }


def _persist_manifest(date_dir: Path, manifest: dict) -> None:
    path = date_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


async def run_gather(
    scripts_dir: Path,
    args,
    output_dir: str = ".work/weekly-status",
    line_callback=None,
    refresh: bool = False,
    issue_keys: Optional[list[str]] = None,
) -> tuple:
    """Run gather_status_data.py and return (date_dir, manifest).

    Uses a cache keyed on the gather parameters. Pass refresh=True to force
    re-gather. The script writes manifest JSON to stdout and logs to stderr.
    line_callback, if provided, is called with each stderr line as it arrives.
    date_dir is reconstructed from the manifest's date_range.end field.
    When issue_keys is provided, only those specific issues are gathered.
    """
    if not refresh:
        cached = _find_cached_gather(args)
        if cached:
            if line_callback:
                await line_callback(f"Using cached data from {cached[0]}")
            return cached
    cmd = [
        sys.executable,
        str(scripts_dir / "gather_status_data.py"),
        "--output-dir", output_dir,
        "--verbose",
    ]
    if getattr(args, "project", None):
        cmd.extend(["--project", args.project])
    if getattr(args, "jql", None):
        cmd.extend(["--jql", args.jql])
    if getattr(args, "component", None):
        cmd.extend(["--component", args.component])
    if getattr(args, "label", None):
        cmd.extend(["--label", args.label])
    for assignee in (getattr(args, "assignee", None) or []):
        cmd.extend(["--assignee", assignee])
    for excluded in (getattr(args, "exclude", None) or []):
        cmd.extend(["--exclude-assignee", excluded])
    if getattr(args, "days", None):
        cmd.extend(["--days", str(args.days)])
    if getattr(args, "start_date", None):
        cmd.extend(["--start-date", args.start_date])
    if getattr(args, "end_date", None):
        cmd.extend(["--end-date", args.end_date])
    for key in (issue_keys or []):
        cmd.extend(["--issue", key])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stderr_lines: list = []

    async def _drain_stderr():
        async for raw in proc.stderr:
            line = raw.decode().rstrip()
            stderr_lines.append(line)
            if line_callback:
                await line_callback(line)

    stdout_bytes, _ = await asyncio.gather(proc.stdout.read(), _drain_stderr())
    await proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("gather_status_data.py failed:\n" + "\n".join(stderr_lines))

    new_manifest = json.loads(stdout_bytes.decode())
    end_date = new_manifest["config"]["date_range"]["end"]
    date_dir = Path(output_dir) / end_date

    # Stamp the completed gather so later cached sessions can distinguish a
    # known capture time from legacy manifests that predate this field.
    new_manifest["gathered_at"] = datetime.now(timezone.utc).isoformat()
    gathered_keys = set(issue_keys) if issue_keys else None
    new_manifest["issue_gathered_at"] = {
        issue["key"]: new_manifest["gathered_at"]
        for issue in new_manifest.get("issues", [])
        if issue.get("key") and (gathered_keys is None or issue["key"] in gathered_keys)
    }

    if issue_keys:
        manifest_path = date_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = json.load(f)
            refreshed = {i["key"]: i for i in new_manifest.get("issues", [])}
            merged = [refreshed.pop(i["key"], i) for i in existing["issues"]]
            merged.extend(refreshed.values())
            existing["issues"] = merged
            existing["gathered_at"] = new_manifest["gathered_at"]
            issue_timestamps = existing.get("issue_gathered_at", {})
            if not isinstance(issue_timestamps, dict):
                issue_timestamps = {}
            issue_timestamps.update(new_manifest["issue_gathered_at"])
            existing["issue_gathered_at"] = issue_timestamps
            # Preserve a resolved window only when the gatherer supplies one.
            existing.pop("reporting_window", None)
            if isinstance(new_manifest.get("reporting_window"), dict):
                existing["reporting_window"] = new_manifest["reporting_window"]
            _persist_manifest(date_dir, existing)
            return date_dir, existing
        _persist_manifest(date_dir, new_manifest)
        return date_dir, new_manifest

    _persist_manifest(date_dir, new_manifest)
    _save_cache_meta(args, date_dir)
    return date_dir, new_manifest


# ─── Issue JSON Processing ───────────────────────────────────────────────────

def _field_to_text(value) -> str:
    """Convert a field value to plain text, handling ADF dicts."""
    if not value:
        return ""
    if isinstance(value, dict):
        try:
            return pyadf.Document(value).to_markdown()
        except Exception:
            return json.dumps(value)
    return str(value)


def is_significant(data: dict) -> bool:
    """Return True if the issue had meaningful activity in the reporting range."""
    if data.get("events_in_window"):
        return True
    for cl in data.get("changelog_in_range", []):
        if any(i.get("field") == "status" for i in cl.get("items", [])):
            return True
    if any(not c.get("is_bot") for c in data.get("comments_in_range", [])):
        return True
    desc = data.get("descendants", {})
    if isinstance(desc, dict) and desc.get("updated_in_range"):
        return True
    for pr in data.get("prs", []):
        if pr.get("activity_summary", {}).get("commits_in_range"):
            return True
        if pr.get("activity_summary", {}).get("reviews_in_range"):
            return True
    return False


def _prune_issue(data: dict, manifest: Optional[dict] = None) -> dict:
    """Return a compact subset of the per-issue JSON suitable for LLM input."""
    issue = data.get("issue", data)
    assignee = issue.get("assignee")
    temporal = _temporal_context(manifest, issue.get("key"))
    pruned = {
        "key": issue.get("key"),
        "summary": issue.get("summary"),
        "status": issue.get("status"),
        "assignee": assignee.get("name", "?") if isinstance(assignee, dict) else (assignee or "?"),
        **temporal,
    }

    # The gatherer owns event discovery and classification. Do not infer event
    # dates from current PR state or from counts such as reviews_in_range.
    events = data.get("events_in_window")
    pruned["events_in_window"] = []
    undated = data.get("undated_activity")
    pruned["undated_activity"] = list(undated) if isinstance(undated, list) else []
    carry_forward = data.get("carry_forward_context")
    if not isinstance(carry_forward, dict):
        carry_forward = {}
    pruned["carry_forward_context"] = {
        **carry_forward,
        "current_status_summary": _field_to_text(issue.get("current_status_summary") or ""),
        "last_status_update": issue.get("last_status_summary_update", ""),
        "status": issue.get("status"),
    }
    older_activity = carry_forward.get("older_activity")
    pruned["carry_forward_context"]["older_activity"] = (
        list(older_activity) if isinstance(older_activity, list) else []
    )

    temporal_known = temporal["reporting_window"]["status"] == "known"
    window_start = _parse_instant(temporal["reporting_window"]["start"])
    window_end = _parse_instant(temporal["reporting_window"]["end_exclusive"])

    def classify_event(event: dict) -> None:
        event = dict(event)
        occurred_at = _parse_instant(event.get("occurred_at"))
        if not occurred_at:
            event["date_confidence"] = "unknown"
            pruned["undated_activity"].append(event)
        elif temporal_known and window_start <= occurred_at < window_end:
            pruned["events_in_window"].append(event)
        else:
            # A timestamp outside the range is useful history, but cannot
            # support a new accomplishment for this report.
            pruned["carry_forward_context"]["older_activity"].append(event)

    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                classify_event(event)
            else:
                pruned["undated_activity"].append({
                    "value": event,
                    "date_confidence": "unknown",
                })

    desc = data.get("descendants", {})
    if isinstance(desc, dict):
        pruned["carry_forward_context"]["descendants"] = {
            "total": desc.get("total", 0),
            "completion_pct": round(desc.get("completion_pct", 0.0)),
            "by_status": desc.get("by_status", {}),
        }

    pruned["carry_forward_context"]["pull_requests"] = [
        {
            "number": pr.get("number"),
            "repository": pr.get("repository") or _github_repository(pr.get("url")),
            "url": pr.get("url"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "is_draft": pr.get("is_draft", False),
            "merged_at": pr.get("merged_at"),
            "dates": pr.get("dates", {}),
        }
        for pr in data.get("prs", [])
    ]

    # Legacy source-specific in-range buckets can be retained as dated events
    # only when they carry the event's own timestamp. Missing dates remain an
    # explicit coverage gap. PR state and aggregate commit/review counts are
    # deliberately not converted into events.
    if "events_in_window" not in data:
        for comment in data.get("comments_in_range", []):
            if not comment.get("is_bot"):
                event = {
                    "type": "jira_comment",
                    "occurred_at": comment.get("created_at") or comment.get("created") or comment.get("date"),
                    "author": comment.get("author_name"),
                    "text": comment.get("body"),
                }
                classify_event(event)
        for change in data.get("changelog_in_range", []):
            event = {
                "type": "jira_transition",
                "occurred_at": change.get("created_at") or change.get("date"),
                "author": change.get("author"),
                "changes": change.get("items", []),
            }
            classify_event(event)

    return pruned


def load_current_statuses(date_dir: Path, issue_keys: list) -> dict:
    """Read the previous week's filed status summary from each per-issue JSON file."""
    statuses: dict[str, str] = {}
    for key in issue_keys:
        path = date_dir / "issues" / f"{key}.json"
        try:
            with open(path) as f:
                data = json.load(f)
            statuses[key] = _field_to_text(data["issue"].get("current_status_summary") or "")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            statuses[key] = ""
    return statuses


# ─── Draft Persistence ───────────────────────────────────────────────────────

def _draft_state_dir(date_dir: Path) -> Path:
    """Return state storage scoped to the gathered output directory and date."""
    output_id = hashlib.sha256(
        str(date_dir.parent.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    return _get_state_dir() / "drafts" / output_id / date_dir.name


def _drafts_path(date_dir: Path) -> Path:
    return _draft_state_dir(date_dir) / "drafts.json"


def _draft_inputs_path(date_dir: Path) -> Path:
    return _draft_state_dir(date_dir) / "draft_inputs.json"


def _legacy_draft_path(date_dir: Path, filename: str) -> Path:
    """Path used before draft state moved out of the gathered data directory."""
    return date_dir / filename


def _input_fingerprint(summary: str) -> str:
    return hashlib.sha256(summary.encode("utf-8")).hexdigest()


def _load_draft_inputs(date_dir: Path) -> dict:
    path = _draft_inputs_path(date_dir)
    if not path.exists():
        path = _legacy_draft_path(date_dir, "draft_inputs.json")
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_draft_input(date_dir: Path, key: str, summary: str) -> None:
    """Record which exact gathered input produced a draft."""
    inputs = _load_draft_inputs(date_dir)
    inputs[key] = _input_fingerprint(summary)
    path = _draft_inputs_path(date_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(inputs, f, indent=2)


def load_saved_drafts(date_dir: Path) -> dict:
    """Load previously saved drafts from disk, or empty dict."""
    path = _drafts_path(date_dir)
    if not path.exists():
        path = _legacy_draft_path(date_dir, "drafts.json")
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_drafts(date_dir: Path, drafts: dict) -> None:
    """Persist current drafts to disk."""
    path = _drafts_path(date_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(drafts, f, indent=2)


# ─── Batch Drafting ──────────────────────────────────────────────────────────

# Generic draft attempts and the base of the exponential backoff between them.
# Truncation retries use the doubled-budget helper in llm.py instead.
DRAFT_MAX_ATTEMPTS = 3
DRAFT_RETRY_BACKOFF_BASE = 2


async def batch_draft(
    client,
    model: str,
    system_prompt: str,
    date_dir: Path,
    issues: list,
    usage: TokenUsage,
    callback=None,
    only_significant: bool = False,
    manifest: Optional[dict] = None,
    force_redraft: bool = False,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple:
    """Return drafts, summaries, failed keys, and per-key outcomes.

    Outcomes are drafted, quiet_skip, missing_data, draft_failed, or invalid_draft.
    A truncated response is retried once with the output budget doubled.
    """
    saved = {} if force_redraft else load_saved_drafts(date_dir)
    saved_inputs = _load_draft_inputs(date_dir)
    sem = asyncio.Semaphore(10)

    async def draft_one(issue, idx):
        async with sem:
            key = issue["key"]
            path = date_dir / "issues" / f"{key}.json"
            try:
                with open(path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                if callback:
                    await callback(idx + 1, len(issues), f"Skipped {key} (no data file)")
                return key, None, None, "missing_data"

            if only_significant and not is_significant(data):
                if callback:
                    await callback(idx + 1, len(issues), f"Skipped {key} (not significant)")
                return key, None, None, "quiet_skip"

            try:
                summary = json.dumps(_prune_issue(data, manifest), indent=2)
            except Exception as e:
                msg = f"Failed {key} (prune): {e}"
                print(f"Warning: {msg}", file=sys.stderr)
                if callback:
                    await callback(idx + 1, len(issues), msg)
                return key, None, None, "draft_failed"

            if key in saved and saved_inputs.get(key) == _input_fingerprint(summary):
                _, cache_error = validate_draft(saved[key])
                if not cache_error:
                    if callback:
                        await callback(idx + 1, len(issues), f"Loaded {key} (cached draft)")
                    return key, saved[key], summary, "drafted"

            async def on_truncation_retry(cap):
                msg = f"Truncated {key}; retrying with max_tokens={cap}"
                print(f"Warning: {msg}", file=sys.stderr)
                if callback:
                    await callback(idx + 1, len(issues), msg)

            last_err = None
            for attempt in range(1, DRAFT_MAX_ATTEMPTS + 1):
                try:
                    draft = await call_with_truncation_retry(
                        lambda cap: draft_ryg(
                            client, model, system_prompt, summary, usage,
                            max_tokens=cap,
                        ),
                        max_tokens,
                        on_truncation_retry,
                    )
                    _, validation_error = validate_draft(draft)
                    if validation_error:
                        msg = f"Invalid draft for {key}: {validation_error}"
                        if callback:
                            await callback(idx + 1, len(issues), msg)
                        return key, None, summary, "invalid_draft"
                    if callback:
                        await callback(idx + 1, len(issues), f"Drafted {key}")
                    return key, draft, summary, "drafted"
                except TruncatedResponseError as e:
                    last_err = e
                    # A second truncation is terminal; the helper already used
                    # the one doubled-budget retry.
                    break
                except Exception as e:
                    last_err = e
                    if attempt < DRAFT_MAX_ATTEMPTS:
                        await asyncio.sleep(DRAFT_RETRY_BACKOFF_BASE ** (attempt - 1))

            msg = f"Failed {key} (draft, {attempt} attempts): {last_err}"
            print(f"Warning: {msg}", file=sys.stderr)
            if callback:
                await callback(idx + 1, len(issues), msg)
            return key, None, None, "draft_failed"

    tasks = [draft_one(issue, i) for i, issue in enumerate(issues)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    drafts: dict[str, str] = {}
    result_summaries: dict[str, str] = {}
    failed: list[str] = []
    outcomes: dict[str, str] = {}
    for r in raw_results:
        if isinstance(r, Exception):
            print(f"Warning: draft failed: {r}", file=sys.stderr)
            continue
        key, draft, summary, outcome = r
        outcomes[key] = outcome
        if draft:
            drafts[key] = draft
        elif outcome in ("missing_data", "draft_failed", "invalid_draft"):
            failed.append(key)
        if summary:
            result_summaries[key] = summary

    save_drafts(date_dir, drafts)
    draft_inputs = {
        key: _input_fingerprint(result_summaries[key])
        for key in drafts
        if key in result_summaries
    }
    with _draft_inputs_path(date_dir).open("w") as f:
        json.dump(draft_inputs, f, indent=2)
    return drafts, result_summaries, failed, outcomes
