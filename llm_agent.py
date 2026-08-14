"""
LLM-powered bug-triage agent on top of the rule-based tools in script.py.

Uses Gemini's OpenAI-compatible Chat Completions API with tool calling
(plain `requests` — no LangChain / SDKs). Gemini exposes the same
request/response shape OpenAI/Groq use, so this is a drop-in swap: only
GEMINI_API_URL / GEMINI_MODEL / the API key env var changed from the
previous Groq version — the whole tool-calling loop below is untouched.

Changes in this revision:
  1. Bounded per-round tool-result payload (Option B): every tool call the
     model requests is still actually executed and cached (so the model
     never has to re-derive anything), but if the cumulative size of results
     sent back in a single round would blow up the next prompt, the
     overflow results are replaced with a small "deferred" marker instead
     of being dropped from execution. This keeps every tool_call_id
     answered (required by the API) while bounding token growth per step.
  2. Proactive token throttling: after each call, actual usage is read from
     the API response and used to decide whether to pause before the next
     call, instead of only reacting after a 429 has already happened.
  3. Step-level checkpointing: (messages, tracking, tool_cache) are
     persisted to a JSON file after every step, and investigate_with_llm()
     can resume from a checkpoint instead of restarting from step 1.
  4. run_batch_investigations(): runs a whole list of stack traces back to
     back and reports efficiency stats across the batch — how many were
     answered cleanly, how many needed the "undiffed candidates" gate to
     kick in (i.e. the model had to be told to go back and regenerate/redo
     work before its answer was accepted), and how many hit the max_steps
     ceiling and were force-answered. Useful for a "X/N resolved cleanly,
     avg N steps, N needed regeneration" style summary.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from script import get_blame, get_diff, get_recent_commits, parse_stack_trace, rank_frames

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Gemini's OpenAI-compatible endpoint — same request/response shape as
# Groq's, so the rest of this file (message format, tool-calling loop,
# checkpointing) needed no changes, only the URL/model/key below.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# gemini-3.1-flash-lite: the cheapest/fastest Gemini model, generous free
# tier (higher RPM/RPD than Flash), good enough for tool-calling triage.
# Swap to "gemini-3.5-flash" if you want stronger reasoning at the cost of
# a lower free-tier request rate.
GEMINI_MODEL = "gemini-3.1-flash-lite"
# Tight cap to force faster conclusions and lower API usage per bug.
MAX_AGENT_STEPS = 12
MAX_TOOL_RESULT_CHARS = 1500
# How many of the most recent tool results to send in FULL to the API.
# Older tool results get replaced with a short marker before each call.
KEEP_LAST_N_TOOL_RESULTS = 3
# How many times we'll refuse a "final answer" that still has undiffed
# candidate commits pending, before giving up and accepting it anyway.
MAX_GATE_REMINDERS = 2
# Cap on how many chars of *fresh* tool-result content get sent back to the
# model in a single round. All requested tool calls are still executed and
# cached; results beyond this budget are replaced with a "deferred" marker
# instead of being withheld from execution. This is what actually bounds
# the token spike from a wide parallel tool_calls burst — not how many
# calls run, but how much gets stuffed into the next prompt.
MAX_TOOL_RESULTS_PER_ROUND_CHARS = 6000
# If a step's total token usage (from the API's own `usage` field) exceeds
# this, pause before the next call to let the TPM window clear, instead of
# waiting for a 429 to happen first.
TOKEN_THROTTLE_THRESHOLD = 8000
TOKEN_THROTTLE_SLEEP_S = 10.0
# Where step checkpoints get written by default (caller can override).
DEFAULT_CHECKPOINT_PATH = "agent_checkpoint.json"


def checkout_commit(repo_path: str, commit_sha: str) -> dict:
    try:
        result = subprocess.run(
            ["git", "checkout", "--force", commit_sha],
            cwd=repo_path, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()
        return {
            "ok": True,
            "checked_out": commit_sha,
            "HEAD": head,
            "stderr": (result.stderr or "").strip(),
        }
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"Git checkout failed: {(e.stderr or str(e)).strip()}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key from Google AI Studio "
            "(https://aistudio.google.com/apikey) and export it before running, e.g.\n"
            "  PowerShell:  $env:GEMINI_API_KEY = 'AIza...'\n"
            "  bash:        export GEMINI_API_KEY=AIza..."
        )
    return key


def _extract_key_term(error_message: str) -> Optional[str]:
    """Pull the single most diagnostic identifier out of an exception message,
    e.g. "'_FileInFile' object has no attribute 'fileno'" -> "fileno".
    Used to mechanically check whether a diff adds/removes that identifier.
    """
    if not error_message:
        return None
    m = re.search(r"no attribute ['\"]([^'\"]+)['\"]", error_message)
    if m:
        return m.group(1)
    m = re.search(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", error_message)
    if m:
        return m.group(1)
    return None


def _mechanical_pattern_check(diff_text: str, key_term: Optional[str]) -> dict:
    """Actually compute (not guess) whether key_term appears in added-only,
    removed-only, both, or neither lines of a unified diff.
    """
    if not key_term:
        return {"verdict": "unclear", "added_hits": 0, "removed_hits": 0}
    added_hits = 0
    removed_hits = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") and key_term in line:
            added_hits += 1
        elif line.startswith("-") and key_term in line:
            removed_hits += 1
    if added_hits and not removed_hits:
        verdict = "introduces"
    elif removed_hits and not added_hits:
        verdict = "removes"
    else:
        verdict = "unclear"
    return {"verdict": verdict, "added_hits": added_hits, "removed_hits": removed_hits}


def _is_known_commit(sha: str, known: set) -> bool:
    """Match abbreviated SHAs against full SHAs from prior tool results."""
    for k in known:
        if k.startswith(sha) or sha.startswith(k):
            return True
    return False


def _is_diffed(sha: str, diffed: set) -> bool:
    """SHAs from get_recent_commits are full-length; the model might pass an
    abbreviated form to get_diff. Match by prefix in either direction."""
    for d in diffed:
        if d.startswith(sha) or sha.startswith(d):
            return True
    return False


def _build_digest_message(tracking: Dict[str, Any]) -> Optional[dict]:
    """A persistent summary of every commit judged so far, plus any still-
    pending candidates. This survives even after the full diff text for an
    old commit has been trimmed out of the message history, so the model
    doesn't lose its own earlier judgments (or forget which candidates it
    still owes a diff to).
    """
    digests: Dict[str, str] = tracking.get("digests", {})
    known: set = tracking.get("known_commits", set())
    diffed: set = tracking.get("diffed_commits", set())
    pending = {s for s in known if not _is_diffed(s, diffed)}
    found_strong = tracking.get("found_strong_introducer", False)

    if not digests and not pending:
        return None

    lines = [
        "PERSISTENT INVESTIGATION DIGEST (the full diff text for older commits "
        "may have been trimmed out of the history above to save tokens — this "
        "digest is your memory of what you already judged; do not re-request "
        "a commit just because its full diff scrolled out of view):"
    ]
    for sha, digest in digests.items():
        lines.append(f"  - {digest}")

    if found_strong:
        lines.append("")
        lines.append(
            "NOTE: at least one diffed commit already returned verdict='introduces' "
            "(a strong, computed signal). Per the rules, you do not need to diff "
            "every remaining candidate before answering — if you're satisfied this "
            "is the introducer, give your final answer now."
        )

    if pending:
        lines.append("")
        lines.append(
            f"UNDIFFED CANDIDATES ({len(pending)}) — required before a final answer "
            f"ONLY IF no strong 'introduces' verdict has been found yet:"
        )
        for sha in sorted(pending):
            lines.append(f"  - {sha}")

    return {"role": "user", "content": "\n".join(lines)}


def _trim_old_tool_results(messages: List[dict], keep_last_n: int = KEEP_LAST_N_TOOL_RESULTS) -> List[dict]:
    """Return a COPY of messages where only the last N tool results keep their
    full content; older ones are replaced with a short marker. The original
    `messages` list (kept by the caller for the final answer / record) is
    left untouched — only the copy sent to the API is trimmed.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    trim_before = tool_indices[-keep_last_n] if len(tool_indices) > keep_last_n else -1
    trimmed = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i < trim_before:
            m = dict(m)
            m["content"] = "[older tool result trimmed to save tokens — already seen earlier in this conversation, do not re-request it]"
        trimmed.append(m)
    return trimmed


def call_llm(messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    print("\n" + "=" * 60)
    print("[call_llm] Sending request to Gemini...")
    print(f"  model={GEMINI_MODEL}  messages={len(messages)}  tools={len(tools or [])}")

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=90)
        for attempt in range(3):
            if resp.status_code != 429:
                break
            wait_s = 15
            try:
                err_body = resp.json()
                msg = err_body.get("error", {}).get("message", "")
                m = re.search(r"try again in ([\d.]+)s", msg, re.I)
                if m:
                    wait_s = max(5.0, float(m.group(1)) + 2.0)
            except Exception:
                pass
            print(f"  !! Rate limited (429). Waiting {wait_s:.1f}s then retry "
                  f"{attempt + 1}/3...")
            time.sleep(wait_s)
            resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        if choice.get("tool_calls"):
            names = [tc["function"]["name"] for tc in choice["tool_calls"]]
            print(f"  <- model requested tool_calls: {names}")
        else:
            preview = (choice.get("content") or "")[:120].replace("\n", " ")
            print(f"  <- model returned final-ish text: {preview!r}...")
        usage = data.get("usage") or {}
        if usage:
            print(f"  tokens: prompt={usage.get('prompt_tokens')} "
                  f"completion={usage.get('completion_tokens')} "
                  f"total={usage.get('total_tokens')}")
        return data
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else str(e)
        print(f"  !! HTTP error from Gemini: {e} | body={body[:500]}")
        return {"error": f"HTTPError: {e}", "body": body}
    except requests.RequestException as e:
        print(f"  !! Network error calling Gemini: {e}")
        return {"error": f"RequestException: {e}"}
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"  !! Unexpected response shape: {e}")
        return {"error": f"Bad response: {e}"}


TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_commits",
            "description": (
                "Lists recent commits that touched a file (file-level signal). "
                "Call with max_commits=10 first. You MUST call get_diff on each "
                "returned commit and judge whether it explains the crash before "
                "deciding it's a dead end. If none of the 10 look convincing, call "
                "this again with max_commits=30, then diff-check the new commits "
                "you haven't already checked. If still unconvincing, widen once "
                "more to max_commits=50 as your final widen — do not widen further "
                "than that. You may freely interleave this with get_blame at any "
                "point; there is no fixed order between commit-scanning and blame."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Absolute path to the local git repository."},
                    "file_path": {
                        "type": "string",
                        "description": "Repo-relative path to the crashing file (e.g. 'requests/utils.py'), NOT a site-packages path.",
                    },
                    "max_commits": {
                        "type": "integer",
                        "description": "How many recent commits to return. Start at 10. Widen to 30, then 50 only if earlier widenings did not turn up a convincing candidate after per-commit diff review. Never request more than 50.",
                    },
                },
                "required": ["repo_path", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_blame",
            "description": (
                "Run git blame on ONE exact line to find the commit SHA that last "
                "touched that line. Strongest line-exact signal for the introducer. "
                "You may call this before, after, or interleaved with "
                "get_recent_commits — use your own judgment about which signal to "
                "pursue first given the trace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Absolute path to the local git repository."},
                    "file_path": {"type": "string", "description": "Repo-relative path to the file."},
                    "line_number": {"type": "integer", "description": "1-based line number from the stack frame."},
                },
                "required": ["repo_path", "file_path", "line_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diff",
            "description": (
                "Show the diff for a commit SHA scoped to the crashing file. "
                "ALWAYS pass file_path (the repo-relative path under investigation) "
                "so you only see changes to that file, not unrelated docs/changelog. "
                "Use after blame or for each get_recent_commits candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Absolute path to the local git repository."},
                    "commit_sha": {"type": "string", "description": "Full or abbreviated git commit SHA from get_blame or get_recent_commits."},
                    "file_path": {
                        "type": "string",
                        "description": "Repo-relative path to the file being investigated (e.g. requests/utils.py). Required — scopes the diff to this file only.",
                    },
                },
                "required": ["repo_path", "commit_sha", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkout_commit",
            "description": (
                "Check out a commit in the repo (e.g. 'abc1234~1') so subsequent "
                "get_blame / get_recent_commits see that older tree. Use this ONLY "
                "after you have quoted specific diff evidence showing the current "
                "candidate repairs/removes the crashing call (see walk-past rules "
                "in the system prompt) — checkout candidate_sha~1, then get_blame "
                "again on the same file:line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Absolute path to the local git repository."},
                    "commit_sha": {"type": "string", "description": "Commit-ish to check out, e.g. a full SHA or 'SHA~1'."},
                },
                "required": ["repo_path", "commit_sha"],
            },
        },
    },
]


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {len(text) - limit} more chars]"


# Tool names whose results are safe/worth caching by exact-argument key.
# (checkout_commit is intentionally NOT cached — it has a side effect on the
# repo's working tree, so it must always actually run.)
_CACHEABLE_TOOLS = {"get_recent_commits", "get_blame", "get_diff"}
# After this many cache replays for the same tool+args, nudge the model to move on.
CACHE_REPEAT_WARN_AFTER = 2

_REPEAT_CALL_MESSAGE = (
    "You've already asked this exact question multiple times. Stop repeating it. "
    "Either call get_recent_commits with a higher max_commits to check new candidates, "
    "or give your final answer now based on what you already know."
)


def execute_tool(
    name: str,
    arguments: dict,
    cache: Optional[Dict[str, str]] = None,
    cache_repeat_counts: Optional[Dict[str, int]] = None,
    delivered: Optional[set] = None,
    tracking: Optional[Dict[str, Any]] = None,
    key_term: Optional[str] = None,
) -> str:
    print("\n" + "-" * 60)
    print(f"[execute_tool] LLM requested: {name}")
    print(f"  arguments: {json.dumps(arguments, ensure_ascii=False)}")

    cache_key = None
    if cache is not None and name in _CACHEABLE_TOOLS:
        cache_key = f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
        if cache_key in cache:
            prior_replays = (cache_repeat_counts or {}).get(cache_key, 0)
            if prior_replays >= CACHE_REPEAT_WARN_AFTER:
                warn = json.dumps({
                    "error": "REPEATED_TOOL_CALL",
                    "message": _REPEAT_CALL_MESSAGE,
                    "tool": name,
                    "arguments": arguments,
                    "times_served_from_cache": prior_replays,
                }, ensure_ascii=False)
                print(f"  !! [cache] repeat limit reached for {name} "
                      f"({prior_replays} prior cache replays) — returning stop message")
                return warn
            if cache_repeat_counts is not None:
                cache_repeat_counts[cache_key] = prior_replays + 1
            cached_payload = cache[cache_key]
            print(f"  [cache] HIT — returning cached result for {name} with "
                  f"identical arguments (replay {prior_replays + 1}/"
                  f"{CACHE_REPEAT_WARN_AFTER}, no new git/API work)")
            return cached_payload

    try:
        if name == "get_recent_commits":
            max_commits = int(arguments.get("max_commits", 10))
            result = get_recent_commits(
                repo_path=arguments["repo_path"],
                file_path=arguments["file_path"],
                max_commits=max_commits,
            )
            if tracking is not None and isinstance(result, list):
                for c in result:
                    sha = c.get("sha")
                    if sha:
                        tracking["known_commits"].add(sha)

        elif name == "get_blame":
            result = get_blame(
                repo_path=arguments["repo_path"],
                file_path=arguments["file_path"],
                line_number=int(arguments["line_number"]),
            )
            if tracking is not None and isinstance(result, dict) and "error" not in result:
                sha = result.get("sha")
                if sha:
                    tracking["known_commits"].add(sha)

        elif name == "get_diff":
            commit_sha_arg = arguments["commit_sha"]
            file_path_arg = arguments.get("file_path")

            if tracking is not None and not _is_known_commit(commit_sha_arg, tracking["known_commits"]):
                result = json.dumps({
                    "error": "UNKNOWN_COMMIT_SHA",
                    "message": (
                        f"Commit {commit_sha_arg!r} was never returned by get_blame or "
                        "get_recent_commits in this investigation. Do not invent SHAs — "
                        "only request get_diff for commits you have actually seen from "
                        "those tools."
                    ),
                    "requested_sha": commit_sha_arg,
                    "known_shas": sorted(tracking["known_commits"]),
                }, ensure_ascii=False)
                print(f"  !! rejected get_diff: SHA not in known_commits")
            else:
                result = get_diff(
                    repo_path=arguments["repo_path"],
                    commit_sha=commit_sha_arg,
                    file_path=file_path_arg,
                )
                if isinstance(result, str) and not result.startswith("Error:"):
                    check = _mechanical_pattern_check(result, key_term)
                    verdict = check["verdict"]

                    if tracking is not None:
                        tracking["diffed_commits"].add(commit_sha_arg)
                        if key_term:
                            digest = (
                                f"{commit_sha_arg[:12]} -> verdict={verdict} "
                                f"(term={key_term!r}, +{check['added_hits']}/-{check['removed_hits']})"
                            )
                        else:
                            digest = f"{commit_sha_arg[:12]} -> verdict=unclear (no key term extracted from error message)"
                        tracking["digests"][commit_sha_arg] = digest
                        if verdict == "introduces":
                            tracking["found_strong_introducer"] = True

                    result = (
                        result
                        + "\n\n--- MECHANICAL PATTERN CHECK (computed, not guessed) ---\n"
                        + f"key_term_from_error_message: {key_term!r}\n"
                        + f"verdict: {verdict}\n"
                        + f"lines_added_containing_term: {check['added_hits']}\n"
                        + f"lines_removed_containing_term: {check['removed_hits']}\n"
                        + "\n--- JUDGMENT CHECKLIST (follow strictly) ---\n"
                        + "If you reached this diff via BLAME on the crashing line:\n"
                        + "1. If this diff ADDS or EXPANDS the crashing call/path "
                        + "(e.g. adds o.fileno() / the AttributeError site) → "
                        + "this commit is the INTRODUCER. Do NOT walk past. Give final answer.\n"
                        + "2. Only walk past (checkout SHA~1 + blame again) if you can QUOTE "
                        + "the specific line(s) in THIS diff that repair/remove/guard the "
                        + "crashing call. A vague impression of 'looks like a fix' is NOT "
                        + "sufficient — if you cannot quote the specific repairing line(s), "
                        + "treat this commit as the introducer and stop here.\n"
                        + "3. Ignore the word 'fix' in the commit title for this decision.\n\n"
                        + "If you reached this diff via a get_recent_commits candidate:\n"
                        + "4. Judge on its own merits whether THIS specific change plausibly "
                        + "explains the error. If not convincing, move to the next candidate "
                        + "in the current list, or widen max_commits if you've exhausted it.\n"
                    )
                elif tracking is not None:
                    # Even a failed diff counts as "handled" so the gate doesn't
                    # loop forever on a bad/unreachable SHA.
                    tracking["diffed_commits"].add(commit_sha_arg)

        elif name == "checkout_commit":
            result = checkout_commit(
                repo_path=arguments["repo_path"],
                commit_sha=arguments["commit_sha"],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        if isinstance(result, str):
            payload = result
        else:
            payload = json.dumps(result, ensure_ascii=False, indent=2)

        payload = _truncate(payload)
        preview = payload[:300].replace("\n", " ")
        print(f"  result preview: {preview}...")
        print(f"  result length: {len(payload)} chars")

        if cache_key is not None:
            cache[cache_key] = payload

        return payload

    except Exception as e:
        err = json.dumps({"error": f"{type(e).__name__}: {e}"})
        print(f"  !! tool failed: {err}")
        if tracking is not None and name == "get_diff":
            tracking["diffed_commits"].add(arguments.get("commit_sha", ""))
        return err


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
# Persist (messages, tracking, tool_cache, and loop bookkeeping) to disk
# after every agent step, so a crash / network death mid-investigation loses
# at most one step of work, not the whole run. tracking's sets are converted
# to sorted lists for JSON and back to sets on load.

def _tracking_to_json(tracking: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "known_commits": sorted(tracking.get("known_commits", set())),
        "diffed_commits": sorted(tracking.get("diffed_commits", set())),
        "digests": tracking.get("digests", {}),
        "found_strong_introducer": tracking.get("found_strong_introducer", False),
    }


def _tracking_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "known_commits": set(data.get("known_commits", [])),
        "diffed_commits": set(data.get("diffed_commits", [])),
        "digests": data.get("digests", {}),
        "found_strong_introducer": data.get("found_strong_introducer", False),
    }


def save_checkpoint(
    path: str,
    step: int,
    messages: List[dict],
    tracking: Dict[str, Any],
    tool_cache: Dict[str, str],
    tool_delivered: set,
    cache_repeat_counts: Dict[str, int],
    gate_reminders_sent: int,
    key_term: Optional[str],
    repo_path: str,
    checked_out: Optional[dict],
) -> None:
    payload = {
        "step": step,
        "messages": messages,
        "tracking": _tracking_to_json(tracking),
        "tool_cache": tool_cache,
        "tool_delivered": sorted(tool_delivered),
        "cache_repeat_counts": cache_repeat_counts,
        "gate_reminders_sent": gate_reminders_sent,
        "key_term": key_term,
        "repo_path": repo_path,
        "checked_out": checked_out,
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)  # atomic on POSIX/NTFS: no half-written checkpoint on crash
        print(f"  [checkpoint] saved step={step} -> {path}")
    except Exception as e:
        print(f"  !! [checkpoint] failed to save: {type(e).__name__}: {e}")


def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["tracking"] = _tracking_from_json(data["tracking"])
        print(f"  [checkpoint] loaded step={data.get('step')} from {path}")
        return data
    except Exception as e:
        print(f"  !! [checkpoint] failed to load {path}: {type(e).__name__}: {e}")
        return None


SYSTEM_PROMPT = """You are a bug-triage investigator. You are given a parsed Python
stack trace and a local git repository path. The repository has ALREADY been
checked out to the pre-fix state (fix_commit~1) before you start. Your job is
to identify the commit that most likely INTRODUCED the crashing behavior
(the root-cause / introducing commit), NOT the commit that later fixed it.

You have these tools:
  - get_recent_commits: cheap file-level history (max_commits configurable)
  - get_blame: who last touched the EXACT crashing line
  - get_diff: what a candidate commit actually changed. Its result now
    INCLUDES a "MECHANICAL PATTERN CHECK" section — a computed (not guessed)
    signal showing whether the crash-related term was ADDED, REMOVED, or is
    UNCLEAR in this diff. Treat this as strong supporting evidence, not a
    replacement for reading the diff yourself.
  - checkout_commit: move HEAD to an older commit (e.g. candidate_sha~1)

If you call a tool with arguments you already used earlier in this
investigation, you will receive the same cached result again (no new git
work) — but only twice. After that, you'll get a REPEATED_TOOL_CALL
message telling you to widen your search or give your final answer.

STOP EARLY: if you already have a get_diff result that plausibly explains
the error (especially verdict="introduces" from the mechanical pattern check,
or blame + a diff that adds the crashing call), give your final answer NOW —
do not make further tool calls. Extra calls waste steps and will be blocked
if they duplicate work you already did.

Some tool results may come back marked "deferred": true. That means the call
was executed and cached, but its result was withheld from this round only to
keep the round's total size bounded — simply re-issue the exact same call
next turn and you'll get the real (cached, instant) result.

ENFORCED RULE — you cannot skip diff-checking: every commit SHA returned by
get_recent_commits or get_blame is tracked. If you try to give a final
answer while any of those SHAs have never been passed to get_diff, your
answer will be REJECTED and you'll be told exactly which SHAs are still
pending — UNLESS a diff has already come back with a computed
verdict="introduces" (a strong, mechanically-confirmed signal). Once that
happens, you do NOT need to diff every remaining candidate — settle on that
commit and answer. Don't keep widening/scanning after you already have a
confirmed introducer just to be thorough; that wastes calls for no benefit.

=============================================================================
USING THE MECHANICAL PATTERN CHECK
=============================================================================
Every get_diff result includes a computed verdict: "introduces", "removes",
or "unclear", based on whether the crash's key term (e.g. the missing
attribute name from an AttributeError) appears in added vs removed lines.

  - verdict="introduces": strong signal this commit is the INTRODUCER. You
    should generally settle on this commit unless you have a specific,
    quotable reason to distrust the mechanical check (e.g. the term appears
    in an unrelated comment, not real code).
  - verdict="removes": strong signal this commit REPAIRS the crash. This is
    exactly the kind of evidence needed to justify walking past (see below).
  - verdict="unclear": the mechanical check could not determine this reliably
    (e.g. term appears on both sides, or no clear term was found). Fall back
    to reading the diff yourself and reasoning as before — do not treat
    "unclear" as evidence of anything either way.

The mechanical check is a computed fact, not a guess — weight it more heavily
than your own general impression of the diff, but always state in your final
answer whether your conclusion agreed or disagreed with it, and why.

=============================================================================
YOU CHOOSE THE INVESTIGATION PATH
=============================================================================
There is no fixed order between "check recent commits" and "blame the exact
line." Use your own judgment based on the trace: if the crashing line looks
like something a specific recent commit would plausibly have touched, start
with get_recent_commits. If you'd rather pin down exactly who last changed
that line, start with get_blame. You may switch between the two paths at any
point, and you may interleave them.

PRACTICAL TIP: get_blame is a single, cheap call that often finds the
introducer directly and lets you exit the strong-signal gate above with far
fewer tool calls than exhaustively scanning+diffing 30-50 commits. If your
first 10 commit-scan candidates aren't convincing, consider trying get_blame
before widening to max_commits=30/50 — widening is expensive (many diffs to
review) and should be a fallback, not the default next move.

=============================================================================
PATH 1 — COMMIT-SCANNING WITH PROGRESSIVE WIDENING
=============================================================================
STEP 1 — NARROW FIRST:
  Call get_recent_commits(repo_path, file_path, max_commits=10).

STEP 2 — PER-COMMIT DIFF JUDGMENT (required, do not skip):
  For EACH commit returned, call get_diff(repo_path, commit_sha). Check the
  mechanical pattern check verdict FIRST, then confirm or challenge it by
  reading the surrounding diff context. Do not treat the raw commit list
  (messages/dates alone) as sufficient evidence — inspect diff content (and
  the pattern verdict) for every commit before ruling it in or out.

STEP 3 — WIDEN IF UNCONVINCING:
  If none of the 10 commits show verdict="introduces" (or a convincing
  "unclear" case on manual reading), call get_recent_commits again with
  max_commits=30, diff-check new commits only (you already have diffs for
  the first 10 — do not re-request them, the cache will just hand you back
  what you already saw).

STEP 4 — FINAL WIDEN:
  If still unconvincing after 30, widen ONE more time to max_commits=50. Do
  NOT widen past 50. If nothing convincing turns up, report your best
  available guess (lowest confidence), noting commit-scanning was inconclusive.

=============================================================================
PATH 2 — BLAME + WALK-PAST PARTIAL FIXES (STRICT EVIDENCE REQUIRED)
=============================================================================
STEP A — BLAME THE LINE:
  Call get_blame(repo_path, file_path, line_number) on the crashing line.
  That SHA is your current candidate.

STEP B — INSPECT THE DIFF (required after every blame candidate):
  Call get_diff(repo_path, candidate_sha). Check the mechanical pattern
  check verdict first:
    * verdict="introduces" → treat as the INTRODUCER by default. Do NOT
      walk past unless you have specific, quotable evidence overriding this
      (rare — explain clearly if you do).
    * verdict="removes" → this is exactly the evidence needed to walk past
      (see STEP C).
    * verdict="unclear" → fall back to manual reading: does the diff tighten
      conditions, add guards, revert bad behavior (looks like a fix)? Or does
      it add the crashing logic itself (looks like the introducer)?

STEP C — WALK PAST A PARTIAL FIX (default: AT MOST 1 walk, not 3):
  The default budget for walking past a candidate is exactly ONE walk. Do
  NOT walk past a candidate on a general impression that it "looks like a
  fix." Before calling checkout_commit to walk past, you MUST, in your
  reasoning, either (a) cite verdict="removes" from the mechanical pattern
  check, or (b) quote the specific line(s)/hunk in THIS diff that repair,
  remove, or guard the exact crashing call. If you cannot point to either,
  DO NOT walk past — treat the current candidate as the introducer and give
  your final answer now.

  If, after walking past once, the NEW candidate's diff ALSO appears to be a
  fix: you may attempt a second walk ONLY if you can again cite verdict=
  "removes" or quote specific repairing evidence, AND you explicitly state
  in your reasoning why a second consecutive fix-on-fix pattern is plausible
  here rather than a misjudgment. Do not exceed 2 total walks under any
  circumstances, and treat needing a second walk as a signal to lower your
  final confidence rating, not raise it.

  CRITICAL — do NOT walk past a commit merely because its subject contains
  "fix"/"bug". Many introducing commits are titled as fixes of something else
  but ADD the crashing logic. If the mechanical check says "introduces" or
  the diff otherwise INTRODUCES/expands the crashing code path, that commit
  IS the introducer — settle on it, even if its title says "fix".

Do NOT pick a commit just because its message says "fix"/"bug" when naming
the INTRODUCER — those words often mark resolving commits. Use the mechanical
pattern check and diff judgment above instead. This applies to both paths.

When you have enough evidence, stop calling tools and reply with:
  - root_cause_commit: the full SHA
  - why: 3-5 sentences. Quote or closely paraphrase the actual changed
    line(s). State explicitly whether the mechanical pattern check verdict
    agreed with your conclusion, and if you overrode it, explain why.
  - path_used: "blame", "commit_scan", or "both" (say which, and if both, the
    order you actually used)
  - walked_past_fixes: true/false and how many times (0, 1, or 2 max). If
    true, cite the mechanical verdict or quote the specific evidence that
    justified each walk.
  - widened_commit_scan: true/false and to what max_commits, if commit
    scanning was used
  - confidence: low / medium / high (lower this if you needed 2 walks, if
    the mechanical check was "unclear", or if your evidence was borderline)

Use the repo_path from the user message for every tool call.
Use repo-relative file paths (e.g. requests/utils.py), not site-packages paths.
"""


def investigate_with_llm(
    exception_type: str,
    error_message: str,
    file_path: str,
    line_number: int,
    repo_path: str,
    target_commit: Optional[str] = None,
    max_steps: int = MAX_AGENT_STEPS,
    checkpoint_path: Optional[str] = DEFAULT_CHECKPOINT_PATH,
    resume: bool = False,
) -> dict:
    """
    checkpoint_path: where step-level progress is saved after every step.
        Pass None to disable checkpointing entirely.
    resume: if True and a checkpoint exists at checkpoint_path, resume the
        investigation from there instead of starting over (skips the
        pre-fix checkout, since the repo was already left in the right
        state by the previous run).
    """
    checkout_info = None
    messages: List[dict] = []
    tool_cache: Dict[str, str] = {}
    tool_delivered: set = set()
    cache_repeat_counts: Dict[str, int] = {}
    tracking: Dict[str, Any] = {
        "known_commits": set(),
        "diffed_commits": set(),
        "digests": {},
        "found_strong_introducer": False,
    }
    gate_reminders_sent = 0
    key_term = _extract_key_term(error_message)
    start_step = 1

    if resume and checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            messages = ckpt["messages"]
            tracking = ckpt["tracking"]
            tool_cache = ckpt["tool_cache"]
            tool_delivered = set(ckpt.get("tool_delivered", []))
            cache_repeat_counts = ckpt.get("cache_repeat_counts", {})
            gate_reminders_sent = ckpt["gate_reminders_sent"]
            key_term = ckpt["key_term"]
            checkout_info = ckpt["checked_out"]
            start_step = ckpt["step"] + 1
            print(f"\n  [resume] continuing from step {start_step} "
                  f"(repo assumed already at prior checked-out state)")

    if start_step == 1:
        if target_commit:
            print("\n" + "#" * 60)
            print("# PRE-FIX CHECKOUT (before LLM loop)")
            print("#" * 60)
            print(f"  Checking out target_commit={target_commit}")
            print(f"  repo_path={repo_path}")
            checkout_info = checkout_commit(repo_path, target_commit)
            print(f"  result: {json.dumps(checkout_info, ensure_ascii=False)}")
            if not checkout_info.get("ok"):
                return {
                    "error": f"Failed to checkout target_commit {target_commit}: "
                             f"{checkout_info.get('error')}",
                    "steps_used": 0,
                    "messages": [],
                    "final_answer": None,
                    "checked_out": checkout_info,
                }
            print(f"  HEAD is now {checkout_info.get('HEAD')}")

        print(f"  key_term (for mechanical pattern check): {key_term!r}")

        user_payload = {
            "exception_type": exception_type,
            "error_message": error_message,
            "file_path": file_path,
            "line_number": line_number,
            "repo_path": repo_path,
            "target_commit_checked_out": target_commit,
            "reminder": (
                "You may start with either get_recent_commits(max_commits=10) or "
                "get_blame — your choice. If you scan commits, diff-check each one "
                "and widen to 30 then 50 only if unconvincing. If you blame, walk "
                "past a candidate ONLY if you can quote specific repairing evidence "
                "in its diff (default budget: 1 walk, max 2 with strong "
                "justification). You can switch paths at any point. Duplicate "
                "tool calls return the cached result instantly. "
                "If a diff already plausibly explains the crash, stop and answer."
            ),
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Investigate this crash and find the introducing commit.\n\n"
                    + json.dumps(user_payload, indent=2)
                ),
            },
        ]

    print("\n" + "#" * 60)
    print("# LLM BUG-TRIAGE AGENT STARTING" if start_step == 1 else "# LLM BUG-TRIAGE AGENT RESUMING")
    print("#" * 60)
    print(f"  exception : {exception_type}: {error_message}")
    print(f"  location  : {file_path}:{line_number}")
    print(f"  repo      : {repo_path}")
    print(f"  target    : {target_commit}")
    print(f"  max_steps : {max_steps}")

    for step in range(start_step, max_steps + 1):
        print(f"\n>>> AGENT STEP {step}/{max_steps}")

        # Send a TRIMMED copy of the history to the API (older tool results
        # collapsed to a short marker) to keep token usage bounded as the
        # conversation grows. The full, untrimmed `messages` list is still
        # kept for the final record / return value.
        outgoing = _trim_old_tool_results(messages, keep_last_n=KEEP_LAST_N_TOOL_RESULTS)
        digest_msg = _build_digest_message(tracking)
        if digest_msg:
            outgoing = outgoing + [digest_msg]
        data = call_llm(outgoing, tools=TOOLS)
        if "error" in data and "choices" not in data:
            if checkpoint_path:
                save_checkpoint(checkpoint_path, step - 1, messages, tracking, tool_cache,
                                 tool_delivered, cache_repeat_counts, gate_reminders_sent,
                                 key_term, repo_path, checkout_info)
            return {
                "error": data["error"],
                "steps_used": step,
                "messages": messages,
                "final_answer": None,
                "checked_out": checkout_info,
            }

        # --- Proactive token throttle: read *actual* usage from this call
        # and pause before the next one if it was heavy, instead of only
        # reacting after a 429 has already fired.
        usage = data.get("usage") or {}
        step_tokens = usage.get("total_tokens", 0)
        if step_tokens > TOKEN_THROTTLE_THRESHOLD:
            print(f"  [throttle] step used {step_tokens} tokens (> {TOKEN_THROTTLE_THRESHOLD}); "
                  f"pausing {TOKEN_THROTTLE_SLEEP_S:.0f}s to let the TPM window clear")
            time.sleep(TOKEN_THROTTLE_SLEEP_S)

        assistant_msg = data["choices"][0]["message"]
        normalized = {
            "role": "assistant",
            "content": assistant_msg.get("content"),
        }
        if assistant_msg.get("tool_calls"):
            normalized["tool_calls"] = assistant_msg["tool_calls"]
        messages.append(normalized)

        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            pending = {
                s for s in tracking["known_commits"]
                if not _is_diffed(s, tracking["diffed_commits"])
            }
            gate_needed = pending and not tracking.get("found_strong_introducer", False)
            if gate_needed and gate_reminders_sent < MAX_GATE_REMINDERS:
                gate_reminders_sent += 1
                print(f"\n  !! GATE: rejecting final answer — {len(pending)} candidate "
                      f"commit(s) were never diffed: {sorted(pending)} "
                      f"(reminder {gate_reminders_sent}/{MAX_GATE_REMINDERS})")
                messages.append({
                    "role": "user",
                    "content": (
                        f"I can't accept that as a final answer yet: {len(pending)} candidate "
                        f"commit(s) from get_recent_commits/get_blame have never been checked "
                        f"with get_diff — {', '.join(sorted(pending))}. Call get_diff on each of "
                        f"these now and judge them before answering again."
                    ),
                })
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, step, messages, tracking, tool_cache,
                                     tool_delivered, cache_repeat_counts, gate_reminders_sent,
                                     key_term, repo_path, checkout_info)
                continue

            final_text = assistant_msg.get("content") or ""
            print("\n" + "#" * 60)
            print("# FINAL ANSWER (no more tool calls)")
            print("#" * 60)
            if pending:
                reason = (
                    "a strong 'introduces' verdict was already found, so remaining "
                    "candidates were skipped" if tracking.get("found_strong_introducer")
                    else f"accepted after {MAX_GATE_REMINDERS} reminders"
                )
                print(f"  !! NOTE: {len(pending)} candidate(s) still undiffed ({reason}): {sorted(pending)}")
            print(final_text)
            if checkpoint_path:
                save_checkpoint(checkpoint_path, step, messages, tracking, tool_cache,
                                 tool_delivered, cache_repeat_counts, gate_reminders_sent,
                                 key_term, repo_path, checkout_info)
            return {
                "final_answer": final_text,
                "steps_used": step,
                "messages": messages,
                "error": None,
                "checked_out": checkout_info,
                "undiffed_candidates_remaining": sorted(pending) if pending else [],
                "gate_reminders_sent": gate_reminders_sent,
                "hit_max_steps": False,
            }

        # --- Execute every requested tool call (populates cache + tracking
        # for all of them, regardless of round size), but bound how much
        # fresh content gets sent back to the model in THIS round. Anything
        # beyond the budget is answered with a small "deferred" marker
        # instead of the full payload — the tool_call_id is still answered
        # (required by the API), and the real result is already cached for
        # instant retrieval next turn.
        round_bytes_sent = 0
        for tc in tool_calls:
            tc_id = tc["id"]
            fn = tc["function"]
            name = fn["name"]

            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError as e:
                tool_result = json.dumps(
                    {"error": f"Could not parse tool arguments JSON: {e}",
                     "raw": fn.get("arguments")}
                )
                print(f"  !! bad arguments JSON for {name}: {e}")
            else:
                args["repo_path"] = repo_path
                tool_result = execute_tool(
                    name, args, cache=tool_cache, cache_repeat_counts=cache_repeat_counts,
                    delivered=tool_delivered, tracking=tracking, key_term=key_term
                )

            if round_bytes_sent + len(tool_result) > MAX_TOOL_RESULTS_PER_ROUND_CHARS and round_bytes_sent > 0:
                print(f"  [round-budget] withholding full result for {name} "
                      f"(would push round past {MAX_TOOL_RESULTS_PER_ROUND_CHARS} chars); "
                      f"already cached, deferring to next turn")
                sent_content = json.dumps({
                    "deferred": True,
                    "reason": (
                        f"Computed and cached, but withheld from this round to bound "
                        f"token usage (round budget {MAX_TOOL_RESULTS_PER_ROUND_CHARS} chars). "
                        f"Re-issue this exact call next turn to receive the full result."
                    ),
                })
            else:
                sent_content = tool_result
                round_bytes_sent += len(tool_result)
                if name in _CACHEABLE_TOOLS:
                    ck = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                    tool_delivered.add(ck)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": sent_content,
            })
            print(f"  [history] appended tool result for call id={tc_id} "
                  f"({'deferred' if sent_content is not tool_result else 'full'})")

        if checkpoint_path:
            save_checkpoint(checkpoint_path, step, messages, tracking, tool_cache,
                             tool_delivered, cache_repeat_counts, gate_reminders_sent,
                             key_term, repo_path, checkout_info)

    print("\n!! Reached max_steps without a final text answer.")
    messages.append({
        "role": "user",
        "content": (
            "You have reached the investigation step limit. "
            "Based on the tool results so far, give your best final answer now: "
            "root_cause_commit, why, path_used, walked_past_fixes, "
            "widened_commit_scan, and confidence. "
            "Do not call any more tools."
        ),
    })
    outgoing = _trim_old_tool_results(messages, keep_last_n=KEEP_LAST_N_TOOL_RESULTS)
    digest_msg = _build_digest_message(tracking)
    if digest_msg:
        outgoing = outgoing + [digest_msg]
    data = call_llm(outgoing, tools=None)
    final_text = None
    if "choices" in data:
        final_text = data["choices"][0]["message"].get("content")
        messages.append({"role": "assistant", "content": final_text})
        print("\n# FORCED FINAL ANSWER AFTER MAX STEPS")
        print(final_text)

    if checkpoint_path:
        save_checkpoint(checkpoint_path, max_steps, messages, tracking, tool_cache,
                         tool_delivered, cache_repeat_counts, gate_reminders_sent,
                         key_term, repo_path, checkout_info)

    return {
        "final_answer": final_text,
        "steps_used": max_steps,
        "messages": messages,
        "error": "max_steps_reached",
        "checked_out": checkout_info,
        "gate_reminders_sent": gate_reminders_sent,
        "hit_max_steps": True,
    }


def _prepare_frame(stack_trace: str, repo_path: str) -> dict:
    parsed = parse_stack_trace(stack_trace)
    if "error" in parsed:
        return parsed

    frames = parsed.get("frames", [])
    # rank_frames remaps venv site-packages paths to repo-relative paths when
    # the file exists under repo_path — works for requests, flask, fastapi, etc.
    ranked = rank_frames(frames, repo_path)
    return {
        "exception_type": parsed.get("exception_type"),
        "error_message": parsed.get("error_message"),
        "frames": frames,
        "ranked": ranked,
    }


def run_batch_investigations(
    cases: List[Dict[str, Any]],
    max_steps: int = MAX_AGENT_STEPS,
    checkpoint_dir: Optional[str] = "batch_checkpoints",
    resume: bool = False,
    pause_between_cases_s: float = 0.0,
) -> Dict[str, Any]:
    """Run investigate_with_llm() over a whole list of stack traces and
    report how efficiently the model handled the batch.

    Each item in `cases` is a dict:
        {
          "name": "case_01",                 # used for the checkpoint filename
          "stack_trace": "<full traceback text>",
          "repo_path": r"C:\\path\\to\\repo",
          "target_commit": "<fix_sha>~1",    # optional, same as investigate_with_llm
        }

    Returns a dict with a "results" list (one entry per case, in order) and
    a "summary" dict with the efficiency numbers you'd want for a report:
        - total: how many cases were run
        - resolved_cleanly: answered without ever hitting the "undiffed
          candidates" gate (i.e. never had to be told to go back and
          regenerate/redo work before its answer was accepted) and without
          hitting max_steps
        - needed_regeneration: cases where the gate fired at least once
          (model tried to answer early, got rejected, had to redo work)
        - hit_step_limit: cases that never produced a voluntary final answer
          and were force-answered at max_steps
        - errored: cases that failed outright (network/API error)
        - avg_steps_used / min_steps_used / max_steps_used: over cases that
          completed (errored cases excluded)
    """
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        name = case.get("name") or f"case_{i:02d}"
        print("\n" + "=" * 70)
        print(f"BATCH CASE {i}/{len(cases)}: {name}")
        print("=" * 70)

        prepared = _prepare_frame(case["stack_trace"], case["repo_path"])
        if not prepared.get("ranked"):
            results.append({
                "name": name,
                "error": prepared.get("error", "no ranked frames found"),
                "steps_used": None,
                "gate_reminders_sent": None,
                "hit_max_steps": None,
                "final_answer": None,
            })
            continue

        top = prepared["ranked"][0]
        case_checkpoint = (
            os.path.join(checkpoint_dir, f"{name}.json") if checkpoint_dir else None
        )

        outcome = investigate_with_llm(
            exception_type=prepared["exception_type"],
            error_message=prepared["error_message"],
            file_path=top["file_path"],
            line_number=top["line_number"],
            repo_path=case["repo_path"],
            target_commit=case.get("target_commit"),
            max_steps=max_steps,
            checkpoint_path=case_checkpoint,
            resume=resume,
        )
        outcome["name"] = name
        results.append(outcome)

        if pause_between_cases_s:
            print(f"  [batch] pausing {pause_between_cases_s:.0f}s before next case "
                  f"(rate-limit headroom)")
            time.sleep(pause_between_cases_s)

    errored = [r for r in results if r.get("final_answer") is None and not r.get("hit_max_steps")]
    hit_limit = [r for r in results if r.get("hit_max_steps")]
    needed_regen = [r for r in results if (r.get("gate_reminders_sent") or 0) > 0]
    resolved_cleanly = [
        r for r in results
        if r.get("final_answer") is not None
        and not r.get("hit_max_steps")
        and (r.get("gate_reminders_sent") or 0) == 0
    ]
    steps_list = [r["steps_used"] for r in results if isinstance(r.get("steps_used"), int)]

    summary = {
        "total": len(results),
        "resolved_cleanly": len(resolved_cleanly),
        "needed_regeneration": len(needed_regen),
        "hit_step_limit": len(hit_limit),
        "errored": len(errored),
        "avg_steps_used": round(sum(steps_list) / len(steps_list), 2) if steps_list else None,
        "min_steps_used": min(steps_list) if steps_list else None,
        "max_steps_used": max(steps_list) if steps_list else None,
    }

    print("\n" + "#" * 70)
    print("# BATCH SUMMARY")
    print("#" * 70)
    print(json.dumps(summary, indent=2))

    return {"results": results, "summary": summary}


DEFAULT_STACK_TRACE = """
Traceback (most recent call last):
  File "./demo.py", line 7, in <module>
    print(f"super_length of member is {requests.utils.super_len(member)}")
  File "/data/developer/.virtualenvs/req-tar/lib/python3.6/site-packages/requests/utils.py", line 119, in super_len
    fileno = o.fileno()
AttributeError: '_FileInFile' object has no attribute 'fileno'
"""

DEFAULT_REPO_PATH = r"C:\Users\HP\requests"
DEFAULT_FIX_COMMIT = "2d2447e210cf0b9e8c7484bfc6f158de9b24c171"


def _parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the LLM bug-triage agent against a local git repo.",
    )
    parser.add_argument(
        "--repo-path",
        default=os.environ.get("BUG_TRIAGE_REPO_PATH", DEFAULT_REPO_PATH),
        help="Absolute path to the git repo under investigation "
             "(default: BUG_TRIAGE_REPO_PATH env or requests demo path)",
    )
    parser.add_argument(
        "--fix-commit",
        default=os.environ.get("BUG_TRIAGE_FIX_COMMIT", DEFAULT_FIX_COMMIT),
        help="SHA of the commit that fixed the bug; agent checks out fix~1 "
             "(default: BUG_TRIAGE_FIX_COMMIT env or requests demo SHA)",
    )
    parser.add_argument(
        "--stack-trace",
        default=None,
        help="Full traceback text (overrides --stack-trace-file if both given)",
    )
    parser.add_argument(
        "--stack-trace-file",
        default=None,
        help="Path to a file containing the traceback",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Max LLM loop iterations (default: {MAX_AGENT_STEPS})",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Checkpoint file path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable step checkpointing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint instead of starting over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_cli_args()
    repo_path = os.path.abspath(args.repo_path)
    target_commit = f"{args.fix_commit}~1"

    if args.stack_trace:
        stack_trace = args.stack_trace
    elif args.stack_trace_file:
        with open(args.stack_trace_file, encoding="utf-8") as f:
            stack_trace = f.read()
    else:
        stack_trace = DEFAULT_STACK_TRACE

    prepared = _prepare_frame(stack_trace, repo_path)
    if not prepared.get("ranked"):
        print("No ranked frames; aborting.")
        sys.exit(1)

    top = prepared["ranked"][0]
    print(f"repo_path      : {repo_path}")
    print(f"fix_commit     : {args.fix_commit}")
    print(f"target_commit  : {target_commit}")
    print(f"Using top ranked frame: {top['file_path']}:{top['line_number']}")
    print("(checkout happens INSIDE investigate_with_llm, before the LLM loop)")

    resume_run = args.resume or os.environ.get("AGENT_RESUME", "").strip() == "1"
    checkpoint_path = None if args.no_checkpoint else args.checkpoint

    result = investigate_with_llm(
        exception_type=prepared["exception_type"],
        error_message=prepared["error_message"],
        file_path=top["file_path"],
        line_number=top["line_number"],
        repo_path=repo_path,
        target_commit=target_commit,
        max_steps=args.max_steps,
        checkpoint_path=checkpoint_path,
        resume=resume_run,
    )

    print("\n" + "=" * 60)
    print("AGENT RESULT SUMMARY")
    print("=" * 60)
    print(f"checked_out: {result.get('checked_out')}")
    print(f"steps_used : {result.get('steps_used')}")
    print(f"error      : {result.get('error')}")
    print(f"answer     :\n{result.get('final_answer')}")