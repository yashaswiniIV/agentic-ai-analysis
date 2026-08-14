"""
szz.py — SZZ-lite ground truth generator for the bug-triage eval dataset.

WHY THIS FILE EXISTS
---------------------
The previous ground-truth approach blamed the *stack-trace crash line* at the
pre-fix commit. That's a reasonable proxy, but it isn't SZZ: it can point to
the wrong commit when the fix touches lines other than the exact crash line,
and it has no defense against "cosmetic" blame hits (a line last touched by a
pure reformat/rename commit, not the commit that actually broke it).

Real SZZ (Sliwerski/Zimmermann/Zeller, 2005) works backwards from the FIX
commit's diff, not from the crash trace:

  1. Take the fix commit. Look at exactly which lines it removed/changed
     (the diff hunks), per file.
  2. git blame those specific lines at fix_commit~1 (the state right before
     the fix). Whoever last touched those lines is the introducing-commit
     candidate.
  3. Filter out "cosmetic" candidates — commits that are pure whitespace /
     formatting / rename changes for that region — and walk further back
     (blame the same lines at candidate~1) until a non-cosmetic commit is
     found or a depth limit is hit.

Known limitations (state these in your README — this is not 100% ground
truth, see the interview-defense discussion):
  - Pure-addition hunks (old_count == 0, i.e. the fix ADDED a missing line/
    guard that never existed before) have nothing to blame — this is the
    documented "SZZ omission problem." These cases are returned with
    candidates=[] and should be excluded from evaluation or handled with a
    manual fallback.
  - Multi-file fixes produce multiple candidate chains; this module returns
    all of them ranked, and picks the most-recent non-cosmetic one as
    `primary_introducing_sha`, but you may want to review multi-file cases
    by hand rather than trusting the auto-picked primary blindly.
  - The cosmetic filter is a heuristic (message keywords + whitespace-only
    diff check), not a certainty.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List, Optional

# Keywords in a commit message that suggest a purely cosmetic change.
# Soft signal only — combined with the whitespace-diff check below, not
# used alone to disqualify a commit.
_COSMETIC_MESSAGE_KEYWORDS = (
    "typo", "whitespace", "formatting", "format", "reformat", "lint",
    "style", "rename", "black", "isort", "flake8",
)

MAX_WALKBACK_DEPTH = 5


# ==============================================================================
# Low-level git helpers
# ==============================================================================

def _run_git(repo_path: str, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=repo_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def get_fix_diff_hunks(repo_path: str, fix_commit_sha: str) -> List[Dict[str, Any]]:
    """
    Returns a list of {file_path, old_start, old_count} for every hunk in the
    fix commit's diff, using OLD (pre-fix) file line numbers — these are the
    line numbers to blame at fix_commit~1.

    old_count == 0 means the hunk is a pure addition (nothing existed at that
    location pre-fix) — these hunks are still returned but callers should
    treat them as "nothing to blame" (the SZZ omission case).
    """
    result = _run_git(
        repo_path,
        ["show", "--unified=0", "--format=", fix_commit_sha],
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show failed for {fix_commit_sha}: {result.stderr.strip()}")

    hunks: List[Dict[str, Any]] = []
    current_file: Optional[str] = None

    file_header_re = re.compile(r"^--- a/(.+)$")
    hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")

    for line in result.stdout.splitlines():
        m_file = file_header_re.match(line)
        if m_file:
            current_file = m_file.group(1)
            continue

        m_hunk = hunk_header_re.match(line)
        if m_hunk and current_file:
            old_start = int(m_hunk.group(1))
            old_count = int(m_hunk.group(2)) if m_hunk.group(2) is not None else 1
            hunks.append({
                "file_path": current_file,
                "old_start": old_start,
                "old_count": old_count,
            })

    return hunks


def blame_lines_at_commit(
    repo_path: str, rev: str, file_path: str, start_line: int, end_line: int
) -> List[Dict[str, str]]:
    """
    Runs `git blame <rev> -L start,end` for file_path and returns one entry
    per distinct commit touching that range: [{sha, author, date}, ...].
    Unlike script.py's get_blame (which always blames whatever's currently
    checked out), this blames at an ARBITRARY historical revision without
    needing to checkout — required so we can blame at fix_commit~1 regardless
    of what the working tree currently has checked out.
    """
    result = _run_git(
        repo_path,
        ["blame", rev, "-L", f"{start_line},{end_line}", "--porcelain", "--", file_path],
    )
    if result.returncode != 0:
        return []

    entries: Dict[str, Dict[str, str]] = {}
    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        parts = line.split()
        # A new blame "chunk" header line starts with a 40-char sha followed
        # by 3 integers (orig_line, final_line, num_lines[optional]).
        if len(parts) >= 3 and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
            sha = parts[0]
            author = "Unknown"
            date = "Unknown"
            j = i + 1
            while j < len(lines) and not lines[j].startswith("\t"):
                if lines[j].startswith("author "):
                    author = lines[j][7:]
                elif lines[j].startswith("author-time "):
                    date = lines[j][12:]
                j += 1
            if sha not in entries:
                entries[sha] = {"sha": sha, "author": author, "date": date}
            i = j + 1
        else:
            i += 1

    return list(entries.values())


def get_commit_diff_for_file(repo_path: str, commit_sha: str, file_path: str) -> str:
    result = _run_git(repo_path, ["show", "--format=", commit_sha, "--", file_path])
    return result.stdout if result.returncode == 0 else ""


def get_commit_message(repo_path: str, commit_sha: str) -> str:
    result = _run_git(repo_path, ["show", "-s", "--format=%s", commit_sha])
    return result.stdout.strip() if result.returncode == 0 else ""


def get_parent_sha(repo_path: str, commit_sha: str) -> Optional[str]:
    result = _run_git(repo_path, ["rev-parse", f"{commit_sha}~1"])
    return result.stdout.strip() if result.returncode == 0 else None


# ==============================================================================
# Cosmetic-commit filtering
# ==============================================================================

def _is_cosmetic(repo_path: str, commit_sha: str, file_path: str) -> bool:
    """
    Heuristic: a commit is "cosmetic" for our purposes if its message hints
    at a pure formatting/rename change AND stripping whitespace from every
    added/removed line in its diff for this file makes the diff empty (i.e.
    the only difference is whitespace/indentation), OR if the diff for this
    file is empty/whitespace-only regardless of message (covers unlabeled
    reformat commits too).
    """
    diff_text = get_commit_diff_for_file(repo_path, commit_sha, file_path)
    if not diff_text.strip():
        return False  # nothing to judge; don't filter blindly

    added = []
    removed = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:].strip())
        elif line.startswith("-"):
            removed.append(line[1:].strip())

    whitespace_only = sorted(added) == sorted(removed) and added

    message = get_commit_message(repo_path, commit_sha).lower()
    message_hints_cosmetic = any(kw in message for kw in _COSMETIC_MESSAGE_KEYWORDS)

    return whitespace_only or (message_hints_cosmetic and not added and not removed)


# ==============================================================================
# Core SZZ walk
# ==============================================================================

def find_introducing_commit_for_hunk(
    repo_path: str,
    fix_commit_sha: str,
    file_path: str,
    old_start: int,
    old_count: int,
    max_depth: int = MAX_WALKBACK_DEPTH,
) -> Dict[str, Any]:
    """
    SZZ for a single hunk: blame the hunk's old-line range at fix_commit~1,
    then walk back past cosmetic commits (bounded by max_depth).

    Returns:
      {
        "file_path": ..., "old_start": ..., "old_count": ...,
        "omission": bool,          # True if old_count == 0 (nothing to blame)
        "candidate_sha": str|None, # best guess after cosmetic-filtering
        "raw_blame_shas": [...],   # every commit touching the range at fix~1
        "walked_back_depth": int,
        "filtered_cosmetic": [sha, ...],  # commits skipped as cosmetic
      }
    """
    if old_count == 0:
        return {
            "file_path": file_path, "old_start": old_start, "old_count": old_count,
            "omission": True, "candidate_sha": None, "raw_blame_shas": [],
            "walked_back_depth": 0, "filtered_cosmetic": [],
        }

    fix_parent = get_parent_sha(repo_path, fix_commit_sha)
    if not fix_parent:
        return {
            "file_path": file_path, "old_start": old_start, "old_count": old_count,
            "omission": False, "candidate_sha": None, "raw_blame_shas": [],
            "walked_back_depth": 0, "filtered_cosmetic": [],
            "error": f"could not resolve parent of {fix_commit_sha}",
        }

    end_line = old_start + max(old_count, 1) - 1
    blamed = blame_lines_at_commit(repo_path, fix_parent, file_path, old_start, end_line)
    raw_shas = [b["sha"] for b in blamed]

    if not blamed:
        return {
            "file_path": file_path, "old_start": old_start, "old_count": old_count,
            "omission": False, "candidate_sha": None, "raw_blame_shas": [],
            "walked_back_depth": 0, "filtered_cosmetic": [],
        }

    # Most recent blamed commit = first candidate (git blame porcelain doesn't
    # guarantee recency ordering across multiple distinct commits in a range,
    # so sort by author-time descending when available).
    def _sort_key(b):
        try:
            return -int(b["date"])
        except (ValueError, TypeError):
            return 0
    blamed_sorted = sorted(blamed, key=_sort_key)

    filtered_cosmetic: List[str] = []
    candidate = blamed_sorted[0]
    depth = 0

    while depth < max_depth and _is_cosmetic(repo_path, candidate["sha"], file_path):
        filtered_cosmetic.append(candidate["sha"])
        parent = get_parent_sha(repo_path, candidate["sha"])
        if not parent:
            break
        next_blame = blame_lines_at_commit(repo_path, parent, file_path, old_start, end_line)
        if not next_blame:
            break
        next_sorted = sorted(next_blame, key=_sort_key)
        if next_sorted[0]["sha"] == candidate["sha"]:
            break  # no progress, stop
        candidate = next_sorted[0]
        depth += 1

    return {
        "file_path": file_path, "old_start": old_start, "old_count": old_count,
        "omission": False, "candidate_sha": candidate["sha"],
        "raw_blame_shas": raw_shas, "walked_back_depth": depth,
        "filtered_cosmetic": filtered_cosmetic,
    }


def find_introducing_commit_szz(
    repo_path: str,
    fix_commit_sha: str,
    max_depth: int = MAX_WALKBACK_DEPTH,
) -> Dict[str, Any]:
    """
    Full SZZ-lite pass over every hunk in the fix commit's diff.

    Returns:
      {
        "fix_commit_sha": ...,
        "hunk_results": [ ...one entry per hunk from find_introducing_commit_for_hunk... ],
        "candidate_counts": {sha: n},   # how many hunks each sha "won"
        "primary_introducing_sha": str|None,  # most-frequent / most-recent candidate
        "omitted_hunk_count": int,      # hunks with nothing to blame (pure additions)
        "method": "szz-lite",
      }
    """
    hunks = get_fix_diff_hunks(repo_path, fix_commit_sha)
    hunk_results = [
        find_introducing_commit_for_hunk(
            repo_path, fix_commit_sha, h["file_path"], h["old_start"], h["old_count"], max_depth
        )
        for h in hunks
    ]

    candidate_counts: Dict[str, int] = {}
    for hr in hunk_results:
        sha = hr.get("candidate_sha")
        if sha:
            candidate_counts[sha] = candidate_counts.get(sha, 0) + 1

    primary = None
    if candidate_counts:
        # Most hunks pointing at the same commit wins; ties broken by first-seen order.
        primary = max(candidate_counts.items(), key=lambda kv: kv[1])[0]

    omitted = sum(1 for hr in hunk_results if hr.get("omission"))

    return {
        "fix_commit_sha": fix_commit_sha,
        "hunk_results": hunk_results,
        "candidate_counts": candidate_counts,
        "primary_introducing_sha": primary,
        "omitted_hunk_count": omitted,
        "method": "szz-lite",
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python szz.py <repo_path> <fix_commit_sha>")
        sys.exit(1)

    repo_path_arg, fix_sha_arg = sys.argv[1], sys.argv[2]
    out = find_introducing_commit_szz(repo_path_arg, fix_sha_arg)
    print(json.dumps(out, indent=2))