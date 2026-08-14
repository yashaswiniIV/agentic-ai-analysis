"""
dataset_collection.py — builds eval-dataset ground truth using SZZ-lite,
and evaluates agent findings against it.

CHANGE FROM THE PREVIOUS VERSION:
  Old approach: checkout fix_commit~1, then git-blame the STACK TRACE's
  crash line, use that as "introducing_commit_sha".
  New approach: call szz.find_introducing_commit_szz(), which blames the
  lines the FIX COMMIT actually changed (not the crash line), with
  cosmetic-commit filtering and bounded walk-back. This is the real SZZ
  method, not a crash-line proxy.

  No `git checkout` is needed here anymore — szz.py blames at an arbitrary
  historical revision directly (via `git blame <rev> -L ...`), so it works
  regardless of what's currently checked out in the working tree, and it no
  longer needs the parsed stack trace at all (SZZ works purely from the fix
  commit's own diff).
"""

import csv
import json
from typing import Any, Dict, List

from szz import find_introducing_commit_szz


def run_dataset_collection(repo_path: str, dataset_entries: List[Dict[str, Any]], output_csv: str) -> None:
    """
    dataset_entries: list of {"issue_id": ..., "fix_commit_sha": ...}
    (trace_text is no longer required for ground truth — SZZ doesn't use it —
    but keep it in your dataset anyway, the LLM agent still needs it as input.)

    Writes a CSV with columns:
      issue_id, fix_commit_sha, introducing_commit_sha, method,
      omitted_hunk_count, walked_back_any, candidate_count, needs_manual_review
    """
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "issue_id", "fix_commit_sha", "introducing_commit_sha", "method",
            "omitted_hunk_count", "walked_back_any", "candidate_count",
            "needs_manual_review",
        ])

        for entry in dataset_entries:
            issue_id = entry["issue_id"]
            fix_commit = entry["fix_commit_sha"]

            try:
                szz_result = find_introducing_commit_szz(repo_path, fix_commit)
            except Exception as e:
                print(f"[{issue_id}] SZZ failed for fix_commit={fix_commit}: {e}")
                writer.writerow([issue_id, fix_commit, None, "szz-lite-error", None, None, None, True])
                continue

            introducing_sha = szz_result["primary_introducing_sha"]
            omitted = szz_result["omitted_hunk_count"]
            candidate_count = len(szz_result["candidate_counts"])
            walked_back_any = any(
                hr.get("walked_back_depth", 0) > 0 for hr in szz_result["hunk_results"]
            )

            # Flag cases a human should sanity-check before trusting the
            # auto-picked primary: multiple distinct candidates across hunks
            # (multi-file/multi-hunk fix with disagreement), or every hunk
            # being a pure omission (nothing could be blamed at all).
            needs_manual_review = (candidate_count > 1) or (
                introducing_sha is None and omitted == len(szz_result["hunk_results"])
            )

            writer.writerow([
                issue_id, fix_commit, introducing_sha, "szz-lite",
                omitted, walked_back_any, candidate_count, needs_manual_review,
            ])

            print(
                f"[{issue_id}] introducing_commit={introducing_sha} "
                f"(candidates={candidate_count}, omitted_hunks={omitted}, "
                f"walked_back={walked_back_any}, needs_review={needs_manual_review})"
            )

            # Optional: keep the full per-hunk breakdown alongside the CSV for
            # cases flagged needs_manual_review, so you can inspect them later
            # without re-running SZZ.
            if needs_manual_review:
                debug_path = f"szz_debug_{issue_id}.json"
                with open(debug_path, "w", encoding="utf-8") as dbg:
                    json.dump(szz_result, dbg, indent=2)
                print(f"    -> full hunk breakdown saved to {debug_path} for manual review")


def evaluate_agent(agent_results: List[Dict[str, Any]]) -> None:
    """
    Compares agent findings to SZZ-derived ground truth.

    agent_results: list of {
        "issue_id": ..., "dataset_introducing_sha": ..., "agent_finding_sha": ...
    }

    Unchanged in spirit from before (still compares against the derived
    introducing commit, never against fix_commit_sha) — kept here so the
    whole eval flow lives in one file.
    """
    correct_count = 0
    total = len(agent_results)

    for result in agent_results:
        issue_id = result["issue_id"]
        ground_truth_intro_sha = result["dataset_introducing_sha"]
        agent_top_finding_sha = result["agent_finding_sha"]

        if ground_truth_intro_sha and agent_top_finding_sha == ground_truth_intro_sha:
            print(f"[{issue_id}] CORRECT: agent found the introducing commit ({agent_top_finding_sha})")
            correct_count += 1
        else:
            print(f"[{issue_id}] INCORRECT: agent found {agent_top_finding_sha}, expected {ground_truth_intro_sha}")

    print(f"\nEval Summary: {correct_count}/{total} correct ({(correct_count / total) * 100:.1f}%)" if total else "No results to evaluate.")


if __name__ == "__main__":
    print("Dataset collection script (SZZ-based) imported/ready.")