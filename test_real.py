import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from script import parse_stack_trace, rank_frames, investigate
from szz import find_introducing_commit_szz

# ============ EDIT THESE FOR EACH NEW ISSUE — nothing else needs to change ============
repo_path = r"C:\Users\HP\flask"
known_fix_commit = "980168d08498c00a14ab0f687ffac8cc50edb326"

stack_trace = """
Traceback (most recent call last):
  File "/Users/kchung/.virtualenvs/ctfd3/lib/python3.7/site-packages/werkzeug/serving.py", line 303, in run_wsgi
    execute(self.server.app)
  File "/Users/kchung/.virtualenvs/ctfd3/lib/python3.7/site-packages/werkzeug/serving.py", line 294, in execute
    write(data)
  File "/Users/kchung/.virtualenvs/ctfd3/lib/python3.7/site-packages/werkzeug/serving.py", line 274, in write
    assert isinstance(data, bytes), "applications must write bytes"
AssertionError: applications must write bytes

"""
# ========================================================================================

target_commit = f"{known_fix_commit}~1"

print("=== STEP 1: Parse stack trace ===")
parsed = parse_stack_trace(stack_trace)
frames = parsed["frames"]
print(f"Exception: {parsed.get('exception_type')}: {parsed.get('error_message')}")
print(f"Raw frames found: {len(frames)}")

print("\n=== STEP 2: Ground truth via SZZ (fix commit's own diff — no checkout needed) ===")
# This no longer requires checking out the repo first: szz.py blames at an
# arbitrary historical revision directly, and it works from the FIX COMMIT'S
# diff, not from the stack trace's crash line.
szz_result = find_introducing_commit_szz(repo_path, known_fix_commit)
ground_truth_sha = szz_result["primary_introducing_sha"]

print(f"SZZ candidates (by hunk vote): {szz_result['candidate_counts']}")
print(f"SZZ primary introducing commit: {ground_truth_sha}")
print(f"Omitted hunks (pure additions, nothing to blame): {szz_result['omitted_hunk_count']}")
if len(szz_result["candidate_counts"]) > 1:
    print("  !! Multiple distinct candidates across hunks — recommend manual review before trusting this as ground truth.")
for hr in szz_result["hunk_results"]:
    if hr.get("filtered_cosmetic"):
        print(f"  (filtered {len(hr['filtered_cosmetic'])} cosmetic commit(s) while walking back for "
              f"{hr['file_path']}:{hr['old_start']})")

print("\n=== STEP 3: Checkout pre-fix state so the AGENT investigates blind ===")
# The agent must NOT see the fix — it only gets the crash + a repo checked
# out to fix_commit~1, exactly like an engineer paged with only a traceback.
import subprocess
subprocess.run(
    ["git", "checkout", "--force", target_commit],
    cwd=repo_path, capture_output=True, text=True,
    encoding="utf-8", errors="replace", check=True,
)

print("\n=== STEP 4: Rank frames (generic -- no hardcoded package name) ===")
ranked = rank_frames(frames, repo_path)
print("Ranked frames:")
for r in ranked:
    print(f" - {r['file_path']}:{r['line_number']}")

if not ranked:
    print("\nNo repo-owned frames found. Possible causes:")
    print("  - repo_path doesn't point to the right local clone")
    print("  - the repo isn't checked out to a commit where these files exist")
    print("  - the trace's crash is genuinely all third-party (e.g. only werkzeug)")
    sys.exit(1)

top_frame = ranked[0]
print(f"\nUsing top frame: {top_frame['file_path']}:{top_frame['line_number']}")

print("\n=== STEP 5: Run investigate() (rule-based agent) and compare ===")
result = investigate(frames, repo_path, max_steps=8, confidence_threshold=0.8, target_commit=target_commit)

best = result.get("best_finding")
if best:
    agent_sha = None
    if best["evidence_source"] == "recent_commits" and best["evidence"]:
        agent_sha = best["evidence"][0]["sha"]
    elif best["evidence_source"] == "git_blame" and best["evidence"]:
        agent_sha = best["evidence"].get("sha", "")

    print("\n--- FINAL REPORT ---")
    print(f"Frame selected: {best['frame']['file_path']}:{best['frame']['line_number']}")
    print(f"Ground truth (SZZ primary):   {ground_truth_sha}")
    print(f"Agent's top finding:          {agent_sha}")
    print(f"  Score: {best['confidence']:.2f}")
    print(f"  Source (path taken): {best['evidence_source']}")
    print(f"  commit_score={best.get('commit_score', 'n/a')}, blame_score={best.get('blame_score', 'n/a')}")
    print(f"Match ground truth (SZZ)? {'YES' if agent_sha == ground_truth_sha else 'NO'}")
else:
    print("\n--- FINAL REPORT ---")
    print("No finding returned by investigate().")