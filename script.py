import os
import re
import subprocess
from typing import Dict, List, Optional, Any

# ==============================================================================
# PART 1: Stack trace parser
# ==============================================================================

def parse_stack_trace(trace_text: str) -> dict:
    """
    Parses a raw Python stack trace to extract exception type, error message,
    and a list of all frames (file paths and line numbers).
    """
    # Initialize our result dictionary with None/empty values so we always return a consistent structure.
    result = {
        "exception_type": None,
        "error_message": None,
        "frames": []
    }
    
    try:
        # Split the trace into individual lines and remove any empty strings to clean up the input.
        lines = [line.strip() for line in trace_text.strip().split('\n') if line.strip()]
        
        if not lines:
            # If the trace text was empty or only whitespace, return an error dictionary.
            return {"error": "Empty stack trace provided."}
            
        # The last line of a typical Python traceback usually contains the exception type and message.
        # Format: "ExceptionType: error message"
        last_line = lines[-1]
        
        # Check if the last line contains a colon, which separates the type and message.
        if ":" in last_line:
            # Split only on the first colon to handle cases where the message itself contains colons.
            exc_type, exc_msg = last_line.split(":", 1)
            result["exception_type"] = exc_type.strip()
            result["error_message"] = exc_msg.strip()
        else:
            # If there's no colon, it might be an exception without a message (rare, but possible).
            result["exception_type"] = last_line.strip()
            
        # To find the files and line numbers, we look for all lines starting with "File".
        # We iterate through the lines (forward, to preserve top-to-bottom order).
        for line in lines[:-1]:
            # A typical frame line looks like: File "path/to/file.py", line 42, in <module>
            if line.startswith("File "):
                # Use a regular expression to capture the file path (inside quotes) and the line number (after "line ").
                match = re.search(r'File "([^"]+)", line (\d+)', line)
                if match:
                    result["frames"].append({
                        "file_path": match.group(1),
                        "line_number": int(match.group(2))
                    })
                
        return result
    except Exception as e:
        # If any unexpected error occurs during parsing (e.g., regex error), catch it and return gracefully.
        return {"error": f"Failed to parse stack trace: {str(e)}"}

# ==============================================================================
# PART 2: Git investigation tools
# ==============================================================================

def get_recent_commits(repo_path: str, file_path: str, max_commits: int = 10) -> list:
    """
    Runs `git log` scoped to a specific file and returns a list of dictionaries 
    with commit details.
    """
    try:
        # Use a custom format string (--format) to get output in a consistent, easily parseable way.
        # %H = commit hash, %an = author name, %ad = author date, %s = subject (commit message).
        # We separate fields with a unique delimiter (|~|) that is unlikely to appear in commit messages.
        cmd = [
            "git", "log", f"-n{max_commits}", 
            "--format=%H|~|%an|~|%ad|~|%s", 
            "--", file_path
        ]
        
        # subprocess.run executes the command. 
        # capture_output=True captures stdout and stderr so we can read them.
        # text=True returns strings instead of bytes.
        # check=True raises an exception if the git command fails (e.g., repo doesn't exist).
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
        
        commits = []
        # Process each line of the command output.
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
                
            # Split the line by our custom delimiter to get individual fields.
            parts = line.split('|~|')
            if len(parts) == 4:
                # Append the parsed details as a dictionary to our list.
                commits.append({
                    "sha": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                })
        return commits
        
    except subprocess.CalledProcessError as e:
        # If git returns a non-zero exit code (e.g., not a git repository), catch it here and return an empty list.
        # e.stderr contains the error message printed by git.
        print(f"Error in get_recent_commits: Git command failed. {e.stderr.strip()}")
        return []
    except Exception as e:
        # Catch any other unexpected errors (e.g., invalid repo path type).
        print(f"Error in get_recent_commits: {str(e)}")
        return []

def get_blame(repo_path: str, file_path: str, line_number: int) -> dict:
    """
    Runs `git blame` for a specific line in a file and returns the commit details.
    """
    try:
        # Use git blame with -L to restrict it to just the specific line number we want.
        # The line range format is start,end. By using line,line we get just that single line.
        # --porcelain provides machine-readable output which is easier to parse reliably.
        cmd = [
            "git", "blame", "-L", f"{line_number},{line_number}", 
            "--porcelain", "--", file_path
        ]
        
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
        
        # The output might be empty if the file doesn't have that many lines.
        if not result.stdout.strip():
            return {"error": f"Line {line_number} not found in {file_path}"}
            
        lines = result.stdout.strip().split('\n')
        
        # The first line of --porcelain output contains the commit SHA and original/final line numbers.
        first_line_parts = lines[0].split()
        commit_sha = first_line_parts[0]
        
        author = "Unknown"
        date = "Unknown"
        
        # Iterate through the remaining lines of porcelain output to find author and time info.
        for line in lines[1:]:
            if line.startswith("author "):
                # Extract everything after "author "
                author = line[7:]
            elif line.startswith("author-time "):
                # Extract the timestamp (we just keep it as the raw Unix timestamp string for simplicity)
                date = line[12:]
                
        return {
            "sha": commit_sha,
            "author": author,
            "date": date
        }
        
    except subprocess.CalledProcessError as e:
        # Gracefully handle git errors (e.g., file not tracked by git, or invalid line number).
        return {"error": f"Git command failed: {e.stderr.strip()}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_diff(repo_path: str, commit_sha: str, file_path: Optional[str] = None) -> str:
    """
    Runs `git show` to return the diff text for a specific commit.

    When file_path is provided, scopes the diff to that file only (avoids
    unrelated docs/changelog noise and truncation before the crashing hunk).
    """
    try:
        # git show displays the commit object and its diff.
        # Using --format= clears the commit message, leaving only the diff itself.
        cmd = ["git", "show", "--format=", commit_sha]
        if file_path:
            cmd.extend(["--", file_path])

        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        return f"Error: Git command failed. {e.stderr.strip()}"
    except Exception as e:
        return f"Error: Unexpected failure: {str(e)}"

# ==============================================================================
# PART 3: Frame ranking
# ==============================================================================

def _is_repo_owned_frame(file_path: str, repo_path: str) -> tuple:
    """
    Decide if a frame belongs to this repo vs a genuine third-party dependency.

    site-packages / dist-packages often contain BOTH third-party libs and a
    virtualenv's installed copy of THIS repo. Only skip true third-party frames.
    Returns (is_repo_owned, repo_relative_path_or_original).
    """
    normalized = file_path.replace("\\", "/")
    packaged = None
    if "site-packages/" in normalized:
        packaged = normalized.split("site-packages/", 1)[1]
    elif "dist-packages/" in normalized:
        packaged = normalized.split("dist-packages/", 1)[1]

    if packaged is None:
        # Not under site/dist-packages — treat as repo-owned (caller may remap).
        return True, file_path

    # If the path after site-packages exists inside the repo, it's OUR code.
    candidate = packaged
    if os.path.isfile(os.path.join(repo_path, candidate)):
        return True, candidate

    # Some repos (like Flask) keep their package under a "src/" folder —
    # try that location too before giving up.
    src_candidate = os.path.join("src", candidate)
    if os.path.isfile(os.path.join(repo_path, src_candidate)):
        return True, src_candidate.replace("\\", "/")

    return False, file_path

def rank_frames(frames: list, repo_path: str) -> list:
    """
    Ranks frames by prioritizing those that belong to the local repository.
    Filters out genuine third-party site-packages/dist-packages frames, but
    keeps a virtualenv's copy of THIS repo's code. Orders from closest to the
    crash (bottom of the stack) to furthest (top of the stack).
    """
    try:
        # Reverse the frames so the closest to the crash is first.
        # The parser appended them from top to bottom, so the last is the closest.
        reversed_frames = frames[::-1]
        
        ranked = []
        for frame in reversed_frames:
            owned, rel_path = _is_repo_owned_frame(frame["file_path"], repo_path)
            if owned:
                # Remap venv site-packages paths to repo-relative so git tools work.
                updated = dict(frame)
                updated["file_path"] = rel_path
                ranked.append(updated)
                
        if not ranked:
            print("Warning: No repo-owned frames found. The bug might not be fixable in this repository.")
            return reversed_frames # Return all frames as a fallback
            
        return ranked
    except Exception as e:
        print(f"Error in rank_frames: {str(e)}")
        # Graceful fallback: return the frames as-is if something unexpected happens
        return frames

# ==============================================================================
# PART 4: Confidence scoring
# ==============================================================================

def _commit_touches_near_line(repo_path: str, commit_sha: str, file_path: str, line_number: int, window: int = 25) -> bool:
    """
    Returns True if commit_sha's diff for file_path changes lines within
    `window` of line_number. Used as a suspicion signal for cause commits.
    """
    try:
        cmd = ["git", "show", "--format=", "--unified=0", commit_sha, "--", file_path]
        result = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True,
            encoding='utf-8', errors='replace', check=True
        )
        diff_text = result.stdout
        if not diff_text.strip():
            return False

        # Parse unified-diff hunk headers: @@ -old_start,old_count +new_start,new_count @@
        for match in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', diff_text, re.MULTILINE):
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            # Deletion-only hunks have new_count 0; still check around new_start.
            hunk_lo = new_start
            hunk_hi = new_start + max(new_count, 1) - 1
            if line_number >= (hunk_lo - window) and line_number <= (hunk_hi + window):
                return True
        return False
    except (subprocess.CalledProcessError, Exception):
        return False


def score_commits(commits: list, frame: dict, repo_path: str = None) -> float:
    """
    Scores file-level recent commits as a WEAKER signal than line-exact blame.

    Recency alone must NOT outscore a valid blame result — file-level history is
    noisy. Only strong evidence (code changed near the crashing line) can push
    this score above a successful blame baseline (~0.5).
    """
    if not commits:
        return 0.0
        
    try:
        best_score = 0.0
        line_number = frame.get("line_number")
        file_name = frame["file_path"].split("/")[-1].split("\\")[-1]
        
        for i, commit in enumerate(commits):
            score = 0.0
            msg = commit.get("message", "").lower()
            
            # 1. Recency — capped low so "newer file commit" cannot beat line-exact blame.
            # Most recent ≈ 0.35; decays with index. Alone stays under blame baseline (0.5).
            recency_bonus = max(0.0, 0.35 - (i * 0.06))
            score += recency_bonus
                
            # 2. Weak file-name mention in the subject.
            if file_name.lower() in msg:
                score += 0.10
                
            # 3. Strong signal: commit changed code near the crashing line.
            # This is what can legitimately outscore an old-but-valid blame.
            if repo_path and line_number is not None:
                if _commit_touches_near_line(repo_path, commit["sha"], frame["file_path"], line_number):
                    score += 0.40
            
            score = min(1.0, score)
            
            if score > best_score:
                best_score = score
                
        return best_score
    except Exception as e:
        print(f"Error in score_commits: {str(e)}")
        return 0.0

def score_blame(blame_info: dict, repo_path: str = ".") -> float:
    """
    Scores blame: structurally stronger than file-level recent commits because it
    answers who last touched the EXACT crashing line.

    Baseline 0.5 reflects that targeted evidence; recency adds up to +0.3.
    An old introducing commit still beats pure file-level recency.
    """
    if not blame_info or "error" in blame_info:
        return 0.0
        
    try:
        # Structural baseline: line-exact evidence (beats file-level recency alone).
        score = 0.50
        
        if blame_info.get("date") != "Unknown":
            try:
                head_time_str = subprocess.run(
                    ["git", "show", "-s", "--format=%ct", "HEAD"],
                    cwd=repo_path, capture_output=True, text=True,
                    encoding='utf-8', errors='replace', check=True
                ).stdout.strip()
                head_time = int(head_time_str)
                commit_time = int(blame_info.get("date"))
                
                age_days = (head_time - commit_time) / (60 * 60 * 24)
                
                # Recency bonus up to +0.3 on top of the structural baseline.
                if age_days <= 14:
                    score += 0.30  # ~0.80 recent regression on this line
                elif age_days <= 90:
                    score += 0.25  # ~0.75
                elif age_days <= 365:
                    score += 0.15  # ~0.65
                elif age_days <= 730:
                    score += 0.10  # ~0.60
                # else: old latent line-touch keeps baseline 0.50
            except (ValueError, subprocess.CalledProcessError):
                pass
        # Unknown date: keep structural baseline only.
            
        return min(1.0, score)
    except Exception as e:
        print(f"Error in score_blame: {str(e)}")
        return 0.0

# ==============================================================================
# PART 5: Agent loop skeleton
# ==============================================================================

def investigate(frames: list, repo_path: str, max_steps: int = 8, confidence_threshold: float = 0.6, target_commit: str = None) -> dict:
    """
    The main agent loop. Ranks frames, checks recent commits and blame, scores them,
    and stops early if confidence is high enough.
    If target_commit is provided, checks out that commit before investigating.
    """
    if not frames:
        return {"error": "No frames provided"}
        
    try:
        # 1. Checkout target_commit if requested, so we see the code exactly as it was when it crashed.
        if target_commit:
            print(f"Checking out target commit {target_commit} before investigation...")
            try:
                subprocess.run(
                    ["git", "checkout", target_commit], 
                    cwd=repo_path, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True
                )
            except subprocess.CalledProcessError as checkout_err:
                # If checkout fails (e.g., bad SHA or uncommitted changes), fail gracefully.
                return {"error": f"Failed to checkout target commit {target_commit}: {checkout_err.stderr.strip()}"}
                
        ranked_frames = rank_frames(frames, repo_path)
        
        findings = []
        best_finding = None
        steps_used = 0
        
        print(f"Starting investigation with {len(ranked_frames)} ranked frame(s).")
        
        for frame in ranked_frames:
            if steps_used >= max_steps:
                print("Max steps reached. Stopping investigation.")
                break
                
            file_path = frame["file_path"]
            line_number = frame["line_number"]
            steps_used += 1
            
            print(f"\nStep {steps_used}: Checking frame {file_path}:{line_number}")
            
            # Step A: Cheap signal — recent commits on this file (recency + line proximity)
            commits = get_recent_commits(repo_path, file_path, max_commits=5)
            commit_score = score_commits(commits, frame, repo_path=repo_path)
            print(f"  -> Recent commits returned: {len(commits)} commit(s)")
            for c in commits[:3]:
                print(f"       {c['sha'][:12]} | {c['author']} | {c['message'][:60]}")
            print(f"  -> Recent commits score: {commit_score:.2f}")
            
            # Step B: Precise signal — blame the exact crash line.
            print("  -> Checking git blame for exact line...")
            blame_info = get_blame(repo_path, file_path, line_number)
            blame_score = score_blame(blame_info, repo_path)
            print(f"  -> Blame score: {blame_score:.2f}")
            if blame_info and "error" not in blame_info:
                print(f"       blamed sha={blame_info.get('sha', '')[:12]} author={blame_info.get('author')}")
            
            # Within-frame winner: ALWAYS the higher confidence score.
            # On a tie, prefer blame (more precise: exact line vs file-level log).
            blame_finding = {
                "frame": frame,
                "confidence": blame_score,
                "evidence_source": "git_blame",
                "evidence": blame_info,
                "supporting_commits": commits[:2] if commits else [],
                "commit_score": commit_score,
                "blame_score": blame_score,
            }
            commits_finding = {
                "frame": frame,
                "confidence": commit_score,
                "evidence_source": "recent_commits",
                "evidence": commits[:2] if commits else [],
                "commit_score": commit_score,
                "blame_score": blame_score,
            }
            if blame_score > commit_score:
                finding = blame_finding
            elif commit_score > blame_score:
                finding = commits_finding
            else:
                # Tie: prefer blame if it produced usable evidence, else commits.
                finding = blame_finding if blame_score > 0 else commits_finding
            print(
                f"  -> Path chosen: {finding['evidence_source']} "
                f"(commits={commit_score:.2f} vs blame={blame_score:.2f}; "
                f"winner confidence={finding['confidence']:.2f})"
            )

            findings.append(finding)
            
            # Across frames: keep the finding with the HIGHEST confidence so far
            # (not merely the last frame checked).
            if best_finding is None or finding["confidence"] > best_finding["confidence"]:
                print(
                    f"  -> New best_finding "
                    f"(confidence {finding['confidence']:.2f} > "
                    f"{best_finding['confidence'] if best_finding else 0.0:.2f})"
                )
                best_finding = finding
            else:
                print(
                    f"  -> best_finding unchanged "
                    f"(this={finding['confidence']:.2f} <= best={best_finding['confidence']:.2f})"
                )
                
            # Stop early only when THIS step's finding clears the threshold
            if finding["confidence"] >= confidence_threshold:
                print(f"*** Confidence threshold ({confidence_threshold}) reached! Stopping early. ***")
                break
        else:
            # for-loop completed without break → every ranked frame was checked
            print(
                f"\nLoop finished: checked all {len(ranked_frames)} ranked frame(s); "
                f"no finding reached confidence_threshold={confidence_threshold}."
            )
                
        return {
            "best_finding": best_finding,
            "all_findings": findings,
            "steps_used": steps_used
        }
        
    except Exception as e:
        return {"error": f"Investigation loop failed: {str(e)}"}

# ==============================================================================
# EXECUTION AND TESTING
# ==============================================================================

if __name__ == "__main__":
    print("\n--- Agent Loop Test ---")
    
    # We will use our dummy test_file.py for this.
    # Let's create a fake stack trace that points to test_file.py and another fake file
    test_trace = """
    Traceback (most recent call last):
      File "/usr/local/lib/python3.9/site-packages/fake_lib.py", line 99, in do_stuff
        crash()
      File "test_file.py", line 1, in <module>
        print('Hello Universe!')
    TypeError: 'str' object is not callable
    """
    
    parsed = parse_stack_trace(test_trace)
    print(f"Parsed frames: {parsed.get('frames', [])}")
    
    # Run the investigate loop with a low threshold to show early stopping
    result = investigate(parsed.get("frames", []), repo_path=".", max_steps=5, confidence_threshold=0.65)
    
    print("\n--- Final Result ---")
    print(f"Steps used: {result.get('steps_used')}")
    best = result.get("best_finding")
    if best:
        print(f"Best finding confidence: {best['confidence']:.2f}")
        print(f"Best finding source: {best['evidence_source']}")
    else:
        print("No finding returned.")
