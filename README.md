# Agentic AI Analysis

An **LLM-powered bug triage agent** that identifies the *introducing commit* (root-cause) of a Python crash from a stack trace and a local Git repository.

The agent is given only the stack trace and a repository checked out to the pre-fix state (`fix_commit~1`). It must discover the commit that introduced the buggy behavior — **not** the commit that later fixed it.

---

## Overview

This project combines:

1. **Rule-based Git investigation tools** (`script.py`)
2. **An LLM agent with tool-calling** (`llm_agent.py`) that uses those tools intelligently
3. **SZZ-lite ground-truth generation** (`szz.py`) for evaluation
4. **Dataset collection & evaluation utilities** (`dataset_collection.py`)

The LLM agent uses Google Gemini (via the OpenAI-compatible Chat Completions API) with function calling. No LangChain or heavy agent frameworks are used — just plain `requests` and a carefully designed tool-calling loop.

---

## Key Features

### LLM Agent (`llm_agent.py`)
- Tool-calling loop with Gemini (`gemini-3.1-flash-lite` by default)
- Tools available to the agent:
  - `get_recent_commits` – file-level history
  - `get_blame` – exact line blame
  - `get_diff` – commit diff + **mechanical pattern check**
  - `checkout_commit` – move HEAD for walk-back investigation
- Bounded tool-result payloads & token throttling
- Step-level checkpointing (resume from `agent_checkpoint.json`)
- Batch investigation mode with efficiency statistics
- Strong gating rules that force the model to diff candidates before answering

### Rule-based Tools (`script.py`)
- Robust Python stack-trace parser
- Frame ranking that prefers repository-owned code over third-party packages
- Git blame, recent commits, and scoped diffs
- Confidence scoring for candidate commits

### SZZ-lite Ground Truth (`szz.py`)
A practical implementation of the classic SZZ algorithm:
- Works from the **fix commit’s own diff** (not the crash line)
- Blames the lines changed by the fix at `fix_commit~1`
- Filters cosmetic commits (whitespace, formatting, renames)
- Bounded walk-back past cosmetic changes
- Handles multi-hunk / multi-file fixes and the SZZ “omission problem”

---

## Project Structure
agentic-ai-analysis/
├── llm_agent.py              # Main LLM agent with tool-calling loop
├── script.py                 # Rule-based tools (parse, blame, diff, ranking…)
├── szz.py                    # SZZ-lite ground-truth generator
├── dataset_collection.py     # Build eval datasets + score agent results
├── test_real.py              # End-to-end test harness (edit repo + fix commit)
├── agent_checkpoint.json     # Example checkpoint from a previous run
└── pycache/
text---

## Requirements

- Python 3.9+
- Git
- A Gemini API key (free tier works well)

```bash
pip install requests
Set your API key:
Bash# bash
export GEMINI_API_KEY="AIza..."

# PowerShell
$env:GEMINI_API_KEY = "AIza..."

Quick Start
1. Investigate a single crash with the LLM agent
Pythonfrom llm_agent import investigate_with_llm

result = investigate_with_llm(
    stack_trace=your_stack_trace,
    repo_path="/path/to/local/clone",
    # optional: resume_from="agent_checkpoint.json"
)

print(result["final_answer"])
2. Run the rule-based baseline
Pythonfrom script import parse_stack_trace, rank_frames, investigate

parsed = parse_stack_trace(stack_trace)
ranked = rank_frames(parsed["frames"], repo_path)
result = investigate(parsed["frames"], repo_path, max_steps=8)
3. Generate ground truth with SZZ-lite
Pythonfrom szz import find_introducing_commit_szz

result = find_introducing_commit_szz(repo_path, fix_commit_sha)
print(result["primary_introducing_sha"])
4. End-to-end test
Edit the top of test_real.py with:

Path to a local clone
Known fix commit SHA
A real stack trace

Then run:
Bashpython test_real.py

How the Agent Works

The repository is checked out to fix_commit~1 (agent never sees the fix).
The agent receives a parsed stack trace + repository path.
It freely chooses investigation strategy:
Blame path – start with exact line blame, then walk past partial fixes if justified
Commit-scan path – progressive widening of recent commits (10 → 30 → 50) with mandatory diff inspection

Every get_diff result includes a mechanical pattern check (does the crash-related term appear in added vs removed lines?).
Strong enforcement rules prevent the model from answering until candidate commits have been properly diffed (unless a strong “introduces” signal already exists).
Final answer format includes: root-cause SHA, reasoning, path used, confidence, and whether any fixes were walked past.


Evaluation
Ground truth is generated with SZZ-lite (dataset_collection.py). Agent findings are compared against the SZZ-derived introducing commit (never against the fix commit).
Known limitations of SZZ-lite (documented in szz.py):

Pure-addition hunks cannot be blamed (SZZ omission problem)
Multi-file fixes may produce multiple candidates
Cosmetic filter is heuristic

Cases flagged needs_manual_review should be inspected before treating them as hard ground truth.

Design Notes

The agent is deliberately constrained (max steps, token budgets, forced diffing) to keep costs low and force decisive investigation.
Mechanical pattern checking on diffs reduces reliance on pure LLM judgment.
Checkpointing allows long investigations to be resumed after rate limits or interruptions.
No heavy frameworks — the entire tool-calling loop is transparent and easy to inspect/modify.
