#!/usr/bin/env python3
"""
Clean CLI for Agentic AI Bug Triage Agent
"""

import os
import json
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()
console = Console()
app = typer.Typer(help="Agentic AI Bug Triage Agent")

def check_api_key():
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error: GEMINI_API_KEY not set. Create a .env file or export it.[/red]")
        raise typer.Exit(1)

@app.command()
def investigate(
    repo: str = typer.Option(..., "--repo", "-r", help="Path to local git repository"),
    trace_file: Optional[str] = typer.Option(None, "--trace-file", "-t", help="Path to stack trace file"),
    trace: Optional[str] = typer.Option(None, "--trace", help="Stack trace as string"),
    max_steps: int = typer.Option(12, "--max-steps", help="Maximum agent steps"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Resume from checkpoint"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save result to JSON file"),
):
    """Run the LLM agent to find the introducing commit."""
    check_api_key()

    if not trace and not trace_file:
        console.print("[red]Provide either --trace or --trace-file[/red]")
        raise typer.Exit(1)

    if trace_file:
        stack_trace = Path(trace_file).read_text(encoding="utf-8")
    else:
        stack_trace = trace

    from llm_agent import investigate_with_llm

    console.print(Panel.fit("Starting Agentic Investigation...", style="bold blue"))

    result = investigate_with_llm(
        stack_trace=stack_trace,
        repo_path=repo,
        max_steps=max_steps,
        resume_from=checkpoint,
    )

    # Pretty print
    table = Table(title="Investigation Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    final = result.get("final_answer", "No answer")
    table.add_row("Final Answer", str(final)[:300] + ("..." if len(str(final)) > 300 else ""))
    table.add_row("Steps Used", str(result.get("steps_used", "N/A")))
    table.add_row("Status", result.get("status", "unknown"))

    console.print(table)

    if output:
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        console.print(f"[green]Result saved to {output}[/green]")

@app.command()
def szz(
    repo: str = typer.Option(..., "--repo", "-r"),
    fix_commit: str = typer.Option(..., "--fix-commit", "-f"),
):
    """Run SZZ-lite to find introducing commit from a fix commit."""
    from szz import find_introducing_commit_szz

    result = find_introducing_commit_szz(repo, fix_commit)

    console.print(Panel.fit("SZZ-lite Result", style="bold magenta"))
    console.print(f"Primary Introducing Commit : {result.get('primary_introducing_sha')}")
    console.print(f"Omitted Hunks              : {result.get('omitted_hunk_count')}")
    console.print(f"Candidate Counts           : {result.get('candidate_counts')}")
    console.print(f"Needs Manual Review        : {result.get('needs_manual_review', False)}")

@app.command()
def baseline(
    repo: str = typer.Option(..., "--repo", "-r"),
    trace_file: str = typer.Option(..., "--trace-file", "-t"),
):
    """Run the rule-based baseline (no LLM)."""
    from script import parse_stack_trace, rank_frames, investigate

    stack_trace = Path(trace_file).read_text(encoding="utf-8")
    parsed = parse_stack_trace(stack_trace)
    ranked = rank_frames(parsed["frames"], repo)

    console.print(f"Ranked frames: {len(ranked)}")
    result = investigate(parsed["frames"], repo, max_steps=8)

    best = result.get("best_finding")
    if best:
        console.print(Panel.fit(str(best), title="Baseline Result", style="green"))
    else:
        console.print("[yellow]No finding returned by baseline[/yellow]")

if __name__ == "__main__":
    app()
