"""
project_manager.py
──────────────────
Interactive CLI where you act as the project manager.

You define:
  - Project name
  - How many stacks (role types) the project needs
  - For each stack: role title, description, number of seats

This builds the Project + ProjectStack dataclasses and passes
them directly to RetrievalPipeline.run_project().

Run:
    python project_manager.py
"""

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich import box

from models import Project, ProjectStack, AssignedEmployee, ProjectRoster
from memory import list_projects, load_roster, get_project_team
from retrieval import RetrievalPipeline

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────

def prompt_project() -> Project:
    """
    Interactively ask the user to define a project and its stacks.
    Returns a fully constructed Project dataclass.
    """
    console.print(Panel.fit(
        "[bold cyan]Talent RAG — Project Manager[/bold cyan]\n"
        "[dim]Define your project and required stacks.\n"
        "The system will shortlist the best available employee per seat.[/dim]",
        border_style="cyan"
    ))

    # ── Project name ─────────────────────────────────────────────────────────
    project_name = Prompt.ask("\n[bold]Project name[/bold]")
    project = Project(name=project_name)

    # ── How many stacks ───────────────────────────────────────────────────────
    num_stacks = IntPrompt.ask(
        "[bold]How many stacks (role types) does this project need?[/bold]",
        default=1
    )

    # ── Define each stack ─────────────────────────────────────────────────────
    for i in range(1, num_stacks + 1):
        console.print(f"\n[bold yellow]── Stack {i} of {num_stacks} ──[/bold yellow]")

        stack_title = Prompt.ask("  Role title [dim](e.g. React Frontend Engineer)[/dim]")
        description = Prompt.ask("  Description [dim](what this role involves)[/dim]")
        num_roles   = IntPrompt.ask("  Number of seats needed", default=1)

        project.add_stack(ProjectStack(
            stack       = stack_title,
            description = description,
            num_roles   = num_roles,
        ))

    return project


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def display_project_summary(project: Project):
    """Print a summary table of the project before running retrieval."""
    table = Table(
        title=f"Project: {project.name}",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Stack",        style="bold yellow")
    table.add_column("Description",  style="dim",  max_width=50)
    table.add_column("Seats",        style="cyan",  justify="center")

    for stack in project.stacks:
        table.add_row(stack.stack, stack.description, str(stack.num_roles))

    console.print()
    console.print(table)
    console.print(f"[dim]Total seats to fill: [bold]{project.total_roles()}[/bold][/dim]\n")


def display_roster(roster: ProjectRoster):
    """Print the full project roster — assigned employees + unfilled seats."""

    # ── Roster summary panel ──────────────────────────────────────────────────
    status_color = "green" if roster.is_fully_staffed() else "yellow"
    console.print(Panel(
        f"[bold]{roster.project.name}[/bold]\n\n"
        f"Seats filled  : [{status_color}]{roster.seats_filled()} / {roster.seats_required()}[/{status_color}]\n"
        f"Status        : {'[green]✅ Fully staffed[/green]' if roster.is_fully_staffed() else '[yellow]⚠️  Partially staffed[/yellow]'}",
        title="[bold cyan]Project Roster[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    if not roster.assigned:
        console.print("[red]No employees could be assigned.[/red]")
        return

    # ── Assigned employees table ──────────────────────────────────────────────
    table = Table(
        title=f"Hired Team — {roster.project.name}",
        box=box.ROUNDED,
        border_style="green",
        show_lines=True,
    )
    table.add_column("Stack",      style="bold yellow")
    table.add_column("Seat",       justify="center", style="cyan")
    table.add_column("Employee",   style="bold")
    table.add_column("Experience", justify="center")
    table.add_column("Cosine",     justify="center", style="magenta")

    for e in roster.assigned:
        table.add_row(
            e.stack.stack,
            str(e.seat_number),
            e.name,
            f"{e.experience:.0f}y",
            f"{e.cosine_score:.3f}",
        )

    console.print()
    console.print(table)

    # ── Per-stack breakdown ───────────────────────────────────────────────────
    console.print("\n[bold]Breakdown by stack:[/bold]")
    for stack in roster.project.stacks:
        members = roster.by_stack(stack.stack)
        names   = ", ".join(e.name for e in members) if members else "[red]none assigned[/red]"
        console.print(f"  [yellow]{stack.stack}[/yellow] → {names}")

    # ── Unfilled seats ────────────────────────────────────────────────────────
    if roster.unfilled_seats:
        console.print("\n[bold red]Unfilled seats:[/bold red]")
        for stack, seat in roster.unfilled_seats:
            console.print(f"  [red]✗[/red] {stack.stack} — Seat {seat} (no available employee found)")

    # ── Per-employee detail panels ────────────────────────────────────────────
    console.print()
    for e in roster.assigned:
        # Build skill gap block
        gap = e.skill_gap
        if gap.has_gaps():
            gap_color = "green" if gap.gap_score >= 0.7 else "yellow" if gap.gap_score >= 0.4 else "red"
            gap_block = (
                f"\n[dim]Skill gap (score: [{gap_color}]{gap.gap_score:.2f}[/{gap_color}] / 1.00):[/dim]\n"
            )
            if gap.missing_skills:
                gap_block += f"  [red]Missing :[/red] {', '.join(gap.missing_skills)}\n"
            if gap.partial_skills:
                gap_block += "  [yellow]Partial :[/yellow]\n"
                for ps in gap.partial_skills:
                    gap_block += f"    • {ps}\n"
            if gap.growth_areas:
                gap_block += f"  [cyan]Focus on:[/cyan] {gap.growth_areas}"
        else:
            gap_block = f"\n[green]No skill gaps — perfect fit (score: {gap.gap_score:.2f})[/green]"

        console.print(Panel(
            f"[bold]{e.name}[/bold] assigned to [yellow]{e.stack.stack}[/yellow] (Seat {e.seat_number})"
            + gap_block +
            f"\n\n[dim]VibeSDK system prompt:[/dim]\n[green]{e.system_prompt}[/green]",
            title=f"[cyan]{e.project_name}[/cyan]",
            border_style="dim",
            expand=False,
        ))



# ─────────────────────────────────────────────────────────────────────────────
# Memory views
# ─────────────────────────────────────────────────────────────────────────────

def show_all_projects():
    """List every project saved in memory."""
    projects = list_projects()
    if not projects:
        console.print("[yellow]No projects saved yet.[/yellow]")
        return

    table = Table(
        title="All Saved Projects",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Project",          style="bold yellow")
    table.add_column("Stacks",           justify="center", style="cyan")
    table.add_column("Employees Hired",  justify="center", style="green")
    table.add_column("Saved At",         style="dim")

    for p in projects:
        table.add_row(
            p["name"],
            str(p["total_stacks"]),
            str(p["total_assigned"]),
            p["created_at"][:19].replace("T", " "),
        )
    console.print()
    console.print(table)


def show_project_from_memory(project_name: str):
    """Load and display a saved roster from memory."""
    roster = load_roster(project_name)
    if not roster:
        console.print(f"[red]No saved roster found for '{project_name}'.[/red]")
        return
    display_roster(roster)


def show_project_team(project_name: str):
    """Show every employee assigned to a project, grouped by stack."""
    team = get_project_team(project_name)
    if not team:
        console.print(f"[yellow]No team found for project '{project_name}'.[/yellow]")
        return

    table = Table(
        title=f"Team — {project_name}",
        box=box.ROUNDED,
        border_style="magenta",
        show_lines=True,
    )
    table.add_column("Stack",         style="bold yellow")
    table.add_column("Seat",          justify="center", style="cyan")
    table.add_column("Name",          style="bold")
    table.add_column("Experience",    justify="center")
    table.add_column("Cosine",        justify="center", style="magenta")
    table.add_column("Gap Score",     justify="center", style="blue")
    table.add_column("Missing Skills",style="red")
    table.add_column("Assigned At",   style="dim")

    import json
    current_stack = None
    for m in team:
        # Visual separator between stacks
        if m["stack"] != current_stack:
            current_stack = m["stack"]

        gap_score    = m.get("gap_score")
        gap_str      = f"{gap_score:.2f}" if gap_score is not None else "—"
        missing_raw  = m.get("missing_skills") or "[]"
        try:
            missing = ", ".join(json.loads(missing_raw)) or "none"
        except Exception:
            missing = missing_raw

        table.add_row(
            m["stack"],
            str(m["seat_number"]),
            m["name"],
            f"{m['experience']:.0f}y",
            f"{m['cosine_score']:.3f}",
            gap_str,
            missing,
            m["created_at"][:19].replace("T", " "),
        )
    console.print()
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold cyan]What would you like to do?[/bold cyan]",
        border_style="cyan"
    ))

    action = Prompt.ask(
        "Action",
        choices=["run", "list", "load", "history"],
        default="run",
    )

    # ── Run a new project ─────────────────────────────────────────────────────
    if action == "run":
        project = prompt_project()
        display_project_summary(project)

        if not Confirm.ask("[bold]Run retrieval for this project?[/bold]", default=True):
            console.print("[yellow]Aborted.[/yellow]")
            return

        pipeline = RetrievalPipeline()
        roster   = pipeline.run_project(project)   # auto-saves to memory
        display_roster(roster)

    # ── List all saved projects ───────────────────────────────────────────────
    elif action == "list":
        show_all_projects()

    # ── Load a saved project roster ───────────────────────────────────────────
    elif action == "load":
        show_all_projects()
        project_name = Prompt.ask("\nEnter project name to load")
        show_project_from_memory(project_name)

    # ── Show all employees hired for a project ───────────────────────────────
    elif action == "history":
        show_all_projects()
        project_name = Prompt.ask("\nEnter project name to see its team")
        show_project_team(project_name)

    # ── Loop ──────────────────────────────────────────────────────────────────
    if Confirm.ask("\n[bold]Do something else?[/bold]", default=False):
        main()


if __name__ == "__main__":
    main()