"""
models.py
─────────
Dataclasses for the core domain objects.

ProjectStack     — one role within a project (stack + desc + seats)
Project          — a project made up of multiple stacks
AssignedEmployee — result of the retrieval pipeline for one stack seat
SkillGap         — missing skills + gap score for an employee vs their assigned stack
ProjectRoster    — complete hiring record for a project (project + all assigned employees)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectStack:
    """
    A single role requirement within a project.

    Attributes
    ----------
    stack       : Role title, e.g. "React Frontend Engineer"
    description : What the role entails — used to generate the query string
    num_roles   : How many employees are needed for this stack
    """
    stack:       str
    description: str
    num_roles:   int = 1

    def to_dict(self) -> dict:
        return {
            "stack":       self.stack,
            "description": self.description,
            "num_roles":   self.num_roles,
        }


@dataclass
class Project:
    """
    A project that needs staffing across one or more stacks.

    Attributes
    ----------
    name   : Project name, e.g. "NextGen E-Commerce Platform"
    stacks : List of ProjectStack objects — one per role type needed

    Example
    -------
    project = Project(
        name="NextGen E-Commerce Platform",
        stacks=[
            ProjectStack(
                stack="React Frontend Engineer",
                description="Build performant UIs with Next.js and TypeScript.",
                num_roles=2,
            ),
            ProjectStack(
                stack="Python Backend Engineer",
                description="Design microservices with FastAPI and PostgreSQL.",
                num_roles=1,
            ),
        ]
    )

    # Iterate over every seat across all stacks
    for stack, seat_number in project.iter_seats():
        print(f"{stack.stack} — seat {seat_number}")
    """
    name:   str
    stacks: list[ProjectStack] = field(default_factory=list)

    def add_stack(self, stack: ProjectStack):
        """Add a stack requirement to this project."""
        self.stacks.append(stack)

    def total_roles(self) -> int:
        """Total number of employees needed across all stacks."""
        return sum(s.num_roles for s in self.stacks)

    def iter_seats(self):
        """
        Yields (ProjectStack, seat_number) for every individual seat.

        e.g. a stack with num_roles=2 yields that stack twice,
        with seat_number 1 and 2. This drives the retrieval loop —
        each seat needs its own assigned employee.
        """
        for stack in self.stacks:
            for seat in range(1, stack.num_roles + 1):
                yield stack, seat

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "stacks":      [s.to_dict() for s in self.stacks],
            "total_roles": self.total_roles(),
        }


@dataclass
class SkillGap:
    """
    Skill gap analysis for an AssignedEmployee vs their ProjectStack.
    Calculated at assignment time via Groq.

    Attributes
    ----------
    missing_skills  : Technologies the stack needs that the employee lacks entirely
    partial_skills  : Skills they have but need to deepen
                      e.g. ["Docker — has basics, needs production depth"]
    gap_score       : 0.0 (massive gap) → 1.0 (perfect fit)
                      Reflects how ready they are for this specific role
    growth_areas    : Short prose summary of what to focus on
    """
    missing_skills: list[str]       = field(default_factory=list)
    partial_skills: list[str]       = field(default_factory=list)
    gap_score:      float           = 0.0
    growth_areas:   str             = ""

    def has_gaps(self) -> bool:
        return bool(self.missing_skills or self.partial_skills)

    def summary(self) -> str:
        lines = [f"Gap score : {self.gap_score:.2f} / 1.00"]
        if self.missing_skills:
            lines.append(f"Missing   : {', '.join(self.missing_skills)}")
        if self.partial_skills:
            lines.append(f"Partial   : {', '.join(self.partial_skills)}")
        if self.growth_areas:
            lines.append(f"Focus on  : {self.growth_areas}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "gap_score":      self.gap_score,
            "missing_skills": self.missing_skills,
            "partial_skills": self.partial_skills,
            "growth_areas":   self.growth_areas,
        }


@dataclass
class AssignedEmployee:
    """
    The result of the retrieval pipeline for one stack seat.

    Attributes
    ----------
    project_name  : Name of the project
    stack         : The ProjectStack this employee was matched to
    seat_number   : Which seat within that stack (1-based)
    emp_id        : Employee ID from SQLite / Pinecone
    name          : Employee name
    experience    : Years of experience
    cosine_score  : Similarity score from Pinecone (0–1)
    stack_profile : LLM-generated ideal candidate string used as query
    system_prompt : Personalised VibeSDK system prompt generated at assignment
                    time. Passed to the LLM on every VibeSDK call so it knows
                    exactly who it's pair-programming with — their skill levels,
                    past projects, and how to guide them in this role.
    skill_gap     : SkillGap analysis vs the assigned ProjectStack.
                    Contains missing skills, partial skills, gap score, and
                    growth areas. Also injected into the VibeSDK system prompt.
    """
    project_name:  str
    stack:         ProjectStack
    seat_number:   int
    emp_id:        str
    name:          str
    experience:    float
    cosine_score:  float
    stack_profile: str
    system_prompt: str      = ""
    skill_gap:     "SkillGap" = field(default_factory=lambda: SkillGap())
    is_available:  bool     = True

    def display_label(self) -> str:
        return (
            f"{self.stack.stack} — Seat {self.seat_number} "
            f"→ {self.name} ({self.experience}y) "
            f"[cosine: {self.cosine_score:.3f}]"
        )


@dataclass
class ProjectRoster:
    """
    The complete hiring record for a project after retrieval is done.

    Ties a Project to all its AssignedEmployees so you always know:
      - which project
      - which stack + seat each person was hired for
      - how many seats were filled vs required

    Attributes
    ----------
    project        : The Project dataclass this roster belongs to
    assigned       : List of AssignedEmployee — one per filled seat
    unfilled_seats : List of (ProjectStack, seat_number) where no
                     available employee was found

    Example
    -------
    roster = ProjectRoster(project=project)
    roster.add(assigned_employee)

    print(roster.summary())
    for entry in roster.by_stack("React Frontend Engineer"):
        print(entry.name)
    """
    project:        "Project"
    assigned:       list["AssignedEmployee"] = field(default_factory=list)
    unfilled_seats: list[tuple]              = field(default_factory=list)

    def add(self, employee: "AssignedEmployee"):
        """Add a successfully assigned employee to the roster."""
        self.assigned.append(employee)

    def mark_unfilled(self, stack: "ProjectStack", seat: int):
        """Record a seat that could not be filled."""
        self.unfilled_seats.append((stack, seat))

    def by_stack(self, stack_title: str) -> list["AssignedEmployee"]:
        """Return all assigned employees for a specific stack title."""
        return [e for e in self.assigned if e.stack.stack == stack_title]

    def seats_filled(self) -> int:
        return len(self.assigned)

    def seats_required(self) -> int:
        return self.project.total_roles()

    def is_fully_staffed(self) -> bool:
        return self.seats_filled() == self.seats_required()

    def summary(self) -> str:
        status = "✅ Fully staffed" if self.is_fully_staffed() else "⚠️  Partially staffed"
        return (
            f"Project : {self.project.name}\n"
            f"Status  : {status}\n"
            f"Filled  : {self.seats_filled()} / {self.seats_required()} seats\n"
        )

    def to_dict(self) -> dict:
        return {
            "project":  self.project.to_dict(),
            "assigned": [
                {
                    "emp_id":        e.emp_id,
                    "name":          e.name,
                    "stack":         e.stack.stack,
                    "seat":          e.seat_number,
                    "experience":    e.experience,
                    "cosine_score":  e.cosine_score,
                    "system_prompt": e.system_prompt,
                    "skill_gap":     e.skill_gap.to_dict(),
                }
                for e in self.assigned
            ],
            "unfilled": [
                {"stack": s.stack, "seat": seat}
                for s, seat in self.unfilled_seats
            ],
        }