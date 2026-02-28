"""
memory.py
─────────
SQLite-backed memory store for projects, stacks, assigned employees,
and skill gaps. Auto-saved transparently inside RetrievalPipeline.run_project().

Schema
------
projects
    id          INTEGER PK
    name        TEXT UNIQUE
    created_at  TEXT

project_stacks
    id          INTEGER PK
    project_id  INTEGER → projects.id
    stack       TEXT
    description TEXT
    num_roles   INTEGER

assigned_employees
    id            INTEGER PK
    project_id    INTEGER → projects.id
    stack_id      INTEGER → project_stacks.id
    emp_id        TEXT
    name          TEXT
    experience    REAL
    seat_number   INTEGER
    cosine_score  REAL
    stack_profile TEXT
    system_prompt TEXT
    is_available  INTEGER
    created_at    TEXT

skill_gaps
    id             INTEGER PK
    assigned_id    INTEGER → assigned_employees.id  (1-to-1)
    gap_score      REAL
    missing_skills TEXT    (JSON array)
    partial_skills TEXT    (JSON array)
    growth_areas   TEXT

Queries exposed
---------------
save_roster(roster)                  → None
load_roster(project_name)            → ProjectRoster | None
list_projects()                      → list[dict]
get_employee_assignments(emp_id)     → list[dict]
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from models import (
    Project, ProjectStack,
    AssignedEmployee, ProjectRoster, SkillGap,
)

DB_PATH = Path(__file__).parent / "memory.db"


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # safer concurrent writes
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Initialise schema
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist. Safe to call repeatedly."""
    conn = _connect()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL UNIQUE,
            created_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_stacks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER NOT NULL REFERENCES projects(id),
            stack       TEXT    NOT NULL,
            description TEXT    NOT NULL,
            num_roles   INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS assigned_employees (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id    INTEGER NOT NULL REFERENCES projects(id),
            stack_id      INTEGER NOT NULL REFERENCES project_stacks(id),
            emp_id        TEXT    NOT NULL,
            name          TEXT    NOT NULL,
            experience    REAL    NOT NULL,
            seat_number   INTEGER NOT NULL,
            cosine_score  REAL    NOT NULL,
            stack_profile TEXT    NOT NULL,
            system_prompt TEXT    NOT NULL,
            is_available  INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skill_gaps (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            assigned_id    INTEGER NOT NULL UNIQUE REFERENCES assigned_employees(id),
            gap_score      REAL    NOT NULL DEFAULT 0.0,
            missing_skills TEXT    NOT NULL DEFAULT '[]',
            partial_skills TEXT    NOT NULL DEFAULT '[]',
            growth_areas   TEXT    NOT NULL DEFAULT ''
        );
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_roster(roster: ProjectRoster) -> None:
    """
    Persist a ProjectRoster to SQLite.
    If the project already exists it is overwritten (stacks + assignments
    are deleted and re-inserted) so re-running a project stays idempotent.

    Called automatically inside RetrievalPipeline.run_project().
    """
    init_db()
    conn  = _connect()
    c     = conn.cursor()
    now   = datetime.now(timezone.utc).isoformat()

    try:
        # ── Upsert project ────────────────────────────────────────────────────
        c.execute(
            "INSERT INTO projects (name, created_at) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET created_at=excluded.created_at",
            (roster.project.name, now),
        )
        project_id = c.execute(
            "SELECT id FROM projects WHERE name=?", (roster.project.name,)
        ).fetchone()["id"]

        # ── Clear old stacks + assignments for this project (re-run) ─────────
        old_stack_ids = [
            r["id"] for r in
            c.execute("SELECT id FROM project_stacks WHERE project_id=?",
                      (project_id,)).fetchall()
        ]
        if old_stack_ids:
            placeholders = ",".join("?" * len(old_stack_ids))
            old_assigned_ids = [
                r["id"] for r in
                c.execute(
                    f"SELECT id FROM assigned_employees WHERE stack_id IN ({placeholders})",
                    old_stack_ids,
                ).fetchall()
            ]
            if old_assigned_ids:
                ap = ",".join("?" * len(old_assigned_ids))
                c.execute(f"DELETE FROM skill_gaps WHERE assigned_id IN ({ap})",
                          old_assigned_ids)
                c.execute(f"DELETE FROM assigned_employees WHERE id IN ({ap})",
                          old_assigned_ids)
            c.execute(
                f"DELETE FROM project_stacks WHERE id IN ({placeholders})",
                old_stack_ids,
            )

        # ── Insert stacks ─────────────────────────────────────────────────────
        stack_id_map: dict[str, int] = {}   # stack.stack title → db id
        for stack in roster.project.stacks:
            c.execute(
                "INSERT INTO project_stacks (project_id, stack, description, num_roles) "
                "VALUES (?, ?, ?, ?)",
                (project_id, stack.stack, stack.description, stack.num_roles),
            )
            stack_id_map[stack.stack] = c.lastrowid

        # ── Insert assigned employees + skill gaps ────────────────────────────
        for emp in roster.assigned:
            stack_db_id = stack_id_map.get(emp.stack.stack)

            c.execute(
                """INSERT INTO assigned_employees
                   (project_id, stack_id, emp_id, name, experience, seat_number,
                    cosine_score, stack_profile, system_prompt, is_available, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id, stack_db_id, emp.emp_id, emp.name,
                    emp.experience, emp.seat_number, emp.cosine_score,
                    emp.stack_profile, emp.system_prompt,
                    int(emp.is_available), now,
                ),
            )
            assigned_id = c.lastrowid

            gap = emp.skill_gap
            c.execute(
                """INSERT INTO skill_gaps
                   (assigned_id, gap_score, missing_skills, partial_skills, growth_areas)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    assigned_id,
                    gap.gap_score,
                    json.dumps(gap.missing_skills),
                    json.dumps(gap.partial_skills),
                    gap.growth_areas,
                ),
            )

        conn.commit()
        print(f"[Memory] ✅ Saved roster for '{roster.project.name}' "
              f"({len(roster.assigned)} employees)")

    except Exception as e:
        conn.rollback()
        print(f"[Memory] ❌ Failed to save roster: {e}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_roster(project_name: str) -> ProjectRoster | None:
    """
    Load a saved ProjectRoster from SQLite by project name.
    Returns None if the project has never been saved.
    """
    init_db()
    conn = _connect()
    c    = conn.cursor()

    # ── Find project ──────────────────────────────────────────────────────────
    proj_row = c.execute(
        "SELECT * FROM projects WHERE name=?", (project_name,)
    ).fetchone()

    if not proj_row:
        conn.close()
        return None

    project_id = proj_row["id"]

    # ── Rebuild Project + stacks ──────────────────────────────────────────────
    stack_rows = c.execute(
        "SELECT * FROM project_stacks WHERE project_id=?", (project_id,)
    ).fetchall()

    stacks: list[ProjectStack] = []
    stack_map: dict[int, ProjectStack] = {}
    for row in stack_rows:
        ps = ProjectStack(
            stack       = row["stack"],
            description = row["description"],
            num_roles   = row["num_roles"],
        )
        stacks.append(ps)
        stack_map[row["id"]] = ps

    project = Project(name=project_name, stacks=stacks)
    roster  = ProjectRoster(project=project)

    # ── Rebuild assigned employees ────────────────────────────────────────────
    emp_rows = c.execute(
        "SELECT * FROM assigned_employees WHERE project_id=? ORDER BY id",
        (project_id,),
    ).fetchall()

    for emp_row in emp_rows:
        # Skill gap
        gap_row = c.execute(
            "SELECT * FROM skill_gaps WHERE assigned_id=?", (emp_row["id"],)
        ).fetchone()

        skill_gap = SkillGap(
            gap_score      = gap_row["gap_score"]                       if gap_row else 0.0,
            missing_skills = json.loads(gap_row["missing_skills"] or "[]") if gap_row else [],
            partial_skills = json.loads(gap_row["partial_skills"] or "[]") if gap_row else [],
            growth_areas   = gap_row["growth_areas"]                    if gap_row else "",
        )

        stack = stack_map.get(emp_row["stack_id"])
        if not stack:
            continue   # orphaned row, skip

        emp = AssignedEmployee(
            project_name  = project_name,
            stack         = stack,
            seat_number   = emp_row["seat_number"],
            emp_id        = emp_row["emp_id"],
            name          = emp_row["name"],
            experience    = emp_row["experience"],
            cosine_score  = emp_row["cosine_score"],
            stack_profile = emp_row["stack_profile"],
            system_prompt = emp_row["system_prompt"],
            skill_gap     = skill_gap,
            is_available  = bool(emp_row["is_available"]),
        )
        roster.add(emp)

    conn.close()
    return roster


# ─────────────────────────────────────────────────────────────────────────────
# List projects
# ─────────────────────────────────────────────────────────────────────────────

def list_projects() -> list[dict]:
    """
    Return all projects ever saved.

    Returns
    -------
    list of dicts:
        { name, created_at, total_stacks, total_assigned }
    """
    init_db()
    conn = _connect()

    rows = conn.execute("""
        SELECT
            p.name,
            p.created_at,
            COUNT(DISTINCT ps.id)  AS total_stacks,
            COUNT(DISTINCT ae.id)  AS total_assigned
        FROM projects p
        LEFT JOIN project_stacks     ps ON ps.project_id = p.id
        LEFT JOIN assigned_employees ae ON ae.project_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """).fetchall()

    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Employee assignment history
# ─────────────────────────────────────────────────────────────────────────────

def get_employee_assignments(emp_id: str) -> list[dict]:
    """
    Look up every project an employee has been assigned to.

    Returns
    -------
    list of dicts:
        { project_name, stack, seat_number, cosine_score,
          gap_score, created_at }
    ordered most-recent first.
    """
    init_db()
    conn = _connect()

    rows = conn.execute("""
        SELECT
            p.name          AS project_name,
            ps.stack        AS stack,
            ae.seat_number,
            ae.cosine_score,
            ae.created_at,
            sg.gap_score
        FROM assigned_employees ae
        JOIN projects       p  ON p.id  = ae.project_id
        JOIN project_stacks ps ON ps.id = ae.stack_id
        LEFT JOIN skill_gaps sg ON sg.assigned_id = ae.id
        WHERE ae.emp_id = ?
        ORDER BY ae.created_at DESC
    """, (emp_id,)).fetchall()

    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Project team lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_project_team(project_name: str) -> list[dict]:
    """
    Return all employees assigned to a project, grouped by stack.

    Returns
    -------
    list of dicts:
        { emp_id, name, experience, stack, seat_number,
          cosine_score, gap_score, missing_skills, created_at }
    ordered by stack then seat number.
    """
    init_db()
    conn = _connect()

    rows = conn.execute("""
        SELECT
            ae.emp_id,
            ae.name,
            ae.experience,
            ps.stack,
            ae.seat_number,
            ae.cosine_score,
            ae.created_at,
            sg.gap_score,
            sg.missing_skills
        FROM assigned_employees ae
        JOIN projects       p  ON p.id  = ae.project_id
        JOIN project_stacks ps ON ps.id = ae.stack_id
        LEFT JOIN skill_gaps sg ON sg.assigned_id = ae.id
        WHERE p.name = ?
        ORDER BY ps.stack, ae.seat_number
    """, (project_name,)).fetchall()

    conn.close()
    return [dict(r) for r in rows]