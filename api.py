"""
api.py
──────
FastAPI application for the Talent RAG system.

Endpoints
---------
GET  /health                          — liveness check
GET  /projects                        — list all saved projects
GET  /projects/{name}                 — load full roster for a project
GET  /projects/{name}/team            — team members for a project
POST /projects/run                    — run retrieval pipeline for a new project

Run
---
    uvicorn api:app --reload --port 8000

Docs
----
    http://localhost:8000/docs          (Swagger UI)
    http://localhost:8000/redoc         (ReDoc)
"""

# ── Suppress noisy third-party warnings before any imports ───────────────────
import warnings
import os
import logging

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HuggingFace tokenizer warning

# Silence overly verbose loggers from dependencies
for _noisy in (
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "pinecone",
    "httpx",
    "httpcore",
    "urllib3",
    "multipart",
    "passlib",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ── Logger ────────────────────────────────────────────────────────────────────
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR  = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def _build_logger(name: str) -> logging.Logger:
    logger    = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above only
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(formatter)

    # Rotating file handler — DEBUG and above, max 5 MB × 3 backups
    file_h = RotatingFileHandler(
        LOG_DIR / "api.log",
        maxBytes    = 5 * 1024 * 1024,
        backupCount = 3,
        encoding    = "utf-8",
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    logger.addHandler(console_h)
    logger.addHandler(file_h)
    logger.propagate = False
    return logger

logger = _build_logger("talent_rag.api")

# ── FastAPI & stdlib imports ───────────────────────────────────────────────────
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models  import Project, ProjectStack
from memory  import list_projects, load_roster, get_project_team
from retrieval import RetrievalPipeline


# ── Pydantic request / response schemas ──────────────────────────────────────

class StackIn(BaseModel):
    stack:       str = Field(..., description="Role title, e.g. 'React Frontend Engineer'")
    description: str = Field(..., description="What the role entails")
    num_roles:   int = Field(default=1, ge=1, description="Number of seats needed")


class RunProjectRequest(BaseModel):
    name:   str            = Field(..., description="Project name")
    stacks: list[StackIn]  = Field(..., min_length=1, description="List of stacks to fill")


class SkillGapOut(BaseModel):
    gap_score:      float
    missing_skills: list[str]
    partial_skills: list[str]
    growth_areas:   str


class AssignedEmployeeOut(BaseModel):
    emp_id:        str
    name:          str
    stack:         str
    seat:          int
    experience:    float
    cosine_score:  float
    system_prompt: str
    skill_gap:     SkillGapOut


class UnfilledSeatOut(BaseModel):
    stack: str
    seat:  int


class RosterOut(BaseModel):
    project_name:    str
    total_required:  int
    total_filled:    int
    is_fully_staffed: bool
    assigned:        list[AssignedEmployeeOut]
    unfilled:        list[UnfilledSeatOut]


class ProjectSummaryOut(BaseModel):
    name:           str
    created_at:     str
    total_stacks:   int
    total_assigned: int


class TeamMemberOut(BaseModel):
    emp_id:        str
    name:          str
    stack:         str
    seat_number:   int
    experience:    float
    cosine_score:  float
    gap_score:     float | None
    missing_skills: list[str]
    assigned_at:   str


class HealthOut(BaseModel):
    status:  str
    version: str


# ── App lifecycle ─────────────────────────────────────────────────────────────

_pipeline: RetrievalPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise the pipeline once (loads embedding model + Pinecone).
    Shutdown: log graceful exit.
    """
    global _pipeline
    logger.info("Starting Talent RAG API...")
    try:
        _pipeline = RetrievalPipeline()
        logger.info("RetrievalPipeline initialised successfully")
    except Exception as exc:
        logger.critical("Failed to initialise pipeline: %s", exc)
        raise

    yield

    logger.info("Shutting down Talent RAG API")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Talent RAG API",
    description = (
        "Semantic employee-project matching powered by Pinecone + llama3. "
        "Assigns employees to project stacks, calculates skill gaps, and "
        "generates personalised VibeSDK system prompts."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start      = time.perf_counter()

    logger.info(
        "→ [%s] %s %s",
        request_id, request.method, request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("[%s] Unhandled error: %s", request_id, exc)
        return JSONResponse(
            status_code = 500,
            content     = {"detail": "Internal server error", "request_id": request_id},
        )

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "← [%s] %s %s  %d  %.1fms",
        request_id, request.method, request.url.path,
        response.status_code, elapsed,
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s: %s",
                     request.method, request.url.path, exc)
    return JSONResponse(
        status_code = 500,
        content     = {"detail": str(exc)},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_pipeline() -> RetrievalPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Pipeline not initialised yet — try again in a moment",
        )
    return _pipeline


def _roster_to_out(roster) -> RosterOut:
    """Convert a ProjectRoster dataclass to the RosterOut Pydantic schema."""
    import json

    assigned = []
    for e in roster.assigned:
        assigned.append(AssignedEmployeeOut(
            emp_id        = e.emp_id,
            name          = e.name,
            stack         = e.stack.stack,
            seat          = e.seat_number,
            experience    = e.experience,
            cosine_score  = e.cosine_score,
            system_prompt = e.system_prompt,
            skill_gap     = SkillGapOut(
                gap_score      = e.skill_gap.gap_score,
                missing_skills = e.skill_gap.missing_skills,
                partial_skills = e.skill_gap.partial_skills,
                growth_areas   = e.skill_gap.growth_areas,
            ),
        ))

    unfilled = [
        UnfilledSeatOut(stack=s.stack, seat=seat)
        for s, seat in roster.unfilled_seats
    ]

    return RosterOut(
        project_name     = roster.project.name,
        total_required   = roster.seats_required(),
        total_filled     = roster.seats_filled(),
        is_fully_staffed = roster.is_fully_staffed(),
        assigned         = assigned,
        unfilled         = unfilled,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthOut,
    summary        = "Health check",
    tags           = ["System"],
)
async def health():
    """Returns service liveness. Pipeline status is implicitly healthy if this responds."""
    logger.debug("Health check called")
    return HealthOut(status="ok", version=app.version)


@app.get(
    "/projects",
    response_model = list[ProjectSummaryOut],
    summary        = "List all saved projects",
    tags           = ["Projects"],
)
async def get_projects():
    """
    Returns all projects ever run through the pipeline,
    ordered by most recently saved first.
    """
    logger.info("Listing all projects")
    try:
        rows = list_projects()
        return [
            ProjectSummaryOut(
                name           = r["name"],
                created_at     = r["created_at"],
                total_stacks   = r["total_stacks"],
                total_assigned = r["total_assigned"],
            )
            for r in rows
        ]
    except Exception as exc:
        logger.exception("Failed to list projects: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/projects/{name}",
    response_model = RosterOut,
    summary        = "Load a saved project roster",
    tags           = ["Projects"],
)
async def get_project(name: str):
    """
    Loads the full saved roster for a project by name,
    including all assigned employees, skill gaps, and system prompts.
    """
    logger.info("Loading roster for project '%s'", name)
    try:
        roster = load_roster(name)
    except Exception as exc:
        logger.exception("Failed to load roster for '%s': %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if roster is None:
        logger.warning("Project '%s' not found in memory", name)
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"Project '{name}' not found",
        )

    return _roster_to_out(roster)


@app.get(
    "/projects/{name}/team",
    response_model = list[TeamMemberOut],
    summary        = "Get team members for a project",
    tags           = ["Projects"],
)
async def get_team(name: str):
    """
    Returns all employees assigned to a project,
    ordered by stack then seat number.
    """
    import json

    logger.info("Fetching team for project '%s'", name)
    try:
        rows = get_project_team(name)
    except Exception as exc:
        logger.exception("Failed to get team for '%s': %s", name, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not rows:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"No team found for project '{name}'",
        )

    team = []
    for r in rows:
        missing_raw = r.get("missing_skills") or "[]"
        try:
            missing = json.loads(missing_raw)
        except Exception:
            missing = []

        team.append(TeamMemberOut(
            emp_id         = r["emp_id"],
            name           = r["name"],
            stack          = r["stack"],
            seat_number    = r["seat_number"],
            experience     = r["experience"],
            cosine_score   = r["cosine_score"],
            gap_score      = r.get("gap_score"),
            missing_skills = missing,
            assigned_at    = r["created_at"][:19].replace("T", " "),
        ))

    return team


@app.post(
    "/projects/run",
    response_model = RosterOut,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Run retrieval pipeline for a new project",
    tags           = ["Projects"],
)
async def run_project(body: RunProjectRequest):
    """
    Runs the full retrieval pipeline for a project:
    - Generates ideal candidate profiles via llama3
    - Queries Pinecone for top matches
    - Calculates skill gaps
    - Generates VibeSDK system prompts
    - Auto-saves the roster to SQLite memory

    Returns the complete ProjectRoster with assigned employees.
    """
    pipeline = _get_pipeline()

    logger.info(
        "Running pipeline for project '%s' (%d stack(s), %d total seat(s))",
        body.name,
        len(body.stacks),
        sum(s.num_roles for s in body.stacks),
    )

    # Build domain objects from request
    project = Project(name=body.name)
    for s in body.stacks:
        project.add_stack(ProjectStack(
            stack       = s.stack,
            description = s.description,
            num_roles   = s.num_roles,
        ))

    try:
        roster = pipeline.run_project(project)   # auto-saves to SQLite
    except RuntimeError as exc:
        # Ollama / Pinecone errors — client-actionable
        logger.error("Pipeline error for '%s': %s", body.name, exc)
        raise HTTPException(
            status_code = status.HTTP_502_BAD_GATEWAY,
            detail      = str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error running pipeline for '%s': %s", body.name, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(
        "Pipeline complete for '%s' — %d/%d seats filled",
        body.name, roster.seats_filled(), roster.seats_required(),
    )

    return _roster_to_out(roster)

    