"""
retrieval.py
────────────
Retrieval pipeline for matching employees to a project's stacks.

Full flow per stack seat:
    1.  ProjectStack { stack, description }
                ↓
    2.  Groq / llama3  →  ideal candidate profile string
                ↓
    3.  all-MiniLM-L6-v2  →  384-dim query vector
                ↓
    4.  Pinecone cosine search  →  top-K candidates
                ↓
    5.  Pick best available employee
                ↓
    6.  Groq / llama3  →  VibeSDK system prompt
                ↓
        AssignedEmployee
"""

import json
import requests
from pinecone import Pinecone
from groq import Groq

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, GROQ_API_KEY, GROQ_MODEL
from embeddings import EmployeeEmbedder
from models import Project, ProjectStack, AssignedEmployee, ProjectRoster, SkillGap
from memory import save_roster


TOP_K = 5

# Groq client (reused across all calls)
_groq_client = Groq(api_key=GROQ_API_KEY)


# ── Prompts ───────────────────────────────────────────────────────────────────

STACK_PROFILE_PROMPT = """\
You are a technical recruiter assistant.

Given a project stack and its description, write a single concise paragraph (3-5 sentences) \
describing the IDEAL candidate for this role.

Focus on:
- The specific technologies and frameworks they must know
- The type of systems or problems they should have built before
- Relevant depth of experience

Be specific and technical. Output ONLY the paragraph, no intro or outro.

Stack: {stack}
Description: {description}
"""

VIBESDK_GENERATOR_PROMPT = """\
You are generating a system prompt for an AI pair-programming assistant (VibeSDK).

Given this developer's profile, write a system prompt that tells the AI assistant:
1. Who this developer is — their skill level per technology (infer: <2y = beginner, 2-5y = mid, 5y+ = senior based on experience and what they've built)
2. Their past projects as concrete context the assistant can reference during sessions
3. How to pair-program with them specifically — collaborative, think out loud, guide don't dictate

Output ONLY the system prompt text. It will be passed directly to an LLM. No intro, no labels, no markdown.

Developer name: {name}
Years of experience: {experience}
Skills: {skills}
Past projects: {proj_details}
Assigned role: {stack} on project "{project_name}"
Role description: {stack_description}
"""


SKILL_GAP_PROMPT = """\
You are a senior technical lead doing a skills gap analysis.

Given an employee's skills and the requirements of their assigned role, identify:
1. missing_skills  — technologies the role needs that the employee does not have at all
2. partial_skills  — technologies they have but need to deepen for this role (one line each explaining the gap)
3. gap_score       — a float from 0.0 (massive gap, not ready) to 1.0 (perfect fit, no gaps)
4. growth_areas    — one sentence: the single most important area to develop for this role

Respond in this EXACT format and nothing else:
missing_skills: <comma-separated list, or "none">
partial_skills: <comma-separated list of "skill — reason", or "none">
gap_score: <float>
growth_areas: <one sentence>

Employee skills:
{skills}

Employee project history:
{proj_details}

Assigned role: {stack}
Role description: {description}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Groq helper
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str, temperature: float = 0.2, max_tokens: int = 150) -> str:
    """
    Shared Groq LLM call using the official SDK.
    Uses the chat completions API (OpenAI-compatible).
    """
    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — LLM: stack → ideal candidate profile string
# ─────────────────────────────────────────────────────────────────────────────

def generate_stack_profile(stack: ProjectStack) -> str:
    """Calls Groq to generate an ideal candidate description for this stack."""
    prompt = STACK_PROFILE_PROMPT.format(
        stack=stack.stack,
        description=stack.description,
    )
    return _call_groq(prompt, temperature=0.2, max_tokens=150)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Pinecone: query vector → top-K candidates
# ─────────────────────────────────────────────────────────────────────────────

def query_pinecone(index, query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """Queries Pinecone and returns top-K candidates with their metadata."""
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    candidates = []
    for match in results["matches"]:
        meta = match["metadata"]
        candidates.append({
            "emp_id":       match["id"],
            "score":        match["score"],
            "name":         meta.get("name", ""),
            "experience":   float(meta.get("experience", 0)),
            "is_available": bool(meta.get("is_available", False)),
            "skills":       json.loads(meta["skills"])
                            if isinstance(meta.get("skills"), str)
                            else meta.get("skills", {}),
            "proj_details": meta.get("proj_details", ""),
        })

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Pick best available (highest cosine score who is available)
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_available(
    candidates: list[dict],
    exclude_ids: set[str] | None = None,
) -> dict | None:
    """
    Pinecone already returns candidates in cosine score order.
    Walk top-to-bottom and return the first available employee
    who is not already assigned to another seat on this project.

    Args:
        candidates  : Pinecone results ordered by cosine score
        exclude_ids : emp_ids already assigned — skip these
    """
    exclude_ids = exclude_ids or set()
    return next(
        (c for c in candidates
         if c.get("is_available") and c["emp_id"] not in exclude_ids),
        None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — LLM: employee profile → VibeSDK system prompt
# ─────────────────────────────────────────────────────────────────────────────

def generate_vibesdk_system_prompt(
    best: dict,
    stack: ProjectStack,
    project_name: str,
    skill_gap: "SkillGap",
) -> str:
    """
    Calls Groq to generate a personalised VibeSDK system prompt.
    Includes skill gap context so the LLM knows exactly where to guide
    this developer during pair-programming sessions.
    """
    skills = best.get("skills", {})
    if isinstance(skills, str):
        skills = json.loads(skills)

    skills_flat = "\n".join(
        f"  - {cat}: {', '.join(v) if isinstance(v, list) else v}"
        for cat, v in skills.items()
    )

    # Build skill gap section to inject into the system prompt
    gap_section = ""
    if skill_gap.has_gaps():
        lines = [f"SKILL GAPS FOR THIS ROLE (gap score: {skill_gap.gap_score:.2f}/1.00):"]
        if skill_gap.missing_skills:
            lines.append(f"  Missing entirely : {', '.join(skill_gap.missing_skills)}")
        if skill_gap.partial_skills:
            lines.append("  Needs deepening  :")
            for ps in skill_gap.partial_skills:
                lines.append(f"    - {ps}")
        if skill_gap.growth_areas:
            lines.append(f"  Priority focus   : {skill_gap.growth_areas}")
        lines.append(
            "When these topics come up naturally in the session, "
            "use them as teaching moments — guide, don't just fix."
        )
        gap_section = "\n".join(lines)

    prompt = VIBESDK_GENERATOR_PROMPT.format(
        name              = best["name"],
        experience        = best["experience"],
        skills            = skills_flat,
        proj_details      = best.get("proj_details", ""),
        stack             = stack.stack,
        project_name      = project_name,
        stack_description = stack.description,
    )

    # Append gap section to the prompt so LLM weaves it into the system prompt
    if gap_section:
        prompt += f"\n\nSkill gap context to include:\n{gap_section}"

    return _call_groq(prompt, temperature=0.3, max_tokens=500)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — LLM: employee + stack → SkillGap
# ─────────────────────────────────────────────────────────────────────────────

def generate_skill_gap(best: dict, stack: ProjectStack) -> SkillGap:
    """
    Calls Groq to analyse the gap between the employee's current skills
    and what the assigned ProjectStack actually requires.

    Parses the LLM's structured response into a SkillGap dataclass.
    Falls back to an empty SkillGap if parsing fails so the pipeline
    never crashes on a malformed response.

    Returns
    -------
    SkillGap with missing_skills, partial_skills, gap_score, growth_areas
    """
    skills = best.get("skills", {})
    if isinstance(skills, str):
        skills = json.loads(skills)

    skills_flat = "\n".join(
        f"  - {cat}: {', '.join(v) if isinstance(v, list) else v}"
        for cat, v in skills.items()
    )

    prompt = SKILL_GAP_PROMPT.format(
        skills       = skills_flat,
        proj_details = best.get("proj_details", ""),
        stack        = stack.stack,
        description  = stack.description,
    )

    raw = _call_groq(prompt, temperature=0.1, max_tokens=200)

    # ── Parse structured response ─────────────────────────────────────────────
    def _parse_list(line: str) -> list[str]:
        val = line.split(":", 1)[-1].strip()
        if not val or val.lower() == "none":
            return []
        return [item.strip() for item in val.split(",") if item.strip()]

    def _parse_float(line: str) -> float:
        import re
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", line)
        if match:
            return min(1.0, max(0.0, float(match.group())))
        return 0.0

    missing_skills: list[str] = []
    partial_skills: list[str] = []
    gap_score:      float     = 0.0
    growth_areas:   str       = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("missing_skills:"):
            missing_skills = _parse_list(line)
        elif line.lower().startswith("partial_skills:"):
            partial_skills = _parse_list(line)
        elif line.lower().startswith("gap_score:"):
            gap_score = _parse_float(line)
        elif line.lower().startswith("growth_areas:"):
            growth_areas = line.split(":", 1)[-1].strip()

    return SkillGap(
        missing_skills = missing_skills,
        partial_skills = partial_skills,
        gap_score      = gap_score,
        growth_areas   = growth_areas,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    Orchestrates the full retrieval flow for a Project.

    Pinecone API key and index name are loaded automatically from .env
    via config.py — you never pass them manually.

    Usage
    -----
    pipeline = RetrievalPipeline()

    roster = pipeline.run_project(project)       # full project
    result = pipeline.run_stack(stack, "My Project", seat=1)  # single stack
    """

    def __init__(self):
        self.embedder = EmployeeEmbedder()
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)
        print(f"[Pipeline] Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
        print(f"[Pipeline] Using Groq model '{GROQ_MODEL}'")

    def run_stack(
        self,
        stack: ProjectStack,
        project_name: str,
        seat: int = 1,
        top_k: int = TOP_K,
        exclude_ids: set[str] | None = None,
    ) -> AssignedEmployee | None:
        """
        Run the full pipeline for a single stack seat.
        Returns an AssignedEmployee or None if no one is available.

        Args:
            exclude_ids : emp_ids already assigned on this project — never pick these
        """
        print(f"\n[Pipeline] Stack: '{stack.stack}' — Seat {seat}/{stack.num_roles}")

        # Stage 1 — LLM: generate ideal candidate profile
        stack_profile = generate_stack_profile(stack)
        print(f"  ↳ Profile: {stack_profile[:100]}...")

        # Stage 2 — Embed profile + query Pinecone
        # Fetch top_k + len(exclude_ids) so we still get top_k valid candidates
        # even after excluding already-assigned employees
        fetch_k      = top_k + len(exclude_ids or set())
        query_vector = self.embedder.embed_text(stack_profile)
        candidates   = query_pinecone(self.index, query_vector, top_k=fetch_k)
        print(f"  ↳ Pinecone returned {len(candidates)} candidates "
              f"(excluding {len(exclude_ids or set())} already assigned)")

        # Stage 3 — Pick best available by cosine score, skipping assigned employees
        best = pick_best_available(candidates, exclude_ids=exclude_ids)

        if not best:
            print(f"  ↳ ⚠️  No available candidate found.")
            return None

        print(f"  ↳ ✅ {best['name']} (cosine: {best['score']:.3f})")

        # Stage 4 — Skill gap analysis
        print(f"  ↳ Calculating skill gap for {best['name']}...")
        skill_gap = generate_skill_gap(best, stack)
        print(f"  ↳ Gap score: {skill_gap.gap_score:.2f} | "
              f"Missing: {skill_gap.missing_skills or 'none'} | "
              f"Partial: {len(skill_gap.partial_skills)} skill(s)")

        # Stage 5 — Generate VibeSDK system prompt (includes skill gap context)
        print(f"  ↳ Generating VibeSDK system prompt for {best['name']}...")
        system_prompt = generate_vibesdk_system_prompt(best, stack, project_name, skill_gap)
        print(f"  ↳ System prompt ready ({len(system_prompt)} chars)")

        return AssignedEmployee(
            project_name  = project_name,
            stack         = stack,
            seat_number   = seat,
            emp_id        = best["emp_id"],
            name          = best["name"],
            experience    = best["experience"],
            cosine_score  = best["score"],
            stack_profile = stack_profile,
            system_prompt = system_prompt,
            skill_gap     = skill_gap,
            is_available  = best["is_available"],
        )

    def run_project(self, project: Project, top_k: int = TOP_K) -> ProjectRoster:
        """
        Run the pipeline for every seat across all stacks in a Project.
        Returns a ProjectRoster — the complete hiring record for the project.
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] Project: {project.name}")
        print(f"[Pipeline] Total seats: {project.total_roles()}")
        print(f"{'='*60}")

        roster      = ProjectRoster(project=project)
        assigned_ids: set[str] = set()   # tracks emp_ids assigned so far

        for stack, seat in project.iter_seats():
            result = self.run_stack(
                stack, project.name, seat, top_k,
                exclude_ids=assigned_ids,
            )
            if result:
                roster.add(result)
                assigned_ids.add(result.emp_id)   # block this person for all future seats
                print(f"  ↳ [{len(assigned_ids)} assigned so far: {assigned_ids}]")
            else:
                roster.mark_unfilled(stack, seat)

        print(f"\n[Pipeline] Done — {roster.seats_filled()}/{roster.seats_required()} seats filled.")
        if roster.unfilled_seats:
            for stack, seat in roster.unfilled_seats:
                print(f"[Pipeline] ❌ Unfilled: {stack.stack} seat {seat}")

        # Auto-save to SQLite memory
        save_roster(roster)

        return roster