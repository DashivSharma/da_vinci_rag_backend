"""
embeddings.py
─────────────
Flow per employee:
    1. skills (dict) + proj_details (str)
          ↓
    2. Ollama / llama3  →  concise, semantically rich summary string
          ↓
    3. all-MiniLM-L6-v2  →  384-dim normalised float vector

Why LLM first?
    Raw skill lists are noisy ("React, JS, html, css...").
    The LLM synthesises them into coherent prose that the embedding model
    can encode far more meaningfully for semantic retrieval.
"""

import json
import requests
from sentence_transformers import SentenceTransformer


# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


# ── Prompt ────────────────────────────────────────────────────────────────────
SUMMARY_PROMPT = """\
You are a technical recruiter assistant.

Given an employee's skills and project history, write a single concise paragraph (4-6 sentences) that captures:
- What technologies and domains they specialise in
- The kind of problems they have solved
- Their seniority and depth of experience

Be specific and technical. Do NOT include the employee's name. Do NOT add any intro or outro — output only the paragraph.

Skills:
{skills}

Project history:
{proj_details}
"""


class EmployeeEmbedder:
    """
    Generates embeddings for employees using a two-stage pipeline:
        Ollama (llama3) → summary text → all-MiniLM-L6-v2 → vector

    Usage
    -----
    embedder = EmployeeEmbedder()

    # Single employee dict
    summary, vector = embedder.embed_employee(employee)

    # Single raw text (for stack queries — no LLM needed)
    vector = embedder.embed_text("React developer with Next.js experience...")
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, seed: int = 42):
        # Fix random seed so embedding weights are deterministic across runs
        import numpy as np, random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass  # torch not installed, sentence-transformers will still work

        print(f"[Embedder] Loading {self.MODEL_NAME} (seed={seed})...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dimension = 384
        print(f"[Embedder] Ready — output dim: {self.dimension}")

    # ── Stage 1: LLM summary ─────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(employee: dict) -> str:
        """Format the Ollama prompt from employee skills + proj_details."""
        skills = employee.get("skills", {})
        if isinstance(skills, str):
            skills = json.loads(skills)

        # Pretty-print skills dict so the LLM sees it clearly
        skills_text = "\n".join(
            f"  {cat}: {', '.join(v) if isinstance(v, list) else v}"
            for cat, v in skills.items()
        )

        return SUMMARY_PROMPT.format(
            skills=skills_text,
            proj_details=employee.get("proj_details", "").strip(),
        )

    @staticmethod
    def generate_summary(employee: dict) -> str:
        """
        Calls Ollama locally to generate a rich summary string for the employee.
        Returns the summary as a plain string.
        Raises RuntimeError if Ollama is unreachable.
        """
        prompt = EmployeeEmbedder._build_prompt(employee)

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,        # wait for full response
                    "options": {
                        "temperature": 0.2, # low temp = consistent, factual
                        "num_predict": 200, # ~4-6 sentences is enough
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            summary = response.json()["response"].strip()
            return summary

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama at localhost:11434. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out after 60s.")
        except KeyError:
            raise RuntimeError(f"Unexpected Ollama response: {response.text}")

    # ── Stage 2: Embed text ──────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """
        Embed any string directly with all-MiniLM-L6-v2.
        Used for stack / query embeddings (no LLM needed for those).
        Returns a normalised 384-dim float list.
        """
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_many_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Batch embed a list of strings. Efficient for indexing multiple stacks."""
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return vectors.tolist()

    # ── Full pipeline ────────────────────────────────────────────────────────

    def embed_employee(self, employee: dict) -> tuple[str, list[float]]:
        """
        Full two-stage pipeline for a single employee.

        Returns
        -------
        summary : str          — the LLM-generated summary (store this in DB)
        vector  : list[float]  — 384-dim embedding of that summary
        """
        summary = self.generate_summary(employee)
        vector  = self.embed_text(summary)
        return summary, vector

    def embed_stack(self, stack: dict) -> list[float]:
        """
        Embed a project stack for querying.
        No LLM call needed — the stack description is already well-formed.
        """
        query_text = (
            f"{stack['stack']}. "
            f"{stack.get('description', '')}"
        )
        return self.embed_text(query_text)


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    embedder = EmployeeEmbedder()

    sample_employee = {
        "name": "Alice Chen",
        "experience": 5,
        "skills": {
            "frontend": ["React", "TypeScript", "Next.js", "Tailwind"],
            "backend":  ["Node.js", "Express"],
            "tools":    ["Jest", "Git", "Figma"],
        },
        "proj_details": (
            "Built a high-traffic e-commerce storefront with React and Next.js, "
            "improving LCP by 40%. Led migration from class components to hooks. "
            "Created a reusable component library used across 3 products."
        ),
    }

    sample_stack = {
        "stack":       "React Frontend Engineer",
        "description": "Senior React developer with TypeScript and Next.js experience, "
                       "comfortable with component libraries and performance optimisation.",
    }

    print("\n── Generating LLM summary for employee...")
    summary, emp_vec = embedder.embed_employee(sample_employee)

    print(f"\n── LLM Summary ──\n{summary}")
    print(f"\n── Vector dim: {len(emp_vec)}  |  first 5: {[round(v,4) for v in emp_vec[:5]]}")

    stack_vec = embedder.embed_stack(sample_stack)
    similarity = float(np.dot(emp_vec, stack_vec))
    print(f"\n── Cosine similarity (employee ↔ stack): {similarity:.4f}")