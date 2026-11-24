"""Entry point and task routing for the data-science agent."""

from __future__ import annotations

from dsh import load_llm


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _route_task(objective: str, file_paths: list[str]) -> str:
    """Choose the solver route for one task.

    The routing rules are deliberately simple:

    1. A task without files is a calculation task.
    2. A task with files that mentions fitting or regression is a fit task.
    3. Every other task with files is a table task.

    The function returns ``"calc"``, ``"fit"``, or ``"table"``.
    """
    if not file_paths:
        return "calc"

    objective_lower = objective.lower()
    fit_terms = ("fit ", "regression")

    if any(term in objective_lower for term in fit_terms):
        return "fit"

    return "table"


class Agent:
    """Coordinate model generation and the deterministic pipeline."""

    def __init__(self):
        self.llm = load_llm(MODEL_ID)

    def generate_programs(
        self,
        conversations: list[list[dict[str, str]]],
        max_time: float | None = None,
    ) -> list[str]:
        """Generate one Python program for each conversation."""
        return self.llm.chat(
            conversations,
            max_new_tokens=512,
            temperature=0.0,
            batch_size=8,
            thinking=False,
            max_time=max_time,
        )
