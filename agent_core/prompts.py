"""Build compact prompts for routed data-science tasks."""

from __future__ import annotations


ROUTE_GUIDANCE = {
    "calc": "Use standard Python, SymPy, or NumPy for calculations.",
    "table": "Use pandas to read, clean, and analyse the tables.",
    "fit": "Use pandas for data preparation and scikit-learn for regression.",
}


def build_prompt(
    objective: str,
    route: str,
    data_card: str | None = None,
) -> list[dict[str, str]]:
    """Build one compact LLM prompt, split between the model's required system and user roles (format required by the Qwen model used)"""
    if route not in ROUTE_GUIDANCE:
        raise ValueError(f"Unknown route: {route!r}")

    system_lines = [
        "Write a short Python program that solves the objective.",
        "Return executable Python code only, without explanations or Markdown fences.",
        (
            "Follow every detail stated in the objective, including any parsing, missing-value, "
            "deduplication, join, filter, leakage-exclusion, model, and rounding instructions."
        ),
        "Assign the final answer as one finite numeric scalar to RESULT.",
        ROUTE_GUIDANCE[route],
    ]

    if route != "calc":
        system_lines.append(
            "Open attached files by basename, use the data dictionary to map columns, "
            "and compute from the complete files rather than the sample rows."
        )

    user_content = f"OBJECTIVE:\n{objective}"
    if route != "calc" and data_card:
        user_content += f"\n\n{data_card}"

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": user_content},
    ]
