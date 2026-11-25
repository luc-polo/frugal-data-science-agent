"""Evaluate the agent on a fixed tuning or holdout split of dev.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from sklearn.model_selection import train_test_split

from agent import Agent
from dsh import to_number

SPLIT_SEED = 42
#Fixed parameters never touched
TASKS_PER_BATCH = 8
MODEL_LOAD_LIMIT_S = 60.0
SOLVE_LIMIT_S = 40.0
DEV_PATH = Path(__file__).with_name("dev.json")


def _family(task: dict) -> str:
    """takes a task detailed label from dev.json, like "table.join", and returns the broad category, "table"""
    return task["type"].split(".", 1)[0]


def _split_tasks(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    tuning, holdout = train_test_split(
        tasks,
        test_size=0.2,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=[_family(task) for task in tasks],
    )
    return tuning, holdout


def _prepare_task(task: dict, directory: Path) -> dict:
    """Write the attachments and pass the agent only the harness fields."""
    directory.mkdir()
    file_paths = []
    for name, contents in task["files"].items():
        if Path(name).name != name:
            raise ValueError(f"Expected an attachment filename, got {name!r}")
        path = directory / name
        path.write_text(contents, encoding="utf-8")
        file_paths.append(str(path))
    return {"id": task["id"], "objective": task["objective"], "files": file_paths}


def _is_correct(answer: object, task: dict) -> bool:
    try:
        value = to_number(answer)
    except (ValueError, OverflowError):
        return False
    return abs(value - task["answer"]) <= task["tol"]


def _print_time(stage: str, seconds: float, limit: float) -> None:
    status = f" (over {limit:g} s limit)" if seconds > limit else ""
    print(f"{stage}: {seconds:.2f} s{status}")


def evaluate(tasks: list[dict], model_log: Path | None = None) -> None:
    """Run the selected tasks and report accuracy and wall-clock time."""
    if not tasks:
        raise ValueError("No tasks to evaluate")
    if model_log is not None and model_log.exists():
        raise FileExistsError(f"Model log already exists: {model_log}")

    started = perf_counter()
    agent = Agent(log_model_io=model_log is not None)
    _print_time("Model load", perf_counter() - started, MODEL_LOAD_LIMIT_S)

    correct = Counter()
    totals = Counter()
    failed_ids = []
    log_records = []
    for start in range(0, len(tasks), TASKS_PER_BATCH):
        batch = tasks[start:start + TASKS_PER_BATCH]
        with TemporaryDirectory(prefix="dsh-dev-") as temp_dir:
            inputs = [_prepare_task(task, Path(temp_dir) / str(i)) for i, task in enumerate(batch)]
            started = perf_counter()
            replies = agent.solve(inputs)
            elapsed = perf_counter() - started

        if model_log is not None:
            log_records.extend(agent.last_batch_log)

        batch_number = start // TASKS_PER_BATCH + 1
        _print_time(f"Batch {batch_number} ({len(batch)} tasks)", elapsed, SOLVE_LIMIT_S)
        answers = {reply["id"]: reply.get("answer") for reply in replies}
        for task in batch:
            passed = _is_correct(answers.get(task["id"]), task)
            if not passed:
                failed_ids.append(task["id"])
            for group in ("overall", _family(task), task["type"]):
                totals[group] += 1
                correct[group] += passed

    print("\nAccuracy")
    groups = ["overall"] + sorted(group for group in totals if group != "overall")
    for group in groups:
        print(f"{group}: {correct[group]}/{totals[group]} ({correct[group] / totals[group]:.1%})")
    print(f"\nFailed task IDs: {', '.join(failed_ids) or 'none'}")

    if model_log is not None:
        with model_log.open("x", encoding="utf-8") as file:
            json.dump(log_records, file, ensure_ascii=False, indent=2)
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("tuning", "holdout"), default="tuning")
    parser.add_argument("--model-log", type=Path, help="Save prompts and replies to one JSON file")
    args = parser.parse_args()
    tuning, holdout = _split_tasks(json.loads(DEV_PATH.read_text(encoding="utf-8")))
    tasks = tuning if args.split == "tuning" else holdout
    print(f"Evaluating {len(tasks)} {args.split} tasks", flush=True)
    evaluate(tasks, model_log=args.model_log)


if __name__ == "__main__":
    main()
