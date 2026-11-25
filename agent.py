"""Entry point and task routing for the data-science agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait

from agent_core.config import (
    EXECUTION_RESERVE_S,
    LLM_BATCH_SIZE,
    MAX_NEW_TOKENS,
    MAX_WORKERS,
    MODEL_ID,
    PROGRAM_TIMEOUT_S,
    SAFETY_MARGIN_S,
)
from agent_core.executor import execute_program
from agent_core.profiler import build_data_card
from agent_core.prompts import build_prompt
from agent_core.validation import validate_result
from dsh import Clock, load_llm


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


def _execute_answer(reply: str, file_paths: list[str], clock: Clock) -> float:
    """Execute and validate a reply within the time left for this batch."""
    timeout_s = min(PROGRAM_TIMEOUT_S, clock.left())
    if timeout_s <= 0:
        raise TimeoutError("No time left to run the program")

    result = execute_program(reply, file_paths, timeout_s=timeout_s)
    return validate_result(result)


class Agent:
    """Coordinate model generation and the deterministic pipeline."""

    def __init__(self):
        self.llm = load_llm(MODEL_ID)

    def solve(self, tasks: list[dict]) -> list[dict]:
        """Orchestrates all the pipeline to return one numeric answer per task in the original order."""
        clock = Clock(seconds=40.0, margin=SAFETY_MARGIN_S)

        # Build the dictionnary that will store future answer to the 8 tasks of this batch
        answers = [{"id": task["id"], "answer": 0.0} for task in tasks]
        # Build the list that will store the prompts of the 8 tasks of this batch
        conversations = []
        task_indices = []
 
        # Build the prompts and all the input data we will feed the LLM with
        for index, task in enumerate(tasks):
            if clock.left() <= EXECUTION_RESERVE_S:
                break

            try:
                objective = task["objective"]
                file_paths = task["files"]
                # Make out the type of question
                route = _route_task(objective, file_paths)
                # If eligible build a data card of the attached files
                data_card = build_data_card(objective, file_paths) if file_paths else None
                # Build the prompt associated with this question (deterministic)
                conversation = build_prompt(objective, route, data_card)
            except Exception:
                continue

            conversations.append(conversation)
            task_indices.append(index)

        # Reserve these seconds for running and validating the programs after generation.
        generation_time = clock.left() - EXECUTION_RESERVE_S
        if not conversations or generation_time <= 0:
            return answers
        
        # Lunch LLM call for the 8 tasks in parallel 
        try:
            replies = self.generate_programs(conversations, max_time=generation_time)
        except Exception:
            return answers

        if clock.left() <= 0:
            return answers

        # Lunch code execution by batches of 3
        workers = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        try:
            
            pending = {} # Keep each submitted job linked to its original task index.
            for index, reply in zip(task_indices, replies):
                if reply.strip():
                    # Schedules _execute_answer() with the code generated (reply), its task files, and the shared clock. It returns a future immediately.
                    future = workers.submit(_execute_answer, reply, tasks[index]["files"], clock)
                    pending[future] = index
                    # future holds a Python Future object. It’s a handle for one submitted job. The job’s numeric answer is available later through future.result().

            # Wait for the jobs within the time left in this solve() call.
            finished, _ = wait(pending, timeout=max(0.0, clock.left()))
            for future in finished:
                try:
                    answers[pending[future]]["answer"] = future.result()
                except Exception:
                    continue
        finally:
            workers.shutdown(wait=False, cancel_futures=True)

        return answers

    def generate_programs(
        self,
        conversations: list[list[dict[str, str]]],
        max_time: float | None = None,
    ) -> list[str]:
        """Generate one Python reply per prepared conversation."""
        return self.llm.chat(
            conversations,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            batch_size=LLM_BATCH_SIZE,
            thinking=False,
            max_time=max_time,
        )
