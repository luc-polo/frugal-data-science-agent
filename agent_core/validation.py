"""Validate answers returned by generated programs."""

from dsh import ExecResult, to_number


def validate_result(result: ExecResult) -> float:
    """Return a finite number or raise an error explaining the failure."""
    if not result.ok:
        raise ValueError(result.error or "The program failed")

    try:
        return to_number(result.value)
    except OverflowError as error:
        raise ValueError("RESULT is outside the finite float range") from error
