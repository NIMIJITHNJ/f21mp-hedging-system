from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import subprocess
from typing import Iterable


@dataclass
class ValValidationResult:
    status: str
    valid: bool | None
    validator_path: str | None
    domain_file: str
    problem_file: str
    plan_file: str
    stdout: str
    stderr: str
    return_code: int | None


def find_val_executable() -> Path | None:
    env_path = os.environ.get("VAL_VALIDATE")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path("tools/val/Validate.exe"),
        Path("tools/val/Validate"),
        Path("tools/val/VAL.exe"),
        Path("tools/val/VAL"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def normalise_action(action: str) -> str:
    action = action.strip()
    if not action:
        return ""
    return action.lower()


def write_plan_file(action_sequence: Iterable[str], plan_file: Path) -> Path:
    plan_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for action in action_sequence:
        cleaned = normalise_action(action)
        if cleaned:
            lines.append(f"({cleaned})")

    plan_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan_file


def validate_plan(
    domain_file: str | Path,
    problem_file: str | Path,
    action_sequence: Iterable[str],
    plan_file: str | Path | None = None,
    timeout_seconds: int = 10,
) -> ValValidationResult:
    domain_path = Path(domain_file)
    problem_path = Path(problem_file)

    if plan_file is None:
        plan_path = problem_path.parent / "plan.val"
    else:
        plan_path = Path(plan_file)

    write_plan_file(action_sequence, plan_path)

    validator = find_val_executable()

    if validator is None:
        return ValValidationResult(
            status="SKIPPED_VAL_NOT_FOUND",
            valid=None,
            validator_path=None,
            domain_file=str(domain_path),
            problem_file=str(problem_path),
            plan_file=str(plan_path),
            stdout="",
            stderr="VAL executable was not found. Set VAL_VALIDATE or place Validate.exe under tools/val/.",
            return_code=None,
        )

    completed = subprocess.run(
        [str(validator), str(domain_path), str(problem_path), str(plan_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    combined_output = (completed.stdout + "\n" + completed.stderr).lower()

    valid = (
        completed.returncode == 0
        and (
            "plan valid" in combined_output
            or "successful plans" in combined_output
            or "plan executed successfully" in combined_output
        )
    )

    status = "VALID" if valid else "INVALID_OR_UNKNOWN"

    return ValValidationResult(
        status=status,
        valid=valid,
        validator_path=str(validator),
        domain_file=str(domain_path),
        problem_file=str(problem_path),
        plan_file=str(plan_path),
        stdout=completed.stdout,
        stderr=completed.stderr,
        return_code=completed.returncode,
    )


def result_as_dict(result: ValValidationResult) -> dict:
    return asdict(result)
