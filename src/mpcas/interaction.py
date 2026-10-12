"""Extract numeric task-interaction metrics from response.json records."""

from __future__ import annotations

from pathlib import Path

from mpcas_tasks.features import build_interaction_table
from mpcas_tasks.records import discover_phase_records


def extract_interaction(data_root: Path, output_csv: Path, tasks: list[str], versions: list[str]) -> None:
    records = discover_phase_records(
        data_root=data_root, tasks=tasks, versions=versions,
        phases=["formal", "formalcolor", "practice", "practicecolor"],
        include_t001=False,
    )
    if not records:
        raise FileNotFoundError("No task phase records found under the supplied raw data root")
    result = build_interaction_table(records)
    if result.empty:
        raise ValueError("No response.json files found in the selected task phases")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
