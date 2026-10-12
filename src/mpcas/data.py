"""Input validation and task-table assembly. No patient data is bundled."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


META = {"subject", "cohort", "center", "diagnosis", "subtype"}
FORBIDDEN = {"label", "label_raw", "label_binary", "diagnosis", "diagnosis_label", "cdr_label", "cohort", "subtype", "center", "subject_code", "device_id", "inner_subject", "versions", "moca", "cdr"}


def _protected_feature_name(name: str) -> bool:
    parts = name.lower().split("__")
    return any(part in FORBIDDEN or part.startswith(("cdr_", "moca_", "diagnosis_", "subtype_")) for part in parts)


def read_metadata(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, low_memory=False)
    original = {"Subject_ID", "cohort_planA", "CDR_label"}
    if original.issubset(source.columns):
        mapped = pd.DataFrame({
            "subject": source["Subject_ID"],
            "cohort": source["cohort_planA"].map({"cohort1": "discovery", "cohort2": "validation"}),
            "diagnosis": source["CDR_label"].astype(str).str.upper().map({"NCI": 0, "MCI": 1, "DEMENTIA": 2, "DE": 2}),
        })
        if "来源" in source:
            mapped["center"] = source["来源"]
        if "sub_MCI_label" in source:
            mapped["subtype"] = source["sub_MCI_label"].map({"单域aMCI": "single_aMCI", "多域aMCI": "multi_aMCI", "其他MCI": "other_MCI"})
        if mapped[["subject", "cohort", "diagnosis"]].isna().any().any():
            raise ValueError("Original metadata contains unknown subjects, cohorts, or CDR labels")
        return mapped
    return source


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input column names")
    required = {"subject", "cohort", "diagnosis"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    frame["subject"] = frame["subject"].astype("string").str.strip()
    if frame["subject"].isna().any() or frame["subject"].eq("").any() or frame["subject"].duplicated().any():
        raise ValueError("Subjects must be unique and nonempty")
    frame["cohort"] = frame["cohort"].astype("string").str.strip().str.lower()
    if not frame["cohort"].isin(["discovery", "validation"]).all():
        raise ValueError("Cohort must be discovery or validation")
    frame["diagnosis"] = pd.to_numeric(frame["diagnosis"], errors="raise")
    if frame["diagnosis"].isna().any() or not frame["diagnosis"].isin([0, 1, 2]).all():
        raise ValueError("Diagnosis must be 0=NCI, 1=MCI, or 2=dementia")
    if "subtype" not in frame:
        frame["subtype"] = pd.NA
    if frame.loc[frame.diagnosis.eq(1), "subtype"].dropna().isin(["single_aMCI", "multi_aMCI", "other_MCI"]).eq(False).any():
        raise ValueError("Unknown MCI subtype")
    features = [c for c in frame if c not in META]
    if not features or any(_protected_feature_name(c) for c in features):
        raise ValueError("No features or a protected label/identifier column is present")
    if any("__" not in c or not c.split("__", 1)[0].startswith("T") for c in features):
        raise ValueError("Feature names must start with task ID and '__', e.g. T006__speech_rate")
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    return frame.sort_values("subject").reset_index(drop=True)


def assemble_tasks(task_dir: Path, metadata_csv: Path, output_csv: Path, tasks: list[str]) -> None:
    metadata = read_metadata(metadata_csv)
    required = {"subject", "cohort", "diagnosis"}
    if not required.issubset(metadata):
        raise ValueError(f"Metadata missing {sorted(required - set(metadata))}")
    metadata = metadata[[c for c in ["subject", "cohort", "center", "diagnosis", "subtype"] if c in metadata]].copy()
    metadata["subject"] = metadata["subject"].astype(str).str.strip()
    if metadata.subject.duplicated().any():
        raise ValueError("Duplicate subjects in metadata")
    result = metadata.set_index("subject")
    expected = set(result.index)
    for task in tasks:
        source = pd.read_csv(task_dir / f"{task}.csv", low_memory=False)
        if "subject" not in source:
            raise ValueError(f"{task} has no subject column")
        source["subject"] = source["subject"].astype(str).str.strip()
        if source.subject.duplicated().any() or set(source.subject) != expected:
            raise ValueError(f"{task} subject IDs do not match metadata")
        columns = [c for c in source if c not in {"subject"} and not _protected_feature_name(c) and c not in META]
        numeric = source[columns].apply(pd.to_numeric, errors="coerce")
        columns = [c for c in columns if numeric[c].notna().any()]
        numeric = numeric[columns]
        numeric.columns = [f"{task}__{c}" for c in columns]
        numeric.index = source.subject
        result = result.join(numeric, validate="one_to_one")
    result = result.reset_index()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    load_table(output_csv)


def outcome_rows(frame: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, np.ndarray]:
    diagnosis = frame.diagnosis.astype(int)
    if outcome == "NCI_vs_CI":
        keep, positive = diagnosis.isin([0, 1, 2]), diagnosis.ne(0)
    elif outcome in {"NCI_vs_nonNCI", "MCI_vs_nonMCI", "DE_vs_nonDE"}:
        label = {"NCI_vs_nonNCI": 0, "MCI_vs_nonMCI": 1, "DE_vs_nonDE": 2}[outcome]
        keep, positive = diagnosis.isin([0, 1, 2]), diagnosis.eq(label)
    elif outcome in {"NCI_vs_MCI", "NCI_vs_DE", "MCI_vs_DE"}:
        negative, pos = {"NCI_vs_MCI": (0, 1), "NCI_vs_DE": (0, 2), "MCI_vs_DE": (1, 2)}[outcome]
        keep, positive = diagnosis.isin([negative, pos]), diagnosis.eq(pos)
    elif outcome in {"single_aMCI", "multi_aMCI", "other_MCI"}:
        keep, positive = diagnosis.eq(1) & frame.subtype.notna(), frame.subtype.eq(outcome)
    else:
        raise ValueError(f"Unknown outcome: {outcome}")
    return frame.loc[keep].reset_index(drop=True), positive.loc[keep].astype(int).to_numpy()


def task_features(frame: pd.DataFrame, tasks: list[str]) -> list[str]:
    prefixes = tuple(f"{task}__" for task in tasks)
    columns = [c for c in frame if c.startswith(prefixes)]
    if not columns:
        raise ValueError(f"No feature columns for tasks {tasks}")
    return columns
