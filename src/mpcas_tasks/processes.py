#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Subprocess helpers shared by pipeline stages."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    from mpcas_tasks.stage_logging import PIPELINE_VERBOSE, StageLogger
except ImportError:  # pragma: no cover - supports direct script execution
    from stage_logging import PIPELINE_VERBOSE, StageLogger


def run_command(
    cmd: List[str],
    cwd: Path,
    skip_existing: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    logger: Optional[StageLogger] = None,
) -> None:
    if skip_existing is not None and skip_existing.exists():
        if PIPELINE_VERBOSE:
            print(f"[SKIP] exists: {skip_existing}")
        if logger:
            logger.log(f"[SKIP] exists: {skip_existing}")
        return
    if PIPELINE_VERBOSE:
        print("[RUN]", " ".join(str(x) for x in cmd))
    if logger:
        logger.log("[RUN] " + " ".join(str(x) for x in cmd))
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=run_env)
