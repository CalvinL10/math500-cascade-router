"""Configure pytest local paths and project-local temp directory on F:."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_TMP_ROOT = PROJECT_ROOT / ".pytest_tmp"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config) -> None:
    """Keep pytest's base temp directory inside the project directory."""

    if config.option.basetemp is None:
        config.option.basetemp = str(LOCAL_TMP_ROOT)
