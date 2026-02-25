#!/usr/bin/env python
"""Quick test to write churches to Google Sheets."""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
code_dir = project_root / "code"
sys.path.insert(0, str(code_dir))

from spreadsheet_tool import GoogleSheetsSpreadsheetTool


def _load_env_if_present() -> None:
    """Load key/value pairs from code/.env if the file exists."""
    env_path = code_dir / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ[key] = value


_load_env_if_present()

churches = [
    {
        "name": "St Mark's Episcopal Church",
        "address": "Palo Alto, CA",
        "phone": "",
        "website": "",
        "hours": "",
    },
    {
        "name": "St. Thomas Aquinas Parish",
        "address": "Palo Alto, CA",
        "phone": "",
        "website": "",
        "hours": "",
    },
    {
        "name": "All Saints' Episcopal Church Palo Alto",
        "address": "Palo Alto, CA",
        "phone": "",
        "website": "",
        "hours": "",
    },
    {
        "name": "First Presbyterian Church Palo Alto",
        "address": "Palo Alto, CA",
        "phone": "",
        "website": "",
        "hours": "",
    },
    {
        "name": "Palo Alto Vineyard Church",
        "address": "Palo Alto, CA",
        "phone": "",
        "website": "",
        "hours": "",
    },
]

print("Writing churches to Google Sheets...")
print(f"Found {len(churches)} churches\n")

try:
    tool = GoogleSheetsSpreadsheetTool()
    churches_json = json.dumps(churches)
    result = tool(churches_json)
    print(f"Success.\n{result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
