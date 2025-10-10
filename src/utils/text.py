
from __future__ import annotations
import re

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()
