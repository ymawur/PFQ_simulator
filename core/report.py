from __future__ import annotations

from datetime import datetime
from typing import Mapping


def export_report(section_title: str, metrics: Mapping[str, float], notes: str) -> bytes:
    timestamp = datetime.utcnow().isoformat()
    lines = [f"# {section_title}", f"Generated: {timestamp}", "", "## Metrics"]
    for name, value in metrics.items():
        lines.append(f"- {name}: {value:.4f}")
    if notes:
        lines.extend(["", "## Notes", notes])
    return "\n".join(lines).encode("utf-8")
