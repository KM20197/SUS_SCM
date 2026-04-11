from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_outputs(fp: Dict[str, Any]) -> Dict[str, str]:
    out = fp.get("outputs", {})
    return {str(k): str(v) for k, v in out.items()} if isinstance(out, dict) else {}


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    lines = []
    ok = True

    a_raw = a.get("raw_sha256")
    b_raw = b.get("raw_sha256")
    if a_raw != b_raw:
        ok = False
        lines.append(f"- raw_sha256 differs:\n    A: {a_raw}\n    B: {b_raw}")
    else:
        lines.append(f"- raw_sha256: OK ({a_raw})")

    a_out = _as_outputs(a)
    b_out = _as_outputs(b)

    keys = sorted(set(a_out) | set(b_out))
    if not keys:
        ok = False
        lines.append("- outputs: EMPTY (no hashes found in one or both fingerprints)")
        return ok, "\n".join(lines)

    changed = []
    missing_a = []
    missing_b = []
    same = 0

    for k in keys:
        va = a_out.get(k)
        vb = b_out.get(k)
        if va is None:
            ok = False
            missing_a.append(k)
            continue
        if vb is None:
            ok = False
            missing_b.append(k)
            continue
        if va != vb:
            ok = False
            changed.append((k, va, vb))
        else:
            same += 1

    lines.append(f"- outputs: {same} identical / {len(keys)} in union")

    if missing_a:
        lines.append("  Missing in A:")
        lines.extend([f"    - {k}" for k in missing_a])

    if missing_b:
        lines.append("  Missing in B:")
        lines.extend([f"    - {k}" for k in missing_b])

    if changed:
        lines.append("  Changed hashes:")
        for k, va, vb in changed:
            lines.append(f"    - {k}\n        A: {va}\n        B: {vb}")

    return ok, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to fingerprint JSON A")
    ap.add_argument("--b", required=True, help="Path to fingerprint JSON B")
    args = ap.parse_args()

    pa = Path(args.a)
    pb = Path(args.b)

    a = _load_json(pa)
    b = _load_json(pb)

    ok, report = compare(a, b)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
