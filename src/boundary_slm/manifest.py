from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Iterable

from boundary_slm.io import write_json


DEFAULT_CLAIM_AUDIT_NOTES = [
    "Legacy strict-suite output is tier_3_exploratory_only.",
    "Most strong claims failed the previous preregistered gates.",
    "Legacy artifacts are retained as backup evidence and negative-result context, not as support for new paper claims.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_legacy_manifest(legacy_root: Path) -> dict[str, object]:
    files = []
    for path in sorted(legacy_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(legacy_root).as_posix()
        files.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_root": str(legacy_root),
        "file_count": len(files),
        "files": files,
        "claim_audit_notes": DEFAULT_CLAIM_AUDIT_NOTES,
        "important_legacy_pointers": {
            "strict_claims_report": "phase2/outputs/strict_suite/strict_claims_report.json",
            "research_snapshot": "phase2/outputs/research_snapshot.json",
            "temperature_claim_audit": "publication_ready/temperature_refinement_public_repo/CLAIM_AUDIT.md",
            "context_claim_audit": "publication_ready/context_uncertainty_public_repo/CLAIM_AUDIT.md",
        },
    }


def write_legacy_manifest(legacy_root: Path) -> Path:
    output_path = legacy_root / "MANIFEST.json"
    write_json(output_path, build_legacy_manifest(legacy_root))
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build legacy archive manifest.")
    parser.add_argument("--legacy-root", type=Path, default=Path("old/legacy_temperature_context"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    print(write_legacy_manifest(args.legacy_root))


if __name__ == "__main__":
    main()

