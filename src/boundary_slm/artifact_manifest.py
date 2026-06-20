from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_json


DEFAULT_INCLUDE_DIRS = [
    "main/README.md",
    "main/LICENSE-CODE",
    "main/DATA_LICENSE_NOTICE.md",
    "main/configs",
    "main/analysis/raw_tpu_results",
    "main/analysis/paper_metrics",
    "main/analysis/parser_audit",
    "main/analysis/source_manifest",
    "main/analysis/external_benchmark_context",
    "main/analysis/external_evidence",
    "main/analysis/artifact_release_status",
    "main/src/boundary_slm",
    "main/papers/error_persistence_regression_gap",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path, include_dirs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    skip_suffixes = {".aux", ".bcf", ".blg", ".log", ".out", ".run.xml"}
    skip_dirs = {"__pycache__"}
    skip_names = {"f1000_boundary_slm_audit_artifact.tar.gz"}
    internal_release_notes = {
        "PAPER_CONCLUSIONS.md",
        "PAPER_DEVELOPMENT_ROADMAP.md",
        "paper_conclusions.json",
    }
    private_artifact_names = {
        "parser_audit_sample.csv",
        "second_pass_parser_audit_sample.csv",
        "parser_audit_adjudication.csv",
        "parser_audit_labeling_batches.csv",
    }
    skip_path_suffixes = {
        "analysis/raw_tpu_results/per_model",
        "main/analysis/parser_audit/labeling_batches",
    }
    for include in include_dirs:
        base = root / include
        if not base.exists():
            continue
        if base.is_file():
            files.append(base)
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.suffix in skip_suffixes:
                continue
            rel = path.relative_to(root).as_posix()
            if path.name in skip_names:
                continue
            if path.name in internal_release_notes:
                continue
            if path.name in private_artifact_names:
                continue
            if path.name.endswith("_items.csv"):
                continue
            if any(rel.startswith(prefix) for prefix in skip_path_suffixes):
                continue
            files.append(path)
    return sorted(set(files))


def build_manifest(root: Path, include_dirs: Iterable[str]) -> dict[str, Any]:
    files = included_files(root, include_dirs)
    records = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "include_dirs": list(include_dirs),
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "files": records,
        "submission_blockers": [
            "Finalize benchmark/data redistribution terms before applying a data license.",
            "Complete the manual parser-audit human_* columns and regenerate parser_audit_report.json.",
            "Do not include raw model-output JSONL files or item-level response-tail files in public releases.",
        ],
        "public_text_policy": (
            "Manifest defaults exclude main/results, per-model item CSVs, and per-model JSON files that could contain "
            "raw response tails. Public release should contain aggregate/redacted artifacts only."
        ),
    }


def write_tarball(root: Path, files: list[Path], output_tar_gz: Path) -> None:
    output_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix())


def generate(root: Path, output_dir: Path, *, make_archive: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root, DEFAULT_INCLUDE_DIRS)
    manifest_path = output_dir / "F1000_ARTIFACT_MANIFEST.json"
    write_json(manifest_path, manifest)
    if make_archive:
        archive_path = output_dir / "f1000_boundary_slm_audit_artifact.tar.gz"
        write_tarball(root, included_files(root, DEFAULT_INCLUDE_DIRS), archive_path)
        archive_record = {
            "path": archive_path.relative_to(root).as_posix() if archive_path.is_relative_to(root) else str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
        }
        manifest["local_archive"] = archive_record
        write_json(manifest_path, manifest)
    readme = (
        "# Public Artifact Manifest\n\n"
        "This folder contains a local manifest for the GitHub/SSRN proof-of-analysis package.\n"
        "The current release target is a public repository package with hygiene and manifest checks, "
        "not a frozen journal archive.\n\n"
        "Do not present parser-dependent MMLU-Pro claims as confirmatory until the human parser audit is complete.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local manifest for the F1000 artifact package.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("main/artifacts/f1000_boundary_slm_audit"))
    parser.add_argument("--make-archive", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = generate(args.root.resolve(), args.output_dir.resolve(), make_archive=args.make_archive)
    print(json.dumps({"file_count": manifest["file_count"], "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
