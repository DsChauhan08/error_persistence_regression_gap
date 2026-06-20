from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_json


PLACEHOLDERS = {"", "TODO", "TBD", "PLACEHOLDER", "planned", "unknown"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def is_populated(value: Any) -> bool:
    text = str(value or "").strip()
    return text.lower() not in {item.lower() for item in PLACEHOLDERS}


def current_git_commit(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "TODO"
    return proc.stdout.strip() or "TODO"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_cell(value: Any) -> str:
    text = str(value)
    if text.startswith(("http://", "https://")):
        return r"\url{" + text + "}"
    return latex_escape(text)


def release_status(
    *,
    root: Path,
    repository_url: str,
    code_license: str,
    data_output_license: str,
    dependency_file: str,
    expected_runtime: str,
    latest_test_command: str,
    latest_test_result: str,
    parser_gate_path: Path,
    mmlu_gate_path: Path,
    wild_claim_path: Path,
    source_manifest_path: Path,
    hygiene_report_path: Path,
    public_manifest_path: Path,
    archive_identifier: str,
) -> dict[str, Any]:
    parser_gate = read_json(parser_gate_path)
    mmlu_gate = read_json(mmlu_gate_path)
    wild_claim = read_json(wild_claim_path)
    source_manifest = read_json(source_manifest_path)
    hygiene_report = read_json(hygiene_report_path)
    public_manifest = read_json(public_manifest_path)
    parser_validated = bool(parser_gate.get("claim_ready") or parser_gate.get("parser_validated"))
    mmlu_confirmatory = bool(mmlu_gate.get("claim_ready") or mmlu_gate.get("mmlu_pro_confirmatory"))
    wild_ready = bool(wild_claim.get("claim_ready")) and int(wild_claim.get("pairwise_comparison_count") or 0) > 0
    source_ready = (
        bool(source_manifest.get("claim_ready"))
        and int(source_manifest.get("evaluated_item_count") or 0) > 0
        and source_manifest.get("matched_item_count") == source_manifest.get("evaluated_item_count")
        and int(source_manifest.get("mismatch_item_count") or 0) == 0
        and int(source_manifest.get("missing_source_item_count") or 0) == 0
    )
    hygiene_ready = bool(hygiene_report.get("passed")) and not hygiene_report.get("findings")
    manifest_ready = (
        public_manifest_path.exists()
        and int(public_manifest.get("file_count") or 0) > 0
        and int(public_manifest.get("missing") or 0) == 0
        and int(public_manifest.get("mismatches") or 0) == 0
    )
    latest_test_lower = str(latest_test_result or "").lower()
    tests_ready = is_populated(latest_test_result) and "passed" in latest_test_lower and "failed" not in latest_test_lower
    core_claim_ready = wild_ready and source_ready
    archive_ready = is_populated(archive_identifier)
    metadata = {
        "repository_url": repository_url,
        "code_license": code_license,
        "data_output_license": data_output_license,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "dependency_file": dependency_file,
        "expected_runtime": expected_runtime,
        "latest_test_command": latest_test_command,
        "latest_test_result": latest_test_result,
        "archive_identifier": archive_identifier,
    }
    missing = [
        key
        for key in ["repository_url", "code_license", "data_output_license"]
        if not is_populated(metadata.get(key))
    ]
    github_public_ready = not missing and tests_ready and hygiene_ready and manifest_ready and core_claim_ready
    methods_software_article_ready = github_public_ready and archive_ready
    full_empirical_ml_ready = github_public_ready and parser_validated and mmlu_confirmatory and archive_ready
    blockers: list[str] = []
    if missing:
        blockers.append("missing release metadata: " + ", ".join(missing))
    if not tests_ready:
        blockers.append("latest tests are not recorded as passed")
    if not hygiene_ready:
        blockers.append("public-release hygiene scan has not passed")
    if not manifest_ready:
        blockers.append("public-release manifest is missing or not verified")
    if not wild_ready:
        blockers.append("WILD item-level correctness claim gate has not passed")
    if not source_ready:
        blockers.append("MMLU-Pro source-manifest gate has not passed")
    if not archive_ready:
        blockers.append("persistent archive identifier is missing")
    scope_notes: list[str] = []
    if not parser_validated:
        scope_notes.append(
            "Parser validation is not required for the WILD-supported core claim because MMLU-Pro raw-output "
            "results are excluded from the confirmatory claim set."
        )
    if not mmlu_confirmatory:
        scope_notes.append(
            "MMLU-Pro confirmatory model-evaluation claims are not made; that component remains diagnostic."
        )
    if not full_empirical_ml_ready:
        scope_notes.append(
            "Full empirical ML article readiness remains false until parser validation and MMLU-Pro robustness gates pass."
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "github_ssrn_ready": github_public_ready,
        "github_public_ready": github_public_ready,
        "core_claim_ready": core_claim_ready,
        "methods_submission_ready": methods_software_article_ready,
        "methods_software_article_ready": methods_software_article_ready,
        "methods_software_article_status": (
            "ready_for_scoped_methods_software_submission"
            if methods_software_article_ready
            else "not_ready_for_scoped_methods_software_submission"
        ),
        "full_empirical_ml_ready": full_empirical_ml_ready,
        "journal_ready": full_empirical_ml_ready,
        "journal_ready_deprecated_scope": "legacy alias for full_empirical_ml_ready, not the scoped methods/software article gate",
        "mmlu_parser_validated": parser_validated,
        "parser_validated": parser_validated,
        "mmlu_confirmatory_ready": mmlu_confirmatory,
        "mmlu_pro_confirmatory": mmlu_confirmatory,
        "wild_claim_ready": wild_ready,
        "mmlu_pro_source_manifest_ready": source_ready,
        "public_hygiene_ready": hygiene_ready,
        "public_manifest_ready": manifest_ready,
        "archive_ready": archive_ready,
        "tests_ready": tests_ready,
        "parser_gate_path": str(parser_gate_path),
        "mmlu_gate_path": str(mmlu_gate_path),
        "wild_claim_path": str(wild_claim_path),
        "source_manifest_path": str(source_manifest_path),
        "hygiene_report_path": str(hygiene_report_path),
        "public_manifest_path": str(public_manifest_path),
        "parser_gate_status": parser_gate.get("status", "missing"),
        "mmlu_gate_status": mmlu_gate.get("status", "missing"),
        "blockers": blockers,
        "scope_notes": scope_notes,
        "paper_wording_rule": (
            "Use scoped methods/software-article wording when methods_software_article_ready=true. Do not present "
            "MMLU-Pro raw-output accounting as confirmatory until parser validation and robustness gates pass. "
            "Do not use the legacy journal_ready field as a universal article-readiness gate."
        ),
    }


def write_tex(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Repository", status["repository_url"]),
        ("Persistent archive", status["archive_identifier"] if status["archive_ready"] else "not provided"),
        ("Code license", status["code_license"]),
        ("Data-output license", status["data_output_license"]),
        ("Python support", "Python 3.11+ recommended; local run metadata is recorded in JSON"),
        ("Dependency file", status["dependency_file"]),
        ("Latest public tests", status["latest_test_result"]),
        ("WILD accounting layer", "Supported; core claim evidence"),
        ("MMLU-Pro source reconstruction", "Supported; provenance evidence"),
        ("Public proof-of-analysis package", "Ready" if status["github_ssrn_ready"] else "Not ready"),
        (
            "Methods/software article scope",
            "Ready for WILD-supported protocol claims" if status["methods_software_article_ready"] else "Not ready",
        ),
        (
            "MMLU-Pro parser validation",
            "Complete" if status["parser_validated"] else "Not complete; parser-dependent claims excluded",
        ),
        (
            "MMLU-Pro confirmatory evaluation",
            "Supported" if status["mmlu_pro_confirmatory"] else "Not claimed; diagnostic only",
        ),
        (
            "Full empirical ML article scope",
            "Ready" if status["full_empirical_ml_ready"] else "Not ready; not the target article type",
        ),
    ]
    lines = [r"\begin{tabular}{lp{0.62\linewidth}}", r"\toprule", r"Field & Value \\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{latex_escape(key)} & {latex_cell(value)} \\\\")
    if status["blockers"]:
        lines.append(r"\midrule")
        lines.append(f"Blockers & {latex_escape('; '.join(status['blockers']))} \\\\")
    if status.get("scope_notes"):
        lines.append(r"\midrule")
        lines.append(f"Scope note & {latex_escape(' '.join(status['scope_notes']))} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(
    *,
    root: Path,
    output_dir: Path,
    repository_url: str = "TODO",
    code_license: str = "MIT",
    data_output_license: str = "CC-BY-4.0 for derived aggregate outputs; source datasets remain under their original terms",
    dependency_file: str = "requirements.txt",
    expected_runtime: str = "analysis regeneration: minutes; WILD ingest depends on download/cache; CPU rerun optional",
    latest_test_command: str = "PYTHONPATH=main/src pytest -q main/tests",
    latest_test_result: str = "not recorded",
    parser_gate_path: Path = Path("main/analysis/parser_audit/parser_audit_claim_gate.json"),
    mmlu_gate_path: Path = Path("main/analysis/parser_audit/mmlu_claim_gate.json"),
    wild_claim_path: Path = Path("main/analysis/external_evidence/external_claim_check.json"),
    source_manifest_path: Path = Path("main/analysis/source_manifest/mmlu_pro_manifest_report.json"),
    hygiene_report_path: Path = Path("public_release/error_persistence_regression_burden/PUBLIC_RELEASE_HYGIENE.json"),
    public_manifest_path: Path = Path("public_release/error_persistence_regression_burden/PUBLIC_RELEASE_MANIFEST.json"),
    archive_identifier: str = "TODO",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = release_status(
        root=root,
        repository_url=repository_url,
        code_license=code_license,
        data_output_license=data_output_license,
        dependency_file=dependency_file,
        expected_runtime=expected_runtime,
        latest_test_command=latest_test_command,
        latest_test_result=latest_test_result,
        parser_gate_path=parser_gate_path,
        mmlu_gate_path=mmlu_gate_path,
        wild_claim_path=wild_claim_path,
        source_manifest_path=source_manifest_path,
        hygiene_report_path=hygiene_report_path,
        public_manifest_path=public_manifest_path,
        archive_identifier=archive_identifier,
    )
    write_json(output_dir / "artifact_release_status.json", status)
    write_tex(output_dir / "artifact_release_status.tex", status)
    return status


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate journal-release metadata and claim gates.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/artifact_release_status"))
    parser.add_argument("--repository-url", default="TODO")
    parser.add_argument("--code-license", default="MIT")
    parser.add_argument(
        "--data-output-license",
        default="CC-BY-4.0 for derived aggregate outputs; source datasets remain under their original terms",
    )
    parser.add_argument("--dependency-file", default="requirements.txt")
    parser.add_argument(
        "--expected-runtime",
        default="analysis regeneration: minutes; WILD ingest depends on download/cache; CPU rerun optional",
    )
    parser.add_argument("--latest-test-command", default="PYTHONPATH=main/src pytest -q main/tests")
    parser.add_argument("--latest-test-result", default="not recorded")
    parser.add_argument("--parser-gate-path", type=Path, default=Path("main/analysis/parser_audit/parser_audit_claim_gate.json"))
    parser.add_argument("--mmlu-gate-path", type=Path, default=Path("main/analysis/parser_audit/mmlu_claim_gate.json"))
    parser.add_argument("--wild-claim-path", type=Path, default=Path("main/analysis/external_evidence/external_claim_check.json"))
    parser.add_argument(
        "--source-manifest-path",
        type=Path,
        default=Path("main/analysis/source_manifest/mmlu_pro_manifest_report.json"),
    )
    parser.add_argument(
        "--hygiene-report-path",
        type=Path,
        default=Path("public_release/error_persistence_regression_burden/PUBLIC_RELEASE_HYGIENE.json"),
    )
    parser.add_argument(
        "--public-manifest-path",
        type=Path,
        default=Path("public_release/error_persistence_regression_burden/PUBLIC_RELEASE_MANIFEST.json"),
    )
    parser.add_argument(
        "--archive-identifier",
        default="TODO",
        help="Persistent archive identifier for journal submission, e.g. DOI, SWHID, OSF DOI, or repository archive URL.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    status = generate(
        root=args.root.resolve(),
        output_dir=args.output_dir,
        repository_url=args.repository_url,
        code_license=args.code_license,
        data_output_license=args.data_output_license,
        dependency_file=args.dependency_file,
        expected_runtime=args.expected_runtime,
        latest_test_command=args.latest_test_command,
        latest_test_result=args.latest_test_result,
        parser_gate_path=args.parser_gate_path,
        mmlu_gate_path=args.mmlu_gate_path,
        wild_claim_path=args.wild_claim_path,
        source_manifest_path=args.source_manifest_path,
        hygiene_report_path=args.hygiene_report_path,
        public_manifest_path=args.public_manifest_path,
        archive_identifier=args.archive_identifier,
    )
    print(
        json.dumps(
            {
                "github_ssrn_ready": status["github_ssrn_ready"],
                "core_claim_ready": status["core_claim_ready"],
                "methods_software_article_ready": status["methods_software_article_ready"],
                "full_empirical_ml_ready": status["full_empirical_ml_ready"],
                "blockers": status["blockers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
