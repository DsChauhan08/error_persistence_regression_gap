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
    }
    missing = [
        key
        for key in ["repository_url", "code_license", "data_output_license"]
        if not is_populated(metadata.get(key))
    ]
    github_public_ready = not missing and tests_ready and hygiene_ready and manifest_ready and core_claim_ready
    journal_ready = False
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
    if not parser_validated:
        blockers.append("parser_validated=false; non-blocking for GitHub/SSRN because MMLU-Pro raw-output claims are diagnostic only")
    if not mmlu_confirmatory:
        blockers.append("mmlu_pro_confirmatory=false; non-blocking for GitHub/SSRN because WILD carries the core claim")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "github_public_ready": github_public_ready,
        "core_claim_ready": core_claim_ready,
        "journal_ready": journal_ready,
        "parser_validated": parser_validated,
        "mmlu_pro_confirmatory": mmlu_confirmatory,
        "wild_claim_ready": wild_ready,
        "mmlu_pro_source_manifest_ready": source_ready,
        "public_hygiene_ready": hygiene_ready,
        "public_manifest_ready": manifest_ready,
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
        "paper_wording_rule": (
            "Use GitHub/SSRN public-release wording when github_public_ready=true. Do not present "
            "MMLU-Pro raw-output accounting as confirmatory until parser validation and robustness gates pass."
        ),
    }


def write_tex(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Repository URL", status["repository_url"]),
        ("Code license", status["code_license"]),
        ("Data-output license", status["data_output_license"]),
        ("Python / OS", f"{status['python_version']} / {status['os']}"),
        ("Dependency file", status["dependency_file"]),
        ("Expected runtime", status["expected_runtime"]),
        ("Latest test command", status["latest_test_command"]),
        ("Latest test result", status["latest_test_result"]),
        ("WILD item-level gate", str(status["wild_claim_ready"]).lower()),
        ("MMLU-Pro source manifest", str(status["mmlu_pro_source_manifest_ready"]).lower()),
        ("Public hygiene scan passed", str(status["public_hygiene_ready"]).lower()),
        ("Public manifest verified", str(status["public_manifest_ready"]).lower()),
        ("Core WILD-based claim supported", str(status["core_claim_ready"]).lower()),
        ("Parser validation complete", str(status["parser_validated"]).lower()),
        ("MMLU-Pro confirmatory", str(status["mmlu_pro_confirmatory"]).lower()),
        ("GitHub/SSRN package ready", str(status["github_public_ready"]).lower()),
    ]
    lines = [r"\begin{tabular}{lp{0.62\linewidth}}", r"\toprule", r"Field & Value \\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{latex_escape(key)} & {latex_escape(value)} \\\\")
    if status["blockers"]:
        lines.append(r"\midrule")
        lines.append(f"Blockers & {latex_escape('; '.join(status['blockers']))} \\\\")
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
    )
    print(
        json.dumps(
            {
                "github_public_ready": status["github_public_ready"],
                "core_claim_ready": status["core_claim_ready"],
                "journal_ready": status["journal_ready"],
                "blockers": status["blockers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
