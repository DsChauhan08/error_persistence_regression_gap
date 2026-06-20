PYTHON ?= python3
PYTHONPATH := src

.PHONY: test hygiene manifest reproduce-public

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q -p no:cacheprovider tests

hygiene:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m boundary_slm.public_release_hygiene --root .

manifest:
	$(PYTHON) -c 'from __future__ import annotations; import hashlib, json; from datetime import datetime, timezone; from pathlib import Path; root=Path("."); skip_dirs={".git","__pycache__",".pytest_cache",".venv"}; skip_names={"PUBLIC_RELEASE_MANIFEST.json"}; files=[]; [files.append({"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}) for p in sorted(root.rglob("*")) if p.is_file() and not any(part in skip_dirs for part in p.parts) and p.name not in skip_names]; manifest={"created_utc": datetime.now(timezone.utc).isoformat(), "root": ".", "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "missing": 0, "mismatches": 0, "public_text_policy": "Public release excludes raw model responses, full benchmark text, private parser-audit samples, local caches, DOI metadata, and private notes.", "files": files}; Path("PUBLIC_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8"); print(json.dumps({key: manifest[key] for key in ["file_count", "total_bytes", "missing", "mismatches"]}, indent=2))'

reproduce-public:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m boundary_slm.external_evidence_registry --output-dir analysis/external_evidence
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m boundary_slm.external_wild_ingest --output-dir analysis/external_evidence
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m boundary_slm.open_llm_details_ingest --output-dir analysis/external_evidence/open_llm_details
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m boundary_slm.artifact_release_status \
	  --root . \
	  --output-dir analysis/artifact_release_status \
	  --repository-url https://github.com/DsChauhan08/error_persistence_regression_gap \
	  --parser-gate-path analysis/parser_audit/parser_audit_claim_gate.json \
	  --mmlu-gate-path analysis/parser_audit/mmlu_claim_gate.json \
	  --wild-claim-path analysis/external_evidence/external_claim_check.json \
	  --source-manifest-path analysis/source_manifest/mmlu_pro_manifest_report.json \
	  --hygiene-report-path PUBLIC_RELEASE_HYGIENE.json \
	  --public-manifest-path PUBLIC_RELEASE_MANIFEST.json \
	  --archive-identifier 'https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/DsChauhan08/error_persistence_regression_gap.git'
