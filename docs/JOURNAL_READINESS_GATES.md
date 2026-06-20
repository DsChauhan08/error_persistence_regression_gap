# Readiness Gates By Article Type

This project no longer uses one universal `journal_ready` flag for every possible paper. The release status separates the scoped methods/software article from a broader empirical model-evaluation article.

## Current Gate Logic

The scoped paper is a methods/software-tool article:

- `core_claim_ready=true`: WILD item-level correctness supports the replacement-accounting claim, and the MMLU-Pro source manifest verifies the archived item set.
- `github_ssrn_ready=true`: public tests pass, public hygiene passes, the manifest verifies, licenses are present, and the public proof-of-analysis package is complete.
- `methods_software_article_ready=true`: the public proof package is ready and has a persistent archive route.
- `mmlu_parser_validated=false`: MMLU-Pro parser validation is not complete and is not claimed.
- `mmlu_confirmatory_ready=false`: MMLU-Pro raw-output model evaluation is diagnostic only.
- `full_empirical_ml_ready=false`: a full empirical MMLU-Pro model-family paper is not the target article type.

The legacy `journal_ready` field is kept only as a backward-compatible alias for `full_empirical_ml_ready`. Do not use it as the readiness gate for the scoped methods/software article.

## Why Parser Validation Is Non-Blocking Here

Parser validation is required for confirmatory MMLU-Pro raw-output claims. It is not required for the WILD-supported core claim because WILD supplies item-level binary correctness records and does not depend on answer extraction from raw model completions.

The manuscript must therefore keep the claim split clear:

- WILD item-level replacement accounting: supported.
- MMLU-Pro source reconstruction: supported as provenance.
- MMLU-Pro raw-output accounting: diagnostic only.
- Parser-validation claims: not claimed.
- Full empirical MMLU-Pro family conclusions: not claimed.

## Path To Full Empirical Readiness

To make `mmlu_parser_validated=true` and `mmlu_confirmatory_ready=true`, complete the private/source parser audit:

```bash
PYTHONPATH=main/src python3 -m boundary_slm.parser_audit_labeler --limit 25
```

The labeler resumes automatically, prioritizes high-risk rows first, computes `human_parser_correct`, and saves after every row. When all rows are complete, regenerate the gates:

```bash
PYTHONPATH=main/src python3 -m boundary_slm.parser_audit_impact
PYTHONPATH=main/src python3 -m boundary_slm.mmlu_scoring_robustness
```

The full empirical gate should remain false unless the parser audit passes and the MMLU-Pro robustness gate does not materially change the claimed results.

## Archive Route Without Zenodo

Software Heritage can archive public Git repositories and provide Software Hash Identifiers (SWHIDs). The release-status gate uses the stable origin archive URL rather than a single request-specific identifier, because every public push creates a new commit that should be saved again.

For the current release status, use the stable origin archive URL:

```text
https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/DsChauhan08/error_persistence_regression_gap.git
```

After every public GitHub push, request a fresh Software Heritage save so the archive captures the latest commit.

## Sources Checked

- PLOS data availability policy: https://journals.plos.org/plosone/s/data-availability
- PLOS materials/software/code sharing: https://journals.plos.org/plosone/s/materials-software-and-code-sharing
- F1000Research software tool article guidance: https://f1000research.com/for-authors/article-guidelines/software-tool-articles
- F1000Research data/software/code guidance: https://f1000research.com/for-authors/data-guidelines
- PeerJ Computer Science author instructions: https://peerj.com/about/author-instructions/cs
- Software Heritage SWHID feature: https://www.softwareheritage.org/software-heritage-features/
- Software Heritage Save Code Now API documentation: https://docs.softwareheritage.org/devel/swh-web/uri-scheme-api.html
