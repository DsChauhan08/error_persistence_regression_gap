# Journal Readiness Gates

This project separates a public GitHub/SSRN proof package from a stricter journal-ready package.

## Current Gate Logic

`journal_ready=true` requires all of the following:

1. `github_public_ready=true`: tests pass, hygiene scan passes, public manifest verifies, and WILD/source-manifest gates pass.
2. `parser_validated=true`: all 452 first-pass parser-audit rows are human-labeled, all 120 high-risk rows are labeled, and parser agreement is at least 95%.
3. `mmlu_pro_confirmatory=true`: MMLU-Pro scoring robustness passes after parser validation.
4. `archive_ready=true`: the public repository has a persistent archive identifier.

The gate is intentionally not satisfied by WILD alone. WILD supports the core parser-independent claim, but MMLU-Pro cannot become confirmatory until the human parser audit is complete.

## Fastest Path To `parser_validated=true`

Run short labeling sessions from the full private/source workspace, where the withheld parser-audit sample is available:

```bash
PYTHONPATH=main/src python3 -m boundary_slm.parser_audit_labeler --limit 25
```

The labeler resumes automatically, prioritizes high-risk rows first, computes `human_parser_correct`, and saves after every row. When all rows are complete, regenerate the gates:

```bash
PYTHONPATH=main/src python3 -m boundary_slm.parser_audit_impact
PYTHONPATH=main/src python3 -m boundary_slm.mmlu_scoring_robustness
```

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
