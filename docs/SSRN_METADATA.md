# SSRN Submission Metadata

Use this file as the copy-paste source for the SSRN submission form.

## Title

An Item-Level Audit Protocol for Error Persistence and Regression Burden in Small Language Model Replacement

## Article Type / Paper Type Note

Software Tool Article / Methods and Software Artifact

## Author

Dhananjay Singh Chauhan

## Affiliation

Independent Reviewer

## Email

dschauhan08.me@gmail.com

## Date Written

June 20, 2026

## Abstract

Replacement decisions for small language models are often justified by aggregate benchmark gains. That practice can hide a deployment-relevant distinction: a candidate model may improve the mean score while newly failing on items that the current model answered correctly. This software-tool article presents a paired item-level replacement-audit protocol. For each shared labeled item, the protocol separates persistent correct answers, corrected errors, regressions, and persistent errors, then reports correction rate, regression mass, churn mass, error persistence, and normalized regression burden.

The contribution is a practical reporting protocol and public artifact, not a scaling law or a causal account of model families. The primary evidence is a parser-independent WILD correctness-only use case with 276,685 item-level records for five Qwen and Llama models up to approximately four billion parameters across ten tasks. In this public setting, Qwen2.5-0.5B to Qwen2.5-3B improves aggregate correctness by 22.2 percentage points while changing 40.9% of item outcomes; Llama 3.2 1B to 3B improves by 17.8 points while changing 35.3% of outcomes; and the near-parity Qwen2.5-1.5B to Qwen2.5-3B comparison improves by 5.7 points while changing 30.4% of outcomes. A secondary archived-output MMLU-Pro use case reconstructs 3,008 source items across 14 model-output runs and demonstrates raw-response audit tooling, but remains diagnostic because complete inference manifests and human parser validation are not yet available. The core claim is narrow: benchmark-based replacement decisions should report item-level corrections, regressions, persistent errors, and churn alongside aggregate accuracy.

## Keywords

small language models; language model evaluation; model replacement; item-level evaluation; prediction churn; regression burden; error persistence; WILD benchmark; MMLU-Pro; reproducible research; machine learning software artifact

## Suggested SSRN Classifications

Choose the closest available SSRN classifications. If SSRN offers more specific eJournals, prioritize:

- Computer Science
- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Software Engineering
- Evaluation and Benchmarking
- Reproducibility / Open Science, if available

## AI Disclosure Statement

AI-assisted editorial and software-development tools were used during manuscript drafting, code review, and artifact preparation. The author reviewed, edited, verified, and takes responsibility for the final manuscript, code, analyses, and claims.

## Funding Statement

No external funding supported this work.

## Competing Interests

The author declares no competing interests.

## Data and Software Availability

The public proof-of-analysis package is available at:

https://github.com/DsChauhan08/error_persistence_regression_gap

The package contains the manuscript, analysis code, derived MMLU-Pro pairwise metrics, source-row hashes, parser-audit metadata, scoring-mode sensitivity summaries, WILD correctness-only replication outputs, evidence-tier manifests, generated tables, generated figures, public hygiene checks, and manifest hashes. It does not redistribute full MMLU-Pro question text, full answer-option text, raw model responses, item-level response excerpts, private parser-audit samples, local cache paths, or credentials.

## Submission Note

This preprint is framed as a software-tool and methods-artifact article. The core empirical support is the public, parser-independent WILD item-level correctness use case. The MMLU-Pro material is retained as a lower-priority archived-output diagnostic because it documents substantial testing work and shows the raw-response audit workflow, but parser-dependent MMLU-Pro claims are not treated as confirmatory until manual parser labels and complete inference manifests exist.

