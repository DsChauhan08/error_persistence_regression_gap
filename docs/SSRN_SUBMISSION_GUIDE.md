# SSRN Submission Guide

This guide prepares the current paper for SSRN as a preprint. It is not a DOI/release-tag workflow. The public GitHub repository is the proof-of-analysis package for this stage.

Official SSRN submission guidance checked on June 20, 2026:

- SSRN requires a free SSRN account with a complete author profile.
- SSRN requires an English title, date written, English abstract or summary, author names, current affiliations, valid email addresses, and a full-text English PDF.
- The PDF must display the title and all authors with affiliations.
- If AI was used, SSRN requires an AI disclosure statement with the abstract and displayed on the PDF.
- SSRN says eligible content may still be rejected for missing information, research-integrity issues, policy issues, or terms-of-use non-compliance.

Source: https://www.elsevier.support/ssrn/answer/get-started

## Upload Files

Upload this PDF:

```text
paper/paper.pdf
```

Do not upload raw model-output JSONL files, private parser-audit samples, raw response excerpts, local cache files, or old working folders. The GitHub package already exposes public-safe derived outputs and reproducibility commands.

## Form Fields

Use `docs/SSRN_METADATA.md` as the copy-paste source for:

- title
- article type note
- author name
- affiliation
- email
- date written
- abstract
- keywords
- AI disclosure statement
- funding statement
- competing-interest statement
- data/software availability statement

## Affiliation

Use:

```text
Independent Researcher
```

Do not enter any city or country unless you personally want SSRN to display that location. The manuscript PDF now removes location fields from the author block.

## Article Positioning

Use this framing consistently:

```text
Software Tool Article / Methods and Software Artifact
```

Avoid presenting the paper as a broad empirical ML benchmark paper. The safest and strongest SSRN framing is:

- Core contribution: item-level replacement-audit protocol and public artifact.
- Primary use case: WILD correctness-only audit.
- Secondary use case: MMLU-Pro raw-output diagnostic.
- Not claimed: completed human parser validation for MMLU-Pro.
- Not claimed: confirmatory MMLU-Pro model-family conclusions.

## Suggested SSRN Categories

SSRN category names can change. Pick the closest available categories or eJournals in this order:

1. Computer Science
2. Artificial Intelligence
3. Machine Learning
4. Natural Language Processing
5. Software Engineering
6. Evaluation and Benchmarking
7. Reproducibility / Open Science, if available

## Submission Steps

1. Sign in or create an SSRN account at https://hq.ssrn.com/login/pubSignInJoin.cfm.
2. Complete the author profile with the name, email, and affiliation above.
3. Start a new paper submission.
4. Upload `paper/paper.pdf`.
5. Paste the title, abstract, keywords, and author metadata from `docs/SSRN_METADATA.md`.
6. Paste the AI disclosure statement when the form asks for it, and keep it in the PDF.
7. Paste the data/software availability statement and GitHub repository URL.
8. Choose the closest subject classifications.
9. Submit only after the PDF preview shows the title, author name, affiliation, email, article type, abstract, and AI disclosure.

## Pre-Submit Checklist

- [ ] PDF opens and shows the title correctly.
- [ ] Author block shows Dhananjay Singh Chauhan, `dschauhan08.me@gmail.com`, and Independent Researcher.
- [ ] No city or country appears in the PDF author block.
- [ ] Article type appears on the first page.
- [ ] AI disclosure statement appears near the abstract on the first page.
- [ ] Acknowledgments thank Kaggle and Google Colab for accessible compute used in the MMLU-Pro testing.
- [ ] MMLU-Pro is described as a diagnostic raw-output use case, not confirmatory evidence.
- [ ] GitHub URL is included in Data and Software Availability.
- [ ] `make test`, `make hygiene`, and `make manifest` pass in the public package.

## What SSRN Might Still Reject

SSRN does not guarantee posting. The most realistic rejection risks are missing metadata, missing AI disclosure, PDF metadata/display problems, or a reviewer treating the work as a guide/framework rather than a rigorous software-tool article. The current manuscript reduces that risk by explicitly labeling the article type, stating the method, reporting a public WILD use case, and keeping unsupported MMLU-Pro claims out of the core claim set.
