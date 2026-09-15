# 📌 Master Sources of Truth & Project Knowledge Base

> **CRITICAL DIRECTIVE FOR ALL AI AGENTS:**  
> Before answering questions, making changes, or writing code/manuscripts, **READ THIS DOCUMENT FIRST**.  
> If you ever update a version number, DOI, author detail, benchmark result, or architectural equation, you **MUST** update all linked source-of-truth files in lockstep. Never leave them desynchronized.

---

## 1. Project Mission & Identity

- **Project:** `slimtsf` (Sliding-Window Multivariate Time-Series Forest)
- **Primary Goal:** Publish a peer-reviewed, first-author journal paper in **Elsevier's *SoftwareX*** (SCIE-indexed, Clarivate Impact Factor ~3.0, Scopus Q2 in Computer Science).
- **Core Immigration / Legal Objective:** Anchor an **EB-2 NIW (National Interest Waiver)** green card petition with an uncontested first-author paper in an internationally recognized Q2 indexed journal.
- **Institutional Funding:** California State University, Fullerton (CSUF) has an active **Transformative Open Access Agreement with Elsevier** that covers **100% of the Article Publishing Charge (APC = $0)** under "Original software publications".

### Authorship Standard:
* **First Author / Corresponding Author:** **Nitipat (Ken) Wuttisasiwat**
  * *Legal Name (Must match passport for USCIS):* Nitipat Wuttisasiwat
  * *Preferred Name:* Ken
  * *ORCID:* `0009-0003-0189-2653`
  * *Affiliation:* Department of Computer Science, California State University, Fullerton, Fullerton, CA 92831, USA
  * *Email:* `nwuttisasiwat@csu.fullerton.edu`
* **Supervising Author / Faculty Mentor:** **Dr. Anli Ji**
  * *Role:* Master's Thesis Committee Chair / Senior Faculty Advisor
  * *Affiliation:* Department of Computer Science, California State University, Fullerton
  * *Email:* `aji@fullerton.edu`

---

## 2. The Canonical Sources of Truth (Hierarchy)

| Area | Canonical File Path | Role & Content |
| :--- | :--- | :--- |
| **Thesis Dissertation** | `slim-tsf/16 2026-04-14 Wuttisasiwat Thesis FINAL.docx` | **Absolute scientific source of truth.** The approved CSUF Master's thesis dissertation. |
| **Thesis Text Mirror** | `slim-tsf/thesis_source_of_truth.md` | Plain-text export of all 606 paragraphs and raw tables from the thesis DOCX for rapid search. |
| **HPC Benchmarks** | `slim-tsf/contexts/merged_results_context.md` | Raw Kubernetes cluster execution logs for 26 UEA datasets $\times$ 30 stratified resamples. |
| **Base Knowledge** | `slim-tsf/contexts/slimtsf-thesis-base-knowledge.md` | Core thesis pillars, research gap, HPC cluster directives, and paper arguments. |
| **LaTeX Paper** | `slim-tsf/softwarex-paper/manuscript.tex` | **Official submission source** for Overleaf. Formatted to Elsevier's `elsarticle` template. |
| **Markdown Paper** | `slim-tsf/softwarex-paper/manuscript.md` | Readable, editable mirror of the manuscript for review. |
| **Paper Checklist** | `slim-tsf/softwarex-paper/README.md` | Submission guide and compliance audit against SoftwareX Guide for Authors. |
| **Public Library** | `slimtsf/pyproject.toml` | Package versioning (`psr`), dependencies, build system, and PyPI metadata. |
| **Citation Specs** | `slimtsf/CITATION.cff` | Machine-readable citation file parsed by GitHub and Zenodo. |
| **Public README** | `slimtsf/README.md` | PyPI / GitHub user-facing documentation and Zenodo DOI badge. |

---

## 3. Registered Identifiers (Never Desynchronize)

* **Current Stable Release:** `v1.6.0`
* **Zenodo DOI:** `10.5281/zenodo.22761482`
* **Zenodo Record URL:** `https://doi.org/10.5281/zenodo.22761482`
* **Zenodo Badge SVG:** `https://zenodo.org/badge/1176562565.svg`
* **GitHub Repository (Public Library):** `https://github.com/kennaruk/slimtsf`
* **GitHub Repository (Research Workspace):** `https://github.com/kennaruk/slim-tsf`

---

## 4. Cross-File Synchronization Invariants

Whenever an agent updates any of the following items, the specified files **MUST** be updated together:

### Invariant A: Version Bump or Release Tag (e.g. `v1.6.0` $\rightarrow$ `v1.7.0`)
When a new version is tagged:
1. `slimtsf/CITATION.cff` $\rightarrow$ Update `version:` and `date-released:`.
2. `slimtsf/README.md` $\rightarrow$ Verify PyPI badge and release notes.
3. `slim-tsf/softwarex-paper/manuscript.tex` $\rightarrow$ Update Table 1, Row C1 (`\texttt{vX.X.X}`).
4. `slim-tsf/softwarex-paper/manuscript.md` $\rightarrow$ Update Table 1, Row C1.
5. `SOURCE_OF_TRUTH.md` $\rightarrow$ Update Section 3.

### Invariant B: Zenodo DOI Update (if a new DOI is minted)
When Zenodo produces a new DOI:
1. `slimtsf/README.md` $\rightarrow$ Update the Zenodo badge link and SVG URL.
2. `slimtsf/CITATION.cff` $\rightarrow$ Update `doi:` field.
3. `slim-tsf/softwarex-paper/manuscript.tex` $\rightarrow$ Update Table 1, Row C3 (`\url{https://doi.org/10.5281/zenodo.XXXXX}`).
4. `slim-tsf/softwarex-paper/manuscript.md` $\rightarrow$ Update Table 1, Row C3.
5. `SOURCE_OF_TRUTH.md` $\rightarrow$ Update Section 3.

### Invariant C: Author Details / Affiliations
If author names, affiliations, or ORCID are altered:
1. `slimtsf/CITATION.cff` $\rightarrow$ Update `authors:` array.
2. `slim-tsf/softwarex-paper/manuscript.tex` $\rightarrow$ Update `\author` and `\address` blocks.
3. `slim-tsf/softwarex-paper/manuscript.md` $\rightarrow$ Update author header.
4. `SOURCE_OF_TRUTH.md` $\rightarrow$ Update Section 1.

### Invariant D: Empirical Benchmark Results
If benchmark numbers, runtimes, or accuracy scores are revised:
1. `slim-tsf/softwarex-paper/manuscript.tex` $\rightarrow$ Update Table 2.
2. `slim-tsf/softwarex-paper/manuscript.md` $\rightarrow$ Update Table 2.
3. Verify against `slim-tsf/contexts/merged_results_context.md`.

---

## 5. Summary of What Has Been Done

- [x] Abstracted `slimtsf` into a modular, scikit-learn compatible library with automated CI/CD and Semantic Release on PyPI.
- [x] Connected GitHub to Zenodo and minted permanent DOI `10.5281/zenodo.22761482` for `v1.6.0`.
- [x] Added `CITATION.cff` with ORCID `0009-0003-0189-2653` and CSUF affiliation.
- [x] Created `examples/` directory in `slimtsf/` with runnable Python script and Jupyter notebook.
- [x] Formatted and wrote the full 6-page SoftwareX paper in both LaTeX (`manuscript.tex`) and Markdown (`manuscript.md`), aligned 100% with the official `softwarex-osp-template.tex`.
- [x] Embedded rich scientific findings from the Master's thesis (Permutation Collapse phenomenon, Wilcoxon Critical Difference diagrams, and runtime/accuracy tables across 5 UEA benchmarks).
- [x] Verified CSUF-Elsevier Transformative Open Access Agreement for 100% APC waiver ($0).
