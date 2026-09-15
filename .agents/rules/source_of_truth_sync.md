# Agent Rule: Source of Truth Synchronization

## Purpose
This rule governs how AI coding assistants must maintain consistency across the `slimtsf-paper` workspace. It prevents hallucinated data, broken DOIs, desynchronized author metadata, or conflicting benchmark figures when a new AI agent session begins.

## Core Rules

1. **Check `SOURCE_OF_TRUTH.md` First:**  
   At the start of any new session or task, read `/Users/ken/Desktop/repository/slimtsf-paper/SOURCE_OF_TRUTH.md` to confirm the current version, DOI, author credentials, and active targets.

2. **The 4 Invariant Sets (Never Break Synchronization):**
   - **Version & Release Tag:** If version changes (e.g. `1.6.0` $\rightarrow$ `1.7.0`), update `slimtsf/CITATION.cff`, `slimtsf/README.md`, `softwarex-paper/manuscript.tex` (Table 1), `softwarex-paper/manuscript.md` (Table 1), and `SOURCE_OF_TRUTH.md`.
   - **Zenodo DOI:** If a new DOI is minted, update `slimtsf/README.md`, `slimtsf/CITATION.cff`, `softwarex-paper/manuscript.tex`, `softwarex-paper/manuscript.md`, and `SOURCE_OF_TRUTH.md`.
   - **Author & Affiliation:** Ensure `Nitipat Wuttisasiwat`, `Ken`, `0009-0003-0189-2653`, and `California State University, Fullerton` match across `CITATION.cff`, `manuscript.tex`, and `manuscript.md`.
   - **Empirical Findings:** All benchmark figures, runtimes, and accuracies must trace back to `slim-tsf/thesis_source_of_truth.md` or `slim-tsf/contexts/merged_results_context.md`. Do NOT invent synthetic benchmark numbers.

3. **Two-Repository Structure Awareness:**
   - `slimtsf/` is the **public library** (Git: `kennaruk/slimtsf`). Changes here affect PyPI and public consumers.
   - `slim-tsf/` is the **internal research workspace** (Git: `kennaruk/slim-tsf`). Houses the thesis, HPC logs, and the SoftwareX paper package (`softwarex-paper/`).

4. **Conventional Commits & Semantic Release:**
   - In `slimtsf/`, commit prefixes `feat:` trigger minor version bumps; `fix:` triggers patch bumps; `docs:`, `chore:`, `test:` trigger NO release. Commit mindfully.
