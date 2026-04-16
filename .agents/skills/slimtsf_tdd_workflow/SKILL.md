---
description: Test-Driven Development (TDD) rules for modifying the SlimTSF codebase.
---

# SlimTSF TDD Workflow

When tasked with modifying the `slimtsf` source code (e.g. adding features, fixing bugs, or refactoring), you **MUST** strictly adhere to the following Test-Driven Development (TDD) workflow. 

## 1. Test First
Always write or modify the tests **before** touching the implementation. 
- All tests live in the `slimtsf/tests/` directory.
- Ensure the tests comprehensively cover the expected behavior, edge cases, configuration bounds, and failure modes.
- *Tip: If you're fixing a bug, write a test that reliably reproduces the bug first. It should fail initially.*

## 2. Make Changes
Once the tests are written:
- Make the minimum necessary changes to the source code (in `slimtsf/`) to satisfy the tests.
- Maintain the library's existing dependency hygiene. Do not import new external libraries without the user's explicit permission.

## 3. Re-Test
Verify that the tests now pass.
- Run `pytest slimtsf/tests/` (or the specific file `pytest slimtsf/tests/test_YOUR_FILE.py`).
- Make sure no pre-existing tests were broken.

## 4. Documentation Validation
- Update the `README.md` if any user-facing APIs were altered.
- Keep other LLM context files updated (e.g. `.agents/llm_context.md`) if architectural details shifted.
