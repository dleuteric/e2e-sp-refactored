# Contribution Guidelines

This repository follows the documentation-first approach introduced in the v1.2 release. When editing files under the repository root (unless a more specific `AGENTS.md` overrides these rules) please follow these conventions:

1. **Style and formatting**
   - Prefer simple python style for ease of reading for non-technical users.
   - Make sure the code is well commented.
   - Keep PlantUML diagrams human-readable with indentation and labels.

2. **Documentation expectations**
   - Every user-facing change must be reflected in `README.md` and, when relevant, under `docs/`.
   - When updating requirements, explain the rationale in `docs/requirements.md`.

3. **Testing**
   - When modifying pipeline code or orchestration scripts, run the appropriate pipeline step or
     provide reasoning if a full execution is not feasible.
   - Document executed commands in the PR body and final summary.

4. **Versioning**
   - Refer to the current release number (`v1.3`) in new documentation unless superseded.

