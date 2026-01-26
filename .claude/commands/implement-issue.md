Implement the feature or fix described in the following GitHub issue:
$ARGUMENTS

## Implementation Process

### 1. Gather Context
- Use the `gh` CLI to fetch the full issue details (title, description, comments, labels)
- Review `docs/architecture/` markdown files for relevant architectural context
- Search the codebase for related code patterns, similar implementations, or affected areas
- Identify which project(s) are affected: `bot2/` (Python), `backend/` (Go), or both

### 2. Planning
- Create a todo list breaking down the implementation into concrete steps
- Identify all files that need to be created or modified
- Note any dependencies, prerequisites, or architectural considerations
- Consider edge cases, error handling, and testing requirements
- If the approach is unclear or multiple approaches exist, ask clarifying questions

### 3. Implementation
- Follow the coding practices and guidelines specified in CLAUDE.md
- For `bot2/`: Use Python typing, Pydantic v2 patterns, `uv` for operations
- For `backend/`: Follow Go conventions, proper error handling, idiomatic patterns
- Implement incrementally, marking todos as in_progress and completed
- Write clean, maintainable code without over-engineering
- Only add what's needed—avoid unnecessary abstractions, features, or refactoring

### 4. Testing
- Write appropriate tests (unit and/or integration as needed)
- For `bot2/`: Use pytest, organize in `tests/unit/` or `tests/integration/`
- For `backend/`: Write table-driven tests, use meaningful test names
- Run the test suite to verify the implementation
- Use task commands: `task bot2:test:unit`, `task be:test`, etc.

### 5. Validation
- Run linting and type checking
- For `bot2/`: `task bot2:check` (runs lint, format, typecheck, test)
- For `backend/`: `task be:check` (runs build, vet, test)
- Fix any issues identified by the checks
- Ensure all tests pass before proceeding

### 6. Documentation
- Update relevant documentation if the changes affect user-facing behavior or APIs
- Add code comments only where logic isn't self-evident
- Do not add unnecessary docstrings or comments to unchanged code

### 7. Completion
- Verify all acceptance criteria from the issue are met
- Ensure the implementation aligns with the issue description and any implementation guidance provided
- Run final checks to confirm everything works as expected

## Important Notes

- Read existing code before making changes—understand patterns and conventions
- Respect the existing architecture; don't introduce unnecessary complexity
- If the issue lacks clarity, ask questions before implementing
- Focus on solving the stated problem, not hypothetical future requirements
- Use the TodoWrite tool throughout to track progress and keep the user informed