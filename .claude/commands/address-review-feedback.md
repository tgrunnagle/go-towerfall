Systematically address the code review feedback documented in `CODE_REVIEW_ISSUES.md`.

!git status --short
!git branch --show-current

## Feedback Resolution Process

### 1. Review the Feedback
- Read the entire `CODE_REVIEW_ISSUES.md` file to understand all identified issues
- Note the branch name, issue number, and review date
- Understand the severity levels: CRITICAL, HIGH, MEDIUM
- Count total issues by severity to plan the work

### 2. Create a Todo List
Generate a comprehensive todo list using the TodoWrite tool:
- Start with CRITICAL issues (must be fixed)
- Then HIGH priority issues (should be fixed)
- Then MEDIUM priority issues (should be addressed)
- Each todo should reference the issue title and location from the review
- Include a final todo for validation (running tests and checks)

**Todo Format Examples**:
- "Fix CRITICAL: [Issue Title] at [location]"
- "Fix HIGH: [Issue Title] at [location]"
- "Fix MEDIUM: [Issue Title] at [location]"
- "Run validation checks and tests"

### 3. Address Issues Systematically
Work through the todo list in order:

#### Before Each Fix
- Read the relevant file(s) to understand the current implementation
- Review the issue description, impact, and recommendation
- If the recommendation is unclear or ambiguous, ask clarifying questions using AskUserQuestion
- Consider how the fix might affect other parts of the codebase

#### During Each Fix
- Follow the coding practices and guidelines specified in CLAUDE.md
- For `bot2/`: Use Python typing, Pydantic v2 patterns, `uv` for operations
- For `backend/`: Follow Go conventions, proper error handling, idiomatic patterns
- Implement the fix incrementally
- Mark the todo as in_progress before starting
- Mark the todo as completed immediately after finishing
- Keep fixes focused—only change what's necessary to address the issue

#### After Each Fix
- Verify the fix addresses the stated problem
- Ensure no new issues were introduced
- Update related tests if necessary

### 4. Handle Ambiguity
If any issue is unclear or has multiple valid solutions:
- Use the AskUserQuestion tool to clarify the expected approach
- Provide context about why clarification is needed
- Offer 2-3 options if applicable
- Wait for user response before proceeding

### 5. Validation
After addressing all issues:
- Run the appropriate test suite for affected projects
  - `bot2/`: `task bot2:test:unit` or `task bot2:test`
  - `backend/`: `task be:test`
- Run linting and type checking
  - `bot2/`: `task bot2:check`
  - `backend/`: `task be:check`
- Fix any new issues identified by the checks
- Ensure all tests pass

### 6. Update Review Document
After successfully addressing all issues:
- Create a summary of changes made
- Optionally append to `CODE_REVIEW_ISSUES.md` with a "## Resolution Summary" section
- List each issue addressed with a brief description of the fix

**Resolution Summary Format**:
```markdown
---

## Resolution Summary

**Resolved By**: Claude Code
**Resolution Date**: [date]

### Issues Addressed

#### CRITICAL Issues
- **[Issue Title]**: [Brief description of fix]

#### HIGH Priority Issues
- **[Issue Title]**: [Brief description of fix]

#### MEDIUM Priority Issues
- **[Issue Title]**: [Brief description of fix]

### Validation Results
- All tests passing: [Yes/No]
- Linting checks: [Pass/Fail]
- Type checking: [Pass/Fail]

### Additional Notes
[Any relevant notes about the fixes or decisions made]
```

### 7. Final Summary
Provide the user with:
- Total count of issues addressed by severity
- Confirmation that tests and checks pass
- Any issues that required user clarification
- Suggestion to review changes before merging

## Important Notes

- **Work systematically**: Address issues in order of severity (CRITICAL → HIGH → MEDIUM)
- **One issue at a time**: Complete each fix before moving to the next
- **Ask when unclear**: Don't guess if a recommendation is ambiguous
- **Test frequently**: Run tests after critical fixes to catch regressions early
- **Stay focused**: Fix only what's needed; don't refactor unrelated code
- **Track progress**: Use TodoWrite to keep the user informed of progress
- **Read before editing**: Always read files before making changes
- **Follow conventions**: Adhere to project-specific coding practices in CLAUDE.md