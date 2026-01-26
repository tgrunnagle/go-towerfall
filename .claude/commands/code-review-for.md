Perform a comprehensive code review of the changes in the current feature branch, which aim to implement the following GitHub issue:
$ARGUMENTS

!git log main..HEAD --oneline --no-decorate
!git diff main...HEAD --stat
!git diff main...HEAD --name-only

## Review Process

### 1. Understand the Context
- Use the `gh` CLI to fetch the full issue details (title, description, acceptance criteria, labels)
- Review the git information above to understand the scope of changes
- Identify which project(s) are affected: `bot2/` (Python), `backend/` (Go), or both
- Review `docs/architecture/` markdown files for relevant architectural context if needed
- Understand the intended behavior and scope of the changes

### 2. Analyze the Implementation
Read all modified and new files to evaluate:

#### Correctness
- Does the implementation meet all acceptance criteria from the issue?
- Are there any bugs, logic errors, or edge cases not handled?
- Does the code produce the intended behavior?
- Are error cases properly handled?

#### Code Quality
- **Coding Standards**: Adherence to CLAUDE.md guidelines
  - `bot2/`: Python typing, Pydantic v2 patterns, module imports
  - `backend/`: Go conventions, error handling, idiomatic patterns
- **Anti-patterns**: Circular dependencies, god objects, tight coupling
- **Over-engineering**: Unnecessary abstractions, premature optimization, unused features
- **Simplicity**: Is the solution as simple as possible while meeting requirements?

#### Testing
- Are there appropriate tests (unit and/or integration)?
- Do tests cover critical paths, edge cases, and error conditions?
- Are tests well-organized and follow project conventions?
- `bot2/`: Tests in `tests/unit/` or `tests/integration/`, proper markers
- `backend/`: Table-driven tests, meaningful names, `t.Helper()` usage

#### Security
- Any potential vulnerabilities (injection, XSS, etc.)?
- Sensitive data handled properly?
- Authentication and authorization correct?

#### Architecture
- Does the implementation respect existing architecture?
- Are there breaking changes or backwards-compatibility issues?
- Is the code in the right location/module?
- Does it introduce unnecessary dependencies?

#### Documentation
- Are code comments present only where logic isn't self-evident?
- Are there unnecessary docstrings or comments on unchanged code?
- Is user-facing documentation updated if behavior changed?

### 3. Categorize Issues by Severity

**CRITICAL**: Must be fixed before merging
- Bugs that cause crashes, data loss, or security vulnerabilities
- Logic errors that produce incorrect results
- Breaks acceptance criteria or core functionality

**HIGH**: Should be fixed before merging
- Significant anti-patterns or architectural violations
- Missing critical error handling
- Missing tests for important functionality
- Significant deviations from coding standards

**MEDIUM**: Should be addressed
- Code clarity issues that make maintenance difficult
- Minor anti-patterns or style violations
- Missing edge case handling
- Test coverage gaps for non-critical paths

**LOW**: Nice to have
- Minor style inconsistencies
- Opportunities for simplification
- Documentation improvements

### 4. Generate Review Report

Write all identified issues to `CODE_REVIEW_ISSUES.md` in the root of the repository.

**IMPORTANT**:
- ONLY include issues that need to be addressed (CRITICAL, HIGH, MEDIUM)
- Omit LOW severity issues unless there are very few total issues
- Omit any positive feedback or praise
- Overwrite any existing content in the file
- Focus on actionable feedback with specific file locations and line numbers

#### Report Format

```markdown
# Code Review Issues

**Branch**: [branch-name]
**Issue**: #[issue-number] - [issue-title]
**Review Date**: [date]

## Summary
[Brief overview of changes reviewed and overall assessment]

---

## Critical Issues
[Issues that must be fixed before merging]

### [Issue Title]
- **Severity**: CRITICAL
- **Location**: [file.ext:line] or [file.ext:start-end]
- **Description**: [Detailed explanation of the problem]
- **Impact**: [What could go wrong]
- **Recommendation**: [Specific suggestion to fix]

---

## High Priority Issues
[Issues that should be fixed before merging]

### [Issue Title]
- **Severity**: HIGH
- **Location**: [file.ext:line]
- **Description**: [Detailed explanation]
- **Recommendation**: [Specific suggestion]

---

## Medium Priority Issues
[Issues that should be addressed]

### [Issue Title]
- **Severity**: MEDIUM
- **Location**: [file.ext:line]
- **Description**: [Detailed explanation]
- **Recommendation**: [Specific suggestion]

---

## Next Steps
1. Address all CRITICAL issues immediately
2. Fix HIGH priority issues before requesting merge
3. Consider MEDIUM priority issues based on timeline
4. Re-run tests and validation checks after fixes
```

### 5. Review Completion

After writing `CODE_REVIEW_ISSUES.md`:
- Provide a brief summary to the user
- Indicate total issue count by severity
- Suggest whether the branch is ready to merge or needs work
- Do not include positive feedback in your summary—focus only on issues