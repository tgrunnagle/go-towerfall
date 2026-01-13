# bot2/ Project Guidance (python)

## Coding practices

1. Use native python typing (e.g. prefer `list` to `typing.List`, `str | None` to `typing.Optional[str]`).
2. Python module imports should use the `bot` package name, e.g. `from bot.models import GameState`.
3. Use Pydantic v2 patterns: `model_config = ConfigDict(...)`, `Field(alias="...")`, and `model_validate()`.
4. When not using a `task`, always use `uv` to execute python operations, e.g. `uv run python`, `uv run pytest`, `uv add httpx`.

## Task commands

Prefer using `task` commands for linting, typechecking, and testing (see `Taskfile.yml`):

- `task bot2:install` - Install dependencies with uv
- `task bot2:lint` - Run ruff linting
- `task bot2:lint:fix` - Auto-fix lint issues
- `task bot2:format` - Format code with ruff
- `task bot2:typecheck` - Run type checking with ty
- `task bot2:test` - Run tests (excluding slow)
- `task bot2:test:unit` - Run unit tests only
- `task bot2:test:integration` - Run integration tests (requires server)
- `task bot2:test:all` - Run all tests including slow
- `task bot2:check` - Run all checks (lint, format, typecheck, test)

## Testing guidelines

1. Place unit tests in `tests/unit/`, integration tests in `tests/integration/`.
2. Use `pytest.mark.slow` for long-running tests.
3. Tests use `asyncio_mode = "auto"` - async test functions work automatically.
4. Use class-based test organization (e.g., `class TestPointModel:`).

---

# backend/ Project Guidance (Go)

## Coding practices

1. Follow standard Go conventions: use `gofmt` formatting, meaningful variable names, and idiomatic Go patterns.
2. Prefer returning errors over panicking; use `error` as the last return value.
3. Use table-driven tests for comprehensive test coverage.
4. Keep packages small and focused; avoid circular dependencies.
5. Use interfaces for abstraction, define them where they are used (not where implemented).
6. Prefer composition over inheritance; embed structs and interfaces appropriately.

## Task commands

Prefer using `task` commands for building, testing, and validation (see `Taskfile.yml`):

- `task be:build` - Build the backend application
- `task be:test` - Run all unit tests with verbose output
- `task be:test:short` - Run tests in short mode (faster)
- `task be:lint` - Run linting with golangci-lint
- `task be:format` - Format code with `go fmt`
- `task be:vet` - Run `go vet` for static analysis
- `task be:tidy` - Tidy Go module dependencies
- `task be:check` - Run all checks (build, vet, test)
- `task be:run` - Run the backend application

## Testing guidelines

1. Place tests in the same package as the code being tested (e.g., `foo_test.go` alongside `foo.go`).
2. Use `t.Helper()` in test helper functions for better error reporting.
3. Use `t.Parallel()` for tests that can run concurrently.
4. Name test functions descriptively: `TestFunctionName_Scenario_ExpectedBehavior`.

## Error handling

1. Wrap errors with context using `fmt.Errorf("context: %w", err)`.
2. Check errors immediately after the call that may return them.
3. Use sentinel errors (e.g., `var ErrNotFound = errors.New("not found")`) for expected error conditions.
