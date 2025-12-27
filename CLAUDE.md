# bot2/ Project Guidance (python)

## Coding practices

1. Use native python typing (e.g. prefer `list` to `typing.List`, `str | None` to `typing.Optional[str]`).
2. Python module imports should ALWAYS be relative to the `bot2` directory, e.g. `from bot2.models import GameState`.
3. Prefer using `task`s for typchecking, lint, and final validation (see `Taskfile.yml` for available tasks).
4. When not using a `task`, always use `uv` to execute python related operations, e.g. `uv run python`, `uv run pytest`, `uv add httpx`.
