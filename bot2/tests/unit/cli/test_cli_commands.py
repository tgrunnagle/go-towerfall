"""Unit tests for CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from bot.cli.main import app
from bot.training.orchestrator_config import OrchestratorConfig

runner = CliRunner()


class TestMainApp:
    """Tests for the main CLI application."""

    def test_help_output(self) -> None:
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "TowerFall" in result.stdout or "towerfall" in result.stdout.lower()

    def test_no_args_shows_help(self) -> None:
        """Test that running without arguments shows help."""
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True returns exit code 0 or 2 depending on version
        # The important thing is that help is shown
        assert "train" in result.stdout.lower()
        assert "model" in result.stdout.lower()
        assert "config" in result.stdout.lower()


class TestTrainCommands:
    """Tests for train subcommands."""

    def test_train_help(self) -> None:
        """Test train --help works."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "start" in result.stdout
        assert "status" in result.stdout

    def test_train_start_help(self) -> None:
        """Test train start --help works."""
        result = runner.invoke(app, ["train", "start", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout
        assert "--timesteps" in result.stdout

    def test_train_list_empty(self) -> None:
        """Test train list with no runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("bot.cli.commands.train.DEFAULT_RUNS_DIR", tmpdir):
                result = runner.invoke(app, ["train", "list"])
                assert result.exit_code == 0
                assert "No training runs found" in result.stdout

    def test_train_status_no_runs(self) -> None:
        """Test train status with no runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("bot.cli.commands.train.DEFAULT_RUNS_DIR", tmpdir):
                result = runner.invoke(app, ["train", "status"])
                assert result.exit_code == 0
                assert "No training runs found" in result.stdout

    def test_train_status_unknown_run(self) -> None:
        """Test train status with unknown run ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("bot.cli.commands.train.DEFAULT_RUNS_DIR", tmpdir):
                result = runner.invoke(app, ["train", "status", "--run-id", "unknown"])
                assert result.exit_code == 1
                assert "not found" in result.stdout.lower()


class TestModelCommands:
    """Tests for model subcommands."""

    def test_model_help(self) -> None:
        """Test model --help works."""
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "evaluate" in result.stdout

    def test_model_list_empty(self) -> None:
        """Test model list with empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["model", "list", "--registry", tmpdir])
            assert result.exit_code == 0
            assert "No models" in result.stdout

    def test_model_show_not_found(self) -> None:
        """Test model show with unknown model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["model", "show", "ppo_gen_999", "--registry", tmpdir]
            )
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_help(self) -> None:
        """Test config --help works."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout
        assert "generate" in result.stdout

    def test_config_validate_valid(self) -> None:
        """Test validating a valid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            OrchestratorConfig().to_yaml(config_path)

            result = runner.invoke(app, ["config", "validate", str(config_path)])
            assert result.exit_code == 0
            assert "valid" in result.stdout.lower()

    def test_config_validate_invalid_yaml(self) -> None:
        """Test validating an invalid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("invalid: yaml: content:")

            result = runner.invoke(app, ["config", "validate", str(config_path)])
            assert result.exit_code == 1
            assert "error" in result.stdout.lower()

    def test_config_validate_empty(self) -> None:
        """Test validating an empty config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("")

            result = runner.invoke(app, ["config", "validate", str(config_path)])
            assert result.exit_code == 1
            assert "empty" in result.stdout.lower()

    def test_config_validate_with_warnings(self) -> None:
        """Test validating a config that generates warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            # Low timesteps should generate a warning
            config = OrchestratorConfig(total_timesteps=100)
            config.to_yaml(config_path)

            result = runner.invoke(app, ["config", "validate", str(config_path)])
            assert result.exit_code == 0
            assert "warning" in result.stdout.lower()

    def test_config_validate_verbose(self) -> None:
        """Test verbose validation output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            OrchestratorConfig().to_yaml(config_path)

            result = runner.invoke(
                app, ["config", "validate", str(config_path), "--verbose"]
            )
            assert result.exit_code == 0
            assert "PPO Hyperparameters" in result.stdout

    def test_config_generate(self) -> None:
        """Test generating a config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.yaml"

            result = runner.invoke(app, ["config", "generate", str(output_path)])
            assert result.exit_code == 0
            assert output_path.exists()

            # Verify it's valid
            config = OrchestratorConfig.from_yaml(output_path)
            assert config.total_timesteps > 0

    def test_config_generate_preset_quick(self) -> None:
        """Test generating a quick preset config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "quick.yaml"

            result = runner.invoke(
                app, ["config", "generate", str(output_path), "--preset", "quick"]
            )
            assert result.exit_code == 0

            config = OrchestratorConfig.from_yaml(output_path)
            assert config.total_timesteps == 50_000
            assert config.num_envs == 2

    def test_config_generate_preset_full(self) -> None:
        """Test generating a full preset config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full.yaml"

            result = runner.invoke(
                app, ["config", "generate", str(output_path), "--preset", "full"]
            )
            assert result.exit_code == 0

            config = OrchestratorConfig.from_yaml(output_path)
            assert config.total_timesteps == 5_000_000
            assert config.num_envs == 8

    def test_config_generate_no_overwrite(self) -> None:
        """Test that generate doesn't overwrite without --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "exists.yaml"
            output_path.write_text("existing content")

            result = runner.invoke(app, ["config", "generate", str(output_path)])
            assert result.exit_code == 1
            assert "already exists" in result.stdout.lower()

    def test_config_generate_force_overwrite(self) -> None:
        """Test that generate overwrites with --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "exists.yaml"
            output_path.write_text("existing content")

            result = runner.invoke(
                app, ["config", "generate", str(output_path), "--force"]
            )
            assert result.exit_code == 0
            assert output_path.exists()

    def test_config_show(self) -> None:
        """Test showing a config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            OrchestratorConfig().to_yaml(config_path)

            result = runner.invoke(app, ["config", "show", str(config_path)])
            assert result.exit_code == 0
            assert "num_envs" in result.stdout

    def test_config_show_json_format(self) -> None:
        """Test showing a config in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            OrchestratorConfig().to_yaml(config_path)

            result = runner.invoke(
                app, ["config", "show", str(config_path), "--format", "json"]
            )
            assert result.exit_code == 0
            # JSON format should have curly braces
            assert "{" in result.stdout

    def test_config_diff_identical(self) -> None:
        """Test diffing identical configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_a = Path(tmpdir) / "a.yaml"
            config_b = Path(tmpdir) / "b.yaml"
            OrchestratorConfig().to_yaml(config_a)
            OrchestratorConfig().to_yaml(config_b)

            result = runner.invoke(
                app, ["config", "diff", str(config_a), str(config_b)]
            )
            assert result.exit_code == 0
            assert "identical" in result.stdout.lower()

    def test_config_diff_different(self) -> None:
        """Test diffing different configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_a = Path(tmpdir) / "a.yaml"
            config_b = Path(tmpdir) / "b.yaml"
            OrchestratorConfig(num_envs=4).to_yaml(config_a)
            OrchestratorConfig(num_envs=8).to_yaml(config_b)

            result = runner.invoke(
                app, ["config", "diff", str(config_a), str(config_b)]
            )
            assert result.exit_code == 0
            assert "num_envs" in result.stdout
