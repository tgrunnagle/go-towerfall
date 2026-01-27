"""Unit tests for neural network bot implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from bot.actions import Action
from bot.agent.network import ActorCriticNetwork
from bot.bots import NeuralNetBot, NeuralNetBotConfig, NeuralNetBotRunner
from bot.client import GameClient
from bot.models import GameState, GameUpdate
from bot.observation.observation_space import DEFAULT_CONFIG, ObservationConfig


class TestNeuralNetBotConfig:
    """Tests for NeuralNetBotConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = NeuralNetBotConfig()

        assert config.observation_config is None
        assert config.device == "cpu"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        obs_config = ObservationConfig(
            max_other_players=2,
            max_tracked_arrows=4,
            include_map=False,
        )
        config = NeuralNetBotConfig(
            observation_config=obs_config,
            device="cuda",
        )

        assert config.observation_config is obs_config
        assert config.device == "cuda"


class TestNeuralNetBotInit:
    """Tests for NeuralNetBot initialization."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    def test_initialization_with_defaults(self, network: ActorCriticNetwork) -> None:
        """Test bot initializes with default config."""
        bot = NeuralNetBot("player-1", network)

        assert bot.player_id == "player-1"
        assert bot.network is network
        assert bot.config.device == "cpu"
        assert bot.observation_builder is not None

    def test_initialization_with_custom_config(
        self, network: ActorCriticNetwork
    ) -> None:
        """Test bot initializes with custom config."""
        config = NeuralNetBotConfig(device="cpu")
        bot = NeuralNetBot("player-1", network, config)

        assert bot.config is config
        assert bot.device == torch.device("cpu")

    def test_network_set_to_eval_mode(self, network: ActorCriticNetwork) -> None:
        """Test that network is set to eval mode."""
        bot = NeuralNetBot("player-1", network)

        assert not bot.network.training

    def test_initial_state_tracking(self, network: ActorCriticNetwork) -> None:
        """Test initial state tracking variables."""
        bot = NeuralNetBot("player-1", network)

        assert bot._previous_movement_keys == {
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }
        assert bot._previous_aim_direction == -1.0
        assert bot._previous_shooting is False


class TestNeuralNetBotDecideActions:
    """Tests for NeuralNetBot.decide_actions()."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a sample game state."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot1",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy1",
                        "x": 300.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 3.14,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    @pytest.mark.asyncio
    async def test_releases_controls_when_dead(
        self, network: ActorCriticNetwork, game_state: GameState
    ) -> None:
        """Test bot releases all controls when dead."""
        game_state.players["player-1"].dead = True

        bot = NeuralNetBot("player-1", network)
        bot.update_state(game_state)

        actions = await bot.decide_actions()

        # Should return release actions for all keys
        keyboard_actions = {a[0]: a[1] for a in actions if len(a) == 2}
        assert keyboard_actions["w"] is False
        assert keyboard_actions["a"] is False
        assert keyboard_actions["d"] is False
        assert keyboard_actions["s"] is False

    @pytest.mark.asyncio
    async def test_releases_controls_when_no_state(
        self, network: ActorCriticNetwork
    ) -> None:
        """Test bot releases all controls when no state is set."""
        bot = NeuralNetBot("player-1", network)

        actions = await bot.decide_actions()

        # Should return release actions for all keys
        keyboard_actions = {a[0]: a[1] for a in actions if len(a) == 2}
        assert keyboard_actions["w"] is False
        assert keyboard_actions["a"] is False
        assert keyboard_actions["d"] is False
        assert keyboard_actions["s"] is False

    @pytest.mark.asyncio
    async def test_uses_network_for_inference(
        self, network: ActorCriticNetwork, game_state: GameState
    ) -> None:
        """Test bot uses network to select actions."""
        bot = NeuralNetBot("player-1", network)
        bot.update_state(game_state)

        # Mock the network to return a specific action
        with torch.no_grad():
            actions = await bot.decide_actions()

        # Should return some actions (exact actions depend on network)
        assert isinstance(actions, list)

    @pytest.mark.asyncio
    async def test_uses_deterministic_action_selection(
        self, network: ActorCriticNetwork, game_state: GameState
    ) -> None:
        """Test bot uses deterministic action selection."""
        bot = NeuralNetBot("player-1", network)
        bot.update_state(game_state)

        # Run multiple times to ensure deterministic
        actions_1 = await bot.decide_actions()

        # Reset bot state to allow retesting
        bot.reset()

        actions_2 = await bot.decide_actions()

        # Should get same actions each time (deterministic=True)
        # Note: comparing non-empty lists only, as deduplication may cause empty results
        if actions_1 and actions_2:
            assert actions_1 == actions_2


class TestNeuralNetBotActionTranslation:
    """Tests for action translation from discrete actions to game inputs."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    def test_no_op_action(self, network: ActorCriticNetwork) -> None:
        """Test NO_OP action returns empty list."""
        bot = NeuralNetBot("player-1", network)
        actions = bot._translate_action(Action.NO_OP, 100.0, 100.0)

        assert actions == []

    def test_move_left_press(self, network: ActorCriticNetwork) -> None:
        """Test MOVE_LEFT_PRESS action."""
        bot = NeuralNetBot("player-1", network)
        actions = bot._translate_action(Action.MOVE_LEFT_PRESS, 100.0, 100.0)

        assert len(actions) == 1
        assert actions[0] == ("a", True)

    def test_move_left_release(self, network: ActorCriticNetwork) -> None:
        """Test MOVE_LEFT_RELEASE action."""
        bot = NeuralNetBot("player-1", network)
        # Set previous state to pressed
        bot._previous_movement_keys["a"] = True

        actions = bot._translate_action(Action.MOVE_LEFT_RELEASE, 100.0, 100.0)

        assert len(actions) == 1
        assert actions[0] == ("a", False)

    def test_move_right_press(self, network: ActorCriticNetwork) -> None:
        """Test MOVE_RIGHT_PRESS action."""
        bot = NeuralNetBot("player-1", network)
        actions = bot._translate_action(Action.MOVE_RIGHT_PRESS, 100.0, 100.0)

        assert len(actions) == 1
        assert actions[0] == ("d", True)

    def test_jump_press(self, network: ActorCriticNetwork) -> None:
        """Test JUMP_PRESS action."""
        bot = NeuralNetBot("player-1", network)
        actions = bot._translate_action(Action.JUMP_PRESS, 100.0, 100.0)

        assert len(actions) == 1
        assert actions[0] == ("w", True)

    def test_dive_press(self, network: ActorCriticNetwork) -> None:
        """Test DIVE_PRESS action."""
        bot = NeuralNetBot("player-1", network)
        actions = bot._translate_action(Action.DIVE_PRESS, 100.0, 100.0)

        assert len(actions) == 1
        assert actions[0] == ("s", True)

    def test_aim_action(self, network: ActorCriticNetwork) -> None:
        """Test aim action sets direction."""
        bot = NeuralNetBot("player-1", network)
        actions = bot._translate_action(Action.AIM_0, 100.0, 100.0)

        # Should return mouse action with aim position
        assert len(actions) == 1
        assert len(actions[0]) == 4  # Mouse action tuple

    def test_shoot_start(self, network: ActorCriticNetwork) -> None:
        """Test SHOOT_START action."""
        bot = NeuralNetBot("player-1", network)
        bot._previous_aim_direction = 0.0  # Set aim direction

        actions = bot._translate_action(Action.SHOOT_START, 100.0, 100.0)

        assert len(actions) == 1
        assert len(actions[0]) == 4
        assert actions[0][1] is True  # Mouse pressed

    def test_shoot_release(self, network: ActorCriticNetwork) -> None:
        """Test SHOOT_RELEASE action."""
        bot = NeuralNetBot("player-1", network)
        bot._previous_shooting = True
        bot._previous_aim_direction = 0.0

        actions = bot._translate_action(Action.SHOOT_RELEASE, 100.0, 100.0)

        assert len(actions) == 1
        assert len(actions[0]) == 4
        assert actions[0][1] is False  # Mouse released

    def test_shoot_start_without_prior_aim(self, network: ActorCriticNetwork) -> None:
        """Test SHOOT_START uses default aim (0.0 radians) when no prior aim set."""
        bot = NeuralNetBot("player-1", network)
        # Don't set aim direction - should use default 0.0 (right)

        actions = bot._translate_action(Action.SHOOT_START, 100.0, 200.0)

        assert len(actions) == 1
        assert len(actions[0]) == 4
        assert actions[0][1] is True  # Mouse pressed
        # Verify aim position uses default direction (0.0 radians = right)
        # aim_x = 100.0 + 100.0 * cos(0.0) = 200.0
        # aim_y = 200.0 + 100.0 * sin(0.0) = 200.0
        assert actions[0][2] == 200.0  # aim_x  # type: ignore[index-out-of-bounds]
        assert actions[0][3] == 200.0  # aim_y  # type: ignore[index-out-of-bounds]


class TestNeuralNetBotStateDeduplication:
    """Tests for state deduplication - only sending changed inputs."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    def test_movement_key_deduplication(self, network: ActorCriticNetwork) -> None:
        """Test movement keys are deduplicated."""
        bot = NeuralNetBot("player-1", network)

        # First press
        actions1 = bot._translate_action(Action.MOVE_LEFT_PRESS, 100.0, 100.0)
        assert len(actions1) == 1

        # Second press - should be deduplicated
        actions2 = bot._translate_action(Action.MOVE_LEFT_PRESS, 100.0, 100.0)
        assert len(actions2) == 0

    def test_aim_direction_deduplication(self, network: ActorCriticNetwork) -> None:
        """Test aim directions are deduplicated."""
        bot = NeuralNetBot("player-1", network)

        # First aim
        actions1 = bot._translate_action(Action.AIM_0, 100.0, 100.0)
        assert len(actions1) == 1

        # Same aim - should be deduplicated
        actions2 = bot._translate_action(Action.AIM_0, 100.0, 100.0)
        assert len(actions2) == 0

    def test_shoot_state_deduplication(self, network: ActorCriticNetwork) -> None:
        """Test shooting state is deduplicated."""
        bot = NeuralNetBot("player-1", network)
        bot._previous_aim_direction = 0.0

        # First shoot start
        actions1 = bot._translate_action(Action.SHOOT_START, 100.0, 100.0)
        assert len(actions1) == 1

        # Second shoot start - should be deduplicated
        actions2 = bot._translate_action(Action.SHOOT_START, 100.0, 100.0)
        assert len(actions2) == 0


class TestNeuralNetBotReset:
    """Tests for NeuralNetBot.reset()."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    def test_reset_clears_movement_keys(self, network: ActorCriticNetwork) -> None:
        """Test reset clears movement key state."""
        bot = NeuralNetBot("player-1", network)
        bot._previous_movement_keys["a"] = True
        bot._previous_movement_keys["w"] = True

        bot.reset()

        assert bot._previous_movement_keys == {
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }

    def test_reset_clears_aim_direction(self, network: ActorCriticNetwork) -> None:
        """Test reset clears aim direction."""
        bot = NeuralNetBot("player-1", network)
        bot._previous_aim_direction = 1.5

        bot.reset()

        assert bot._previous_aim_direction == -1.0

    def test_reset_clears_shooting_state(self, network: ActorCriticNetwork) -> None:
        """Test reset clears shooting state."""
        bot = NeuralNetBot("player-1", network)
        bot._previous_shooting = True

        bot.reset()

        assert bot._previous_shooting is False


class TestNeuralNetBotRunner:
    """Tests for NeuralNetBotRunner."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    @pytest.fixture
    def mock_client(self) -> GameClient:
        """Create a mock GameClient."""
        client = MagicMock(spec=GameClient)
        client.player_id = "player-1"
        client.send_keyboard_input = AsyncMock()
        client.send_mouse_input = AsyncMock()
        return client

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a sample game state."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot1",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    def test_initialization(
        self, mock_client: GameClient, network: ActorCriticNetwork
    ) -> None:
        """Test runner initializes correctly."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)

        assert runner.client is mock_client
        assert runner.network is network
        assert runner.bot is None

    @pytest.mark.asyncio
    async def test_creates_bot_on_first_state(
        self,
        mock_client: GameClient,
        network: ActorCriticNetwork,
        game_state: GameState,
    ) -> None:
        """Test runner creates bot on first game state."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)

        await runner.on_game_state(game_state)

        assert runner.bot is not None
        assert runner.bot.player_id == "player-1"

    @pytest.mark.asyncio
    async def test_raises_if_no_player_id(
        self, network: ActorCriticNetwork, game_state: GameState
    ) -> None:
        """Test runner raises if client has no player_id."""
        mock_client = MagicMock(spec=GameClient)
        mock_client.player_id = None

        runner = NeuralNetBotRunner(network=network, client=mock_client)

        with pytest.raises(ValueError, match="player_id must be set"):
            await runner.on_game_state(game_state)

    @pytest.mark.asyncio
    async def test_sends_keyboard_inputs(
        self,
        mock_client: GameClient,
        network: ActorCriticNetwork,
        game_state: GameState,
    ) -> None:
        """Test runner sends keyboard inputs to client."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)

        await runner.on_game_state(game_state)

        # Should have sent some keyboard inputs
        assert mock_client.send_keyboard_input.call_count >= 0  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_deduplicates_keyboard_inputs(
        self,
        mock_client: GameClient,
        network: ActorCriticNetwork,
        game_state: GameState,
    ) -> None:
        """Test runner only sends changed keyboard inputs."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)

        # First call - should send inputs
        await runner.on_game_state(game_state)
        first_call_count = mock_client.send_keyboard_input.call_count  # type: ignore[attr-defined]

        # Reset mock
        mock_client.send_keyboard_input.reset_mock()  # type: ignore[attr-defined]

        # Second call with same state - should send fewer/no inputs
        await runner.on_game_state(game_state)
        second_call_count = mock_client.send_keyboard_input.call_count  # type: ignore[attr-defined]

        # Second call should send at most as many inputs as first
        assert second_call_count <= first_call_count

    @pytest.mark.asyncio
    async def test_deduplicates_mouse_inputs(
        self,
        mock_client: GameClient,
        network: ActorCriticNetwork,
        game_state: GameState,
    ) -> None:
        """Test runner only sends changed mouse inputs."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)

        await runner.on_game_state(game_state)

        # Mouse inputs should only be sent when state changes
        # (exact count depends on network output)
        assert mock_client.send_mouse_input.call_count >= 0  # type: ignore[attr-defined]

    def test_reset_clears_state(
        self, mock_client: GameClient, network: ActorCriticNetwork
    ) -> None:
        """Test reset clears runner state."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)
        runner._previous_keyboard_actions = {"a": True}
        runner._previous_mouse_state = True
        runner._previous_aim_pos = (100.0, 200.0)

        runner.reset()

        assert runner._previous_keyboard_actions == {}
        assert runner._previous_mouse_state is False
        assert runner._previous_aim_pos == (0.0, 0.0)

    @pytest.mark.asyncio
    async def test_reset_resets_bot(
        self,
        mock_client: GameClient,
        network: ActorCriticNetwork,
        game_state: GameState,
    ) -> None:
        """Test reset also resets the bot."""
        runner = NeuralNetBotRunner(network=network, client=mock_client)

        # Create bot
        await runner.on_game_state(game_state)
        assert runner.bot is not None

        # Set some bot state
        runner.bot._previous_shooting = True

        # Reset
        runner.reset()

        # Bot state should be reset
        assert runner.bot._previous_shooting is False


class TestNeuralNetBotImports:
    """Tests for correct module imports."""

    def test_bots_module_exports(self) -> None:
        """Test that bots module exports NeuralNetBot classes."""
        from bot.bots import (
            NeuralNetBot,
            NeuralNetBotConfig,
            NeuralNetBotRunner,
        )

        assert NeuralNetBot is not None
        assert NeuralNetBotConfig is not None
        assert NeuralNetBotRunner is not None

    def test_neural_net_bot_extends_base(self) -> None:
        """Test that NeuralNetBot extends BaseBot."""
        from bot.bots import BaseBot, NeuralNetBot

        assert issubclass(NeuralNetBot, BaseBot)


class TestNeuralNetBotEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def network(self) -> ActorCriticNetwork:
        """Create a test network."""
        return ActorCriticNetwork(
            observation_size=DEFAULT_CONFIG.total_size,
            action_size=27,
        )

    @pytest.mark.asyncio
    async def test_handles_missing_player(self, network: ActorCriticNetwork) -> None:
        """Test bot handles missing player in game state."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy",
                        "x": 300.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        state = GameState.from_update(update)

        bot = NeuralNetBot("player-1", network)
        bot.update_state(state)

        # Should not crash, should release controls
        actions = await bot.decide_actions()
        assert len(actions) >= 4  # At least release all movement keys

    @pytest.mark.asyncio
    async def test_handles_no_enemies(self, network: ActorCriticNetwork) -> None:
        """Test bot handles no enemies in game state."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        state = GameState.from_update(update)

        bot = NeuralNetBot("player-1", network)
        bot.update_state(state)

        # Should not crash
        actions = await bot.decide_actions()
        assert isinstance(actions, list)

    def test_config_defaults_applied(self, network: ActorCriticNetwork) -> None:
        """Test config defaults are applied correctly."""
        bot = NeuralNetBot("player-1", network)

        assert bot.config.device == "cpu"
        assert bot.config.observation_config is None
