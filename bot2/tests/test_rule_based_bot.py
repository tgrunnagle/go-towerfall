"""Unit tests for rule-based bot movement behaviors."""

import pytest

from bot.bots import BaseBot, RuleBasedBot, RuleBasedBotConfig
from bot.models import GAME_CONSTANTS, GameState, GameUpdate, PlayerState


class TestBaseBot:
    """Tests for the BaseBot abstract class."""

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a sample game state with multiple players."""
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
                    "player-3": {
                        "id": "player-3",
                        "objectType": "player",
                        "name": "Enemy2",
                        "x": 600.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 3.14,
                        "rad": 20.0,
                        "h": 50,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 2,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    def test_get_own_player(self, game_state: GameState) -> None:
        """Test BaseBot.get_own_player returns the correct player."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state)

        own_player = bot.get_own_player()
        assert own_player is not None
        assert own_player.id == "player-1"
        assert own_player.name == "Bot1"

    def test_get_own_player_returns_none_without_state(self) -> None:
        """Test get_own_player returns None when no state is set."""
        bot = RuleBasedBot("player-1")
        assert bot.get_own_player() is None

    def test_get_enemies(self, game_state: GameState) -> None:
        """Test BaseBot.get_enemies returns only alive enemies."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state)

        enemies = bot.get_enemies()
        assert len(enemies) == 2
        assert all(e.id != "player-1" for e in enemies)
        assert all(not e.dead for e in enemies)

    def test_get_enemies_excludes_dead(self, game_state: GameState) -> None:
        """Test get_enemies excludes dead players."""
        game_state.players["player-2"].dead = True

        bot = RuleBasedBot("player-1")
        bot.update_state(game_state)

        enemies = bot.get_enemies()
        assert len(enemies) == 1
        assert enemies[0].id == "player-3"

    def test_get_enemies_returns_empty_without_state(self) -> None:
        """Test get_enemies returns empty list when no state is set."""
        bot = RuleBasedBot("player-1")
        assert bot.get_enemies() == []


class TestRuleBasedBotFindNearestEnemy:
    """Tests for RuleBasedBot._find_nearest_enemy."""

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a game state with bot and multiple enemies at different distances."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 100.0,
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
                        "name": "NearEnemy",
                        "x": 200.0,
                        "y": 100.0,
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
                    "player-3": {
                        "id": "player-3",
                        "objectType": "player",
                        "name": "FarEnemy",
                        "x": 500.0,
                        "y": 100.0,
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

    def test_finds_nearest_enemy(self, game_state: GameState) -> None:
        """Bot should target closest enemy."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state)

        nearest = bot._find_nearest_enemy()
        assert nearest is not None
        assert nearest.id == "player-2"
        assert nearest.name == "NearEnemy"

    def test_finds_nearest_with_diagonal_distance(self) -> None:
        """Bot should find nearest using Euclidean distance."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 100.0,
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
                        "name": "DiagonalClose",
                        "x": 150.0,
                        "y": 150.0,
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
                    "player-3": {
                        "id": "player-3",
                        "objectType": "player",
                        "name": "HorizontalFar",
                        "x": 200.0,
                        "y": 100.0,
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # DiagonalClose is at distance sqrt(50^2 + 50^2) ≈ 70.7
        # HorizontalFar is at distance 100
        nearest = bot._find_nearest_enemy()
        assert nearest is not None
        assert nearest.id == "player-2"

    def test_returns_none_when_no_enemies(self) -> None:
        """Returns None when no enemies exist."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 100.0,
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        assert bot._find_nearest_enemy() is None


class TestRuleBasedBotHorizontalMovement:
    """Tests for RuleBasedBot horizontal movement decisions."""

    def test_moves_right_toward_enemy_on_right(self) -> None:
        """Bot should move right toward enemy on right."""
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        direction = bot._decide_horizontal_movement(300.0)
        assert direction == "d"

    def test_moves_left_toward_enemy_on_left(self) -> None:
        """Bot should move left toward enemy on left."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 400.0,
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
                        "name": "Enemy",
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        direction = bot._decide_horizontal_movement(100.0)
        assert direction == "a"

    def test_stops_in_dead_zone(self) -> None:
        """Bot should stop when within dead zone of target."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # Target is 15 pixels away, within default dead zone of 20
        direction = bot._decide_horizontal_movement(215.0)
        assert direction is None


class TestRuleBasedBotEdgeAvoidance:
    """Tests for RuleBasedBot edge avoidance behavior."""

    def test_avoids_left_edge(self) -> None:
        """Bot should not move left when near left edge."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 30.0,  # Within edge margin (default 50)
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # Trying to move left should be overridden to move right
        adjusted = bot._apply_edge_avoidance("a")
        assert adjusted == "d"

    def test_avoids_right_edge(self) -> None:
        """Bot should not move right when near right edge."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 770.0,  # Within edge margin of 800 (default 50)
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # Trying to move right should be overridden to move left
        adjusted = bot._apply_edge_avoidance("d")
        assert adjusted == "a"

    def test_allows_movement_away_from_edge(self) -> None:
        """Bot should allow moving away from edge."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 30.0,  # Near left edge
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # Moving right away from left edge is allowed
        adjusted = bot._apply_edge_avoidance("d")
        assert adjusted == "d"

    def test_no_edge_avoidance_in_center(self) -> None:
        """Bot should not apply edge avoidance when in center."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 400.0,  # Center of 800 pixel map
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # Both directions allowed in center
        assert bot._apply_edge_avoidance("a") == "a"
        assert bot._apply_edge_avoidance("d") == "d"


class TestRuleBasedBotJumping:
    """Tests for RuleBasedBot jumping behavior."""

    def test_jumps_when_enemy_above(self) -> None:
        """Bot should jump when enemy is above."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 400.0,  # Bot is lower (higher y)
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,  # Has jumps available
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        state = GameState.from_update(update)

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        # Create target that is above (lower y value)
        target = PlayerState.model_validate(
            {
                "id": "target",
                "objectType": "player",
                "name": "Target",
                "x": 200.0,
                "y": 300.0,  # Target is higher (lower y)
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

        assert bot._should_jump(target) is True

    def test_no_jump_when_no_jumps_available(self) -> None:
        """Bot should not jump when max jumps used."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 400.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 2,  # Max jumps used
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        state = GameState.from_update(update)

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        target = PlayerState.model_validate(
            {
                "id": "target",
                "objectType": "player",
                "name": "Target",
                "x": 200.0,
                "y": 300.0,  # Target is above
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

        assert bot._should_jump(target) is False

    def test_jumps_when_stuck(self) -> None:
        """Bot should jump when stuck (low velocity, far from target)."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 400.0,
                        "dx": 0.1,  # Very low velocity
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        target = PlayerState.model_validate(
            {
                "id": "target",
                "objectType": "player",
                "name": "Target",
                "x": 400.0,  # Far away (200 pixels)
                "y": 400.0,
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

        assert bot._should_jump(target) is True

    def test_no_jump_when_moving(self) -> None:
        """Bot should not jump when moving normally."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 400.0,
                        "dx": 10.0,  # Moving at decent speed
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        target = PlayerState.model_validate(
            {
                "id": "target",
                "objectType": "player",
                "name": "Target",
                "x": 400.0,
                "y": 400.0,  # Same height
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

        assert bot._should_jump(target) is False


class TestRuleBasedBotDecideActions:
    """Tests for RuleBasedBot.decide_actions main decision loop."""

    @pytest.mark.asyncio
    async def test_releases_keys_when_dead(self) -> None:
        """Bot should release all keys when dead."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 400.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 0,
                        "dead": True,  # Dead
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        state = GameState.from_update(update)

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        actions = await bot.decide_actions()
        actions_dict = dict(actions)

        assert actions_dict["w"] is False
        assert actions_dict["a"] is False
        assert actions_dict["d"] is False

    @pytest.mark.asyncio
    async def test_moves_to_center_when_no_enemies(self) -> None:
        """Bot should move to center when no enemies exist."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,  # Left of center
                        "y": 400.0,
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        actions = await bot.decide_actions()
        actions_dict = dict(actions)

        # Should move right toward center (400)
        assert actions_dict["d"] is True
        assert actions_dict["a"] is False

    @pytest.mark.asyncio
    async def test_moves_toward_enemy(self) -> None:
        """Bot should move toward nearest enemy."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 400.0,
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
                        "name": "Enemy",
                        "x": 600.0,  # Enemy on right
                        "y": 400.0,
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        actions = await bot.decide_actions()
        actions_dict = dict(actions)

        # Should move right toward enemy
        assert actions_dict["d"] is True
        assert actions_dict["a"] is False

    @pytest.mark.asyncio
    async def test_combined_movement_and_jump(self) -> None:
        """Bot should move and jump when enemy is above and to the side."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 500.0,  # Bot is lower
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,  # Has jumps
                        "ac": 4,
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy",
                        "x": 400.0,  # Enemy on right and above
                        "y": 300.0,
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

        bot = RuleBasedBot("player-1")
        bot.update_state(state)

        actions = await bot.decide_actions()
        actions_dict = dict(actions)

        # Should move right and jump
        assert actions_dict["d"] is True
        assert actions_dict["a"] is False
        assert actions_dict["w"] is True


class TestRuleBasedBotConfig:
    """Tests for RuleBasedBotConfig."""

    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        config = RuleBasedBotConfig()

        assert config.edge_margin == 50.0
        assert config.dead_zone == 20.0
        assert config.vertical_jump_threshold == 40.0
        assert config.stuck_velocity_threshold == 0.5
        assert config.stuck_distance_threshold == 50.0
        assert config.center_x == GAME_CONSTANTS.ROOM_SIZE_PIXELS_X / 2
        assert config.center_dead_zone == 50.0

    def test_custom_config_values(self) -> None:
        """Test custom configuration values."""
        config = RuleBasedBotConfig(
            edge_margin=100.0,
            dead_zone=30.0,
            vertical_jump_threshold=60.0,
        )

        assert config.edge_margin == 100.0
        assert config.dead_zone == 30.0
        assert config.vertical_jump_threshold == 60.0

    @pytest.mark.asyncio
    async def test_config_affects_behavior(self) -> None:
        """Test that config actually affects bot behavior."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 200.0,
                        "y": 400.0,
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

        # With small dead zone, bot should move toward center
        small_dead_zone_config = RuleBasedBotConfig(center_dead_zone=10.0)
        bot_small = RuleBasedBot("player-1", config=small_dead_zone_config)
        bot_small.update_state(state)

        actions = await bot_small.decide_actions()
        actions_dict = dict(actions)
        assert actions_dict["d"] is True  # Should move right toward center (400)

        # With large dead zone, bot at 200 is close enough to center
        large_dead_zone_config = RuleBasedBotConfig(center_dead_zone=250.0)
        bot_large = RuleBasedBot("player-1", config=large_dead_zone_config)
        bot_large.update_state(state)

        actions = await bot_large.decide_actions()
        actions_dict = dict(actions)
        assert actions_dict["d"] is False  # Should not move
        assert actions_dict["a"] is False


class TestRuleBasedBotImports:
    """Tests for correct module imports."""

    def test_bots_module_exports(self) -> None:
        """Test that bots module exports correct classes."""
        from bot.bots import (
            BaseBot,
            RuleBasedBot,
            RuleBasedBotConfig,
            RuleBasedBotRunner,
        )

        assert BaseBot is not None
        assert RuleBasedBot is not None
        assert RuleBasedBotConfig is not None
        assert RuleBasedBotRunner is not None

    def test_basebot_is_abstract(self) -> None:
        """Test that BaseBot cannot be instantiated directly."""
        from abc import ABC

        assert issubclass(BaseBot, ABC)

    def test_rule_based_bot_extends_base(self) -> None:
        """Test that RuleBasedBot extends BaseBot."""
        assert issubclass(RuleBasedBot, BaseBot)
