"""
Rules-based bot implementation with configurable difficulty levels.
This bot serves as the foundation for RL training and provides intelligent baseline behavior.
"""

import asyncio
import logging
import math
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from core.game_state import PlayerState, GameState, Block, ArrowState
from core.base_bot import BaseBot, Action, ActionPriority


class DifficultyLevel(Enum):
    """Bot difficulty levels with different capabilities"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ThreatType(Enum):
    """Types of threats the bot can detect"""

    PROJECTILE = "projectile"
    ENEMY = "enemy"
    BOUNDARY = "boundary"
    ENVIRONMENTAL = "environmental"


class OpportunityType(Enum):
    """Types of opportunities the bot can identify"""

    ATTACK = "attack"
    POWER_UP = "power_up"
    STRATEGIC_POSITION = "strategic_position"
    ESCAPE_ROUTE = "escape_route"


@dataclass
class Threat:
    """Represents a detected threat"""

    threat_type: ThreatType
    position: Tuple[float, float]
    velocity: Optional[Tuple[float, float]]
    severity: float  # 0.0 to 1.0
    time_to_impact: Optional[float]
    source_id: Optional[str]


@dataclass
class Opportunity:
    """Represents a detected opportunity"""

    opportunity_type: OpportunityType
    position: Tuple[float, float]
    value: float  # 0.0 to 1.0
    distance: float
    priority: float  # 0.0 to 1.0


@dataclass
class GameStateAnalysis:
    """Analysis results of the current game state"""

    threats: List[Threat]
    opportunities: List[Opportunity]
    player_position: Tuple[float, float]
    player_velocity: Tuple[float, float]
    player_health: float
    enemies: List[PlayerState]
    projectiles: List[ArrowState]
    # power_ups: List[Dict[str, Any]]
    # game_boundaries: Dict[str, float]
    # safe_zones: List[Tuple[float, float]]


class RulesBasedBot(BaseBot):
    """
    Rules-based bot with configurable difficulty levels and intelligent decision making.
    Serves as baseline for RL training and provides competitive gameplay.
    """

    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE):
        self.difficulty = difficulty
        self.logger = logging.getLogger(__name__)

        # Difficulty-based parameters
        self._configure_difficulty_parameters()

        # Game state tracking
        self.last_analysis = None
        self.action_history: list[Action | None] = []
        self.performance_metrics = {
            "wins": 0,
            "losses": 0,
            "kills": 0,
            "deaths": 0,
            "accuracy": 0.0,
        }

        # Decision making state
        self.current_target = None
        self.current_strategy = "balanced"
        self.last_action_time = 0
        self.reaction_delay_buffer = []

        # Adaptive behavior tracking
        self.game_history = []
        self.recent_performance = {
            "wins": 0,
            "losses": 0,
            "recent_games": 10,  # Number of recent games to consider for adaptation
        }
        self.adaptation_enabled = True
        self.base_config = self.config.copy()  # Store original config for adaptation

        # Initialize rule modules (will be imported when needed)
        self._combat_rules = None
        self._survival_rules = None
        self._strategic_rules = None

    def _configure_difficulty_parameters(self):
        """Configure bot parameters based on difficulty level"""
        difficulty_configs = {
            DifficultyLevel.BEGINNER: {
                "reaction_time": 0.3,  # 300ms delay
                "accuracy_modifier": 0.6,  # 60% accuracy
                "decision_frequency": 0.2,  # Update decisions every 200ms
                "aggression_level": 0.3,  # Conservative play
                "strategic_depth": 1,  # Simple strategies only
                "prediction_horizon": 0.5,  # Look ahead 0.5 seconds
                "risk_tolerance": 0.2,  # Very risk-averse
            },
            DifficultyLevel.INTERMEDIATE: {
                "reaction_time": 0.15,
                "accuracy_modifier": 0.75,
                "decision_frequency": 0.1,
                "aggression_level": 0.5,
                "strategic_depth": 2,
                "prediction_horizon": 1.0,
                "risk_tolerance": 0.4,
            },
            DifficultyLevel.ADVANCED: {
                "reaction_time": 0.08,
                "accuracy_modifier": 0.85,
                "decision_frequency": 0.05,
                "aggression_level": 0.7,
                "strategic_depth": 3,
                "prediction_horizon": 1.5,
                "risk_tolerance": 0.6,
            },
            DifficultyLevel.EXPERT: {
                "reaction_time": 0.03,
                "accuracy_modifier": 0.95,
                "decision_frequency": 0.02,
                "aggression_level": 0.8,
                "strategic_depth": 4,
                "prediction_horizon": 2.0,
                "risk_tolerance": 0.7,
            },
        }

        self.config = difficulty_configs[self.difficulty]

    def process_state_and_get_action(
        self, game_state: GameState
    ) -> tuple[Action | None, dict]:
        """
        Process the current game state and return the bot's action.

        Args:
            game_state: Raw game state data from the game client

        Returns:
            Action to be taken by the bot
        """
        if not game_state.player:
            self.logger.warning("Current player not found in game state")
            return None, {}

        analysis = self.analyze_game_state(game_state)
        action = self._select_action(analysis)
        self.action_history.append(action)
        return action, asdict(analysis)

    def analyze_game_state(self, game_state: GameState) -> GameStateAnalysis:
        """
        Analyze the current game state and extract relevant information.

        Args:
            game_state: Raw game state data from the game client

        Returns:
            GameStateAnalysis object containing processed information
        """

        # Extract basic player information
        player_info = self._extract_player_info(game_state)

        # Detect threats in the environment
        threats = self._detect_threats(game_state, player_info)

        # Find opportunities for advancement
        opportunities = self._find_opportunities(game_state, player_info)

        # Identify safe zones and escape routes
        # safe_zones = self._identify_safe_zones(game_state, threats)

        analysis = GameStateAnalysis(
            threats=threats,
            opportunities=opportunities,
            player_position=(player_info.x, player_info.y),
            player_velocity=(player_info.dx, player_info.dy),
            player_health=player_info.health,
            enemies=game_state.enemies.values(),
            projectiles=game_state.arrows.values(),
            # power_ups=game_state.get("powerUps", []),
            # game_boundaries=game_state.get("boundaries", {}),
            # safe_zones=safe_zones,
        )

        self.last_analysis = analysis
        return analysis

    def _detect_threats(
        self, game_state: GameState, player_info: PlayerState
    ) -> List[Threat]:
        """
        Detect and analyze threats in the current game state.

        Args:
            game_state: Current game state
            player_info: Information about the bot's player

        Returns:
            List of detected threats sorted by severity
        """
        threats = []
        player_pos = (player_info.x, player_info.y)

        # Detect projectile threats
        for projectile in game_state.arrows.values():
            threat = self._analyze_projectile_threat(projectile, player_pos)
            if threat:
                threats.append(threat)

        # Detect enemy threats
        for enemy in game_state.enemies.values():
            threat = self._analyze_enemy_threat(enemy, player_pos)
            if threat:
                threats.append(threat)

        # Sort threats by severity (highest first)
        threats.sort(key=lambda t: t.severity, reverse=True)

        return threats

    def _find_opportunities(
        self, game_state: GameState, player_info: PlayerState
    ) -> List[Opportunity]:
        """
        Identify opportunities for strategic advantage.

        Args:
            game_state: Current game state
            player_info: Information about the bot's player

        Returns:
            List of opportunities sorted by priority
        """
        opportunities = []
        player_pos = (player_info.x, player_info.y)

        # Find attack opportunities:
        for enemy in game_state.enemies.values():
            opportunity = self._analyze_attack_opportunity(enemy, player_pos)
            if opportunity:
                opportunities.append(opportunity)

        # Find strategic positioning opportunities
        strategic_positions = self._identify_strategic_positions(game_state, player_pos)
        for position in strategic_positions:
            opportunities.append(position)

        # Sort opportunities by priority (highest first)
        opportunities.sort(key=lambda o: o.priority, reverse=True)

        return opportunities

    def _select_action(self, analysis: GameStateAnalysis) -> Optional[Action]:
        """
        Select the best action based on game state analysis using rule priorities and decision trees.

        Args:
            analysis: Analyzed game state

        Returns:
            Selected action or None if no action needed
        """
        # Apply reaction time delay based on difficulty
        if not self._should_react_now():
            return None

        # Generate possible actions
        possible_actions = self._generate_possible_actions(analysis)

        # Filter actions based on current strategy and difficulty
        filtered_actions = self._filter_actions_by_strategy(possible_actions, analysis)

        # Apply decision tree logic
        selected_action = self._apply_decision_tree(filtered_actions, analysis)

        # Apply accuracy and confidence modifiers
        if selected_action:
            selected_action = self._apply_difficulty_modifiers(selected_action)

        # Record action for learning
        if selected_action:
            self.action_history.append(
                {
                    "action": selected_action,
                    "analysis": analysis,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

        return selected_action

    def get_difficulty_level(self) -> DifficultyLevel:
        """Get the current difficulty level"""
        return self.difficulty

    def set_difficulty_level(self, difficulty: DifficultyLevel):
        """Set a new difficulty level and reconfigure parameters"""
        self.difficulty = difficulty
        self._configure_difficulty_parameters()
        self.logger.info(f"Bot difficulty changed to {difficulty.value}")

    def update_performance_metrics(self, game_result: Dict[str, Any]):
        """Update performance metrics based on game results"""
        won = game_result.get("won", False)

        if won:
            self.performance_metrics["wins"] += 1
            self.recent_performance["wins"] += 1
        else:
            self.performance_metrics["losses"] += 1
            self.recent_performance["losses"] += 1

        self.performance_metrics["kills"] += game_result.get("kills", 0)
        self.performance_metrics["deaths"] += game_result.get("deaths", 0)

        # Update accuracy based on shots fired vs hits
        shots_fired = game_result.get("shots_fired", 0)
        shots_hit = game_result.get("shots_hit", 0)
        if shots_fired > 0:
            game_accuracy = shots_hit / shots_fired
            # Running average of accuracy
            total_games = (
                self.performance_metrics["wins"] + self.performance_metrics["losses"]
            )
            self.performance_metrics["accuracy"] = (
                self.performance_metrics["accuracy"] * (total_games - 1) + game_accuracy
            ) / total_games

        # Store game result for adaptive behavior
        self.game_history.append(
            {
                "won": won,
                "kills": game_result.get("kills", 0),
                "deaths": game_result.get("deaths", 0),
                "accuracy": game_accuracy if shots_fired > 0 else 0,
                "duration": game_result.get("duration", 0),
                "strategy_used": self.current_strategy,
                "difficulty_at_time": self.difficulty,
            }
        )

        # Keep only recent games for adaptation
        if len(self.game_history) > self.recent_performance["recent_games"]:
            self.game_history.pop(0)

        # Trigger adaptive behavior if enabled
        if self.adaptation_enabled:
            self._adapt_behavior()

        # Reset recent performance counter if we've reached the limit
        recent_total = (
            self.recent_performance["wins"] + self.recent_performance["losses"]
        )
        if recent_total >= self.recent_performance["recent_games"]:
            self.recent_performance["wins"] = 0
            self.recent_performance["losses"] = 0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_games = (
            self.performance_metrics["wins"] + self.performance_metrics["losses"]
        )
        win_rate = (
            self.performance_metrics["wins"] / total_games if total_games > 0 else 0
        )

        return {
            **self.performance_metrics,
            "total_games": total_games,
            "win_rate": win_rate,
            "kd_ratio": (
                self.performance_metrics["kills"]
                / max(1, self.performance_metrics["deaths"])
            ),
        }

    # Private helper methods

    def _extract_player_info(self, game_state: GameState) -> PlayerState:
        """Extract information about the bot's player from game state"""
        # TODO add bot action state here (e.g. currently holding mouse button down)
        return game_state.player

    def _analyze_projectile_threat(
        self, projectile: ArrowState, player_pos: Tuple[float, float]
    ) -> Optional[Threat]:
        """Analyze if a projectile poses a threat"""
        proj_pos = (projectile.x, projectile.y)
        proj_vel = (projectile.dx, projectile.dy)

        # Calculate distance and trajectory
        distance = math.sqrt(
            (proj_pos[0] - player_pos[0]) ** 2 + (proj_pos[1] - player_pos[1]) ** 2
        )

        # Simple trajectory prediction
        if proj_vel[0] != 0 or proj_vel[1] != 0:
            # Calculate if projectile is heading towards player
            direction_to_player = (
                player_pos[0] - proj_pos[0],
                player_pos[1] - proj_pos[1],
            )
            vel_magnitude = math.sqrt(proj_vel[0] ** 2 + proj_vel[1] ** 2)

            if vel_magnitude > 0:
                # Normalize velocity vector
                vel_normalized = (
                    proj_vel[0] / vel_magnitude,
                    proj_vel[1] / vel_magnitude,
                )

                # Check if projectile is moving towards player (dot product)
                dot_product = (
                    direction_to_player[0] * vel_normalized[0]
                    + direction_to_player[1] * vel_normalized[1]
                )

                if dot_product > 0 and distance < 200:  # Threat range
                    severity = max(0.1, 1.0 - (distance / 200))  # Closer = more severe
                    time_to_impact = (
                        distance / vel_magnitude if vel_magnitude > 0 else None
                    )

                    return Threat(
                        threat_type=ThreatType.PROJECTILE,
                        position=proj_pos,
                        velocity=proj_vel,
                        severity=severity,
                        time_to_impact=time_to_impact,
                        source_id=projectile.get("id"),
                    )

        return None

    def _analyze_enemy_threat(
        self, enemy: PlayerState, player_pos: Tuple[float, float]
    ) -> Optional[Threat]:
        """Analyze if an enemy poses a threat"""
        enemy_pos = (enemy.x, enemy.y)
        distance = math.sqrt(
            (enemy_pos[0] - player_pos[0]) ** 2 + (enemy_pos[1] - player_pos[1]) ** 2
        )

        # Enemies are more threatening when closer
        if distance < 300:  # Threat range
            severity = max(0.2, 1.0 - (distance / 300))

            # Increase severity if enemy is aiming at us or has line of sight
            # TODO calculate line of sight
            # if enemy.get("hasLineOfSight", False):
            #     severity *= 1.5

            return Threat(
                threat_type=ThreatType.ENEMY,
                position=enemy_pos,
                velocity=(enemy.dx, enemy.dy),
                severity=min(1.0, severity),
                time_to_impact=None,
                source_id=enemy.id,
            )

        return None

    def _analyze_attack_opportunity(
        self, enemy: PlayerState, player_pos: Tuple[float, float]
    ) -> Optional[Opportunity]:
        """Analyze attack opportunities against an enemy"""
        enemy_pos = (enemy.x, enemy.y)
        distance = math.sqrt(
            (enemy_pos[0] - player_pos[0]) ** 2 + (enemy_pos[1] - player_pos[1]) ** 2
        )

        # Good attack range
        if 50 < distance < 250:
            # Higher value for closer enemies (easier to hit)
            value = max(0.3, 1.0 - (distance / 250))

            # Increase value if enemy is vulnerable (low health, not moving much)
            if enemy.health < 50:
                value *= 1.3

            enemy_vel = (enemy.dx, enemy.dy)
            if (
                abs(enemy_vel[0]) + abs(enemy_vel[1]) < 10
            ):  # Enemy is relatively stationary
                value *= 1.2

            priority = value * self.config["aggression_level"]

            return Opportunity(
                opportunity_type=OpportunityType.ATTACK,
                position=enemy_pos,
                value=min(1.0, value),
                distance=distance,
                priority=min(1.0, priority),
            )

        return None

    def _identify_strategic_positions(
        self, game_state: Dict[str, Any], player_pos: Tuple[float, float]
    ) -> List[Opportunity]:
        """Identify strategic positions on the map"""
        # This is a simplified implementation - would need map knowledge for real strategic positioning
        strategic_positions = []

        # TODO: analyze block positions for strategic positions
        # High ground positions (if available in game state)
        # for platform in game_state.blocks.values():
        #     platform_pos = (platform.get("position", (0, 0)))
        #     distance = math.sqrt(
        #         (platform_pos[0] - player_pos[0]) ** 2
        #         + (platform_pos[1] - player_pos[1]) ** 2
        #     )

        #     if distance < 100 and platform.get("height", 0) > player_pos[1]:
        #         value = 0.6
        #         priority = (
        #             value * 0.5
        #         )  # Strategic positioning is lower priority than immediate threats

        #         strategic_positions.append(
        #             Opportunity(
        #                 opportunity_type=OpportunityType.STRATEGIC_POSITION,
        #                 position=platform_pos,
        #                 value=value,
        #                 distance=distance,
        #                 priority=priority,
        #             )
        #         )

        return strategic_positions

    def _identify_safe_zones(
        self, game_state: Dict[str, Any], threats: List[Threat]
    ) -> List[Tuple[float, float]]:
        """Identify safe zones away from threats"""
        safe_zones = []

        # TODO: analyze block positions for safe zones
        # Simple implementation: find areas away from threats
        # boundaries = game_state.get("boundaries", {})
        # if boundaries:
        #     # Create a grid of potential safe positions
        #     x_min = boundaries.get("left", -400)
        #     x_max = boundaries.get("right", 400)
        #     y_min = boundaries.get("top", -300)
        #     y_max = boundaries.get("bottom", 300)

        #     # Sample positions across the map
        #     for x in range(int(x_min), int(x_max), 50):
        #         for y in range(int(y_min), int(y_max), 50):
        #             position = (x, y)

        #             # Check if position is safe from all threats
        #             is_safe = True
        #             for threat in threats:
        #                 threat_distance = math.sqrt(
        #                     (position[0] - threat.position[0]) ** 2
        #                     + (position[1] - threat.position[1]) ** 2
        #                 )

        #                 # If too close to any threat, not safe
        #                 if threat_distance < 100:
        #                     is_safe = False
        #                     break

        #             if is_safe:
        #                 safe_zones.append(position)

        # return safe_zones[:10]  # Limit to top 10 safe zones
        return safe_zones

    def _should_react_now(self) -> bool:
        """Check if bot should react now based on reaction time and difficulty"""
        current_time = asyncio.get_event_loop().time()

        # Add reaction delay buffer
        if len(self.reaction_delay_buffer) == 0:
            # First decision, add delay
            delay = self.config["reaction_time"] + random.uniform(
                -0.02, 0.02
            )  # Small random variation
            self.reaction_delay_buffer.append(current_time + delay)
            return False

        # Check if enough time has passed
        if current_time >= self.reaction_delay_buffer[0]:
            self.reaction_delay_buffer.pop(0)
            return True

        return False

    def _generate_possible_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate all possible actions based on current analysis"""
        actions = []

        # Movement actions
        actions.extend(self._generate_movement_actions(analysis))

        # Combat actions
        actions.extend(self._generate_combat_actions(analysis))

        # Defensive actions
        actions.extend(self._generate_defensive_actions(analysis))

        # Strategic actions
        actions.extend(self._generate_strategic_actions(analysis))

        return actions

    def _generate_movement_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate movement-related actions"""
        actions = []

        # Basic movement directions
        directions = [
            ("move_left", {"key": "a", "pressed": True}, ActionPriority.MEDIUM),
            ("move_right", {"key": "d", "pressed": True}, ActionPriority.MEDIUM),
            ("jump", {"key": "w", "pressed": True}, ActionPriority.MEDIUM),
            ("crouch", {"key": "s", "pressed": True}, ActionPriority.LOW),
        ]

        for action_type, params, priority in directions:
            actions.append(
                Action(
                    action_type=action_type,
                    parameters=params,
                    priority=priority,
                    confidence=0.7,
                    expected_outcome=f"Move player {action_type.split('_')[1] if '_' in action_type else action_type}",
                    duration=0.1,
                )
            )

        return actions

    def _generate_combat_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate combat-related actions"""
        actions = []

        # Shooting actions for each enemy
        for enemy in analysis.enemies:
            enemy_pos = (enemy.x, enemy.y)
            distance = math.sqrt(
                (enemy_pos[0] - analysis.player_position[0]) ** 2
                + (enemy_pos[1] - analysis.player_position[1]) ** 2
            )

            if distance < 300:  # Within shooting range
                confidence = (
                    max(0.3, 1.0 - (distance / 300)) * self.config["accuracy_modifier"]
                )

                actions.append(
                    Action(
                        action_type="shoot_at_enemy",
                        parameters={
                            "button": "left",
                            "pressed": True,
                            "x": enemy_pos[0],
                            "y": enemy_pos[1],
                            "target_id": enemy.get("id"),
                        },
                        priority=ActionPriority.HIGH,
                        confidence=confidence,
                        expected_outcome=f"Attack enemy at {enemy_pos}",
                        duration=0.05,
                    )
                )

        return actions

    def _generate_defensive_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate defensive actions"""
        actions = []

        # Evasive maneuvers for each threat
        for threat in analysis.threats:
            if threat.threat_type == ThreatType.PROJECTILE:
                # Calculate evasion direction
                threat_to_player = (
                    analysis.player_position[0] - threat.position[0],
                    analysis.player_position[1] - threat.position[1],
                )

                # Move perpendicular to threat direction
                if abs(threat_to_player[0]) > abs(threat_to_player[1]):
                    # Move vertically
                    evasion_key = "w" if threat_to_player[1] > 0 else "s"
                else:
                    # Move horizontally
                    evasion_key = "d" if threat_to_player[0] > 0 else "a"

                actions.append(
                    Action(
                        action_type="evade_threat",
                        parameters={"key": evasion_key, "pressed": True},
                        priority=ActionPriority.CRITICAL,
                        confidence=0.8,
                        expected_outcome=f"Evade {threat.threat_type.value}",
                        duration=0.2,
                    )
                )

        return actions

    def _generate_strategic_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate strategic actions"""
        actions = []

        # Move towards opportunities
        for opportunity in analysis.opportunities[:3]:  # Top 3 opportunities
            direction_x = opportunity.position[0] - analysis.player_position[0]
            direction_y = opportunity.position[1] - analysis.player_position[1]

            # Determine primary movement direction
            if abs(direction_x) > abs(direction_y):
                key = "d" if direction_x > 0 else "a"
            else:
                key = "w" if direction_y > 0 else "s"

            actions.append(
                Action(
                    action_type="move_to_opportunity",
                    parameters={"key": key, "pressed": True},
                    priority=ActionPriority.LOW,
                    confidence=opportunity.value,
                    expected_outcome=f"Move towards {opportunity.opportunity_type.value}",
                    duration=0.15,
                )
            )

        return actions

    def _filter_actions_by_strategy(
        self, actions: List[Action], analysis: GameStateAnalysis
    ) -> List[Action]:
        """Filter actions based on current strategy and difficulty level"""
        filtered_actions = []

        for action in actions:
            # Apply strategy-based filtering
            if self.current_strategy == "aggressive":
                if action.action_type in ["shoot_at_enemy", "move_to_opportunity"]:
                    action.priority = ActionPriority(
                        min(1.0, action.priority.value * 1.2)
                    )
            elif self.current_strategy == "defensive":
                if action.action_type in ["evade_threat", "move_to_opportunity"]:
                    action.priority = ActionPriority(
                        min(1.0, action.priority.value * 1.2)
                    )

            # Apply difficulty-based filtering
            if self.difficulty == DifficultyLevel.BEGINNER:
                # Beginners make simpler decisions
                if action.action_type in ["move_left", "move_right", "shoot_at_enemy"]:
                    filtered_actions.append(action)
            else:
                # Higher difficulties consider all actions
                filtered_actions.append(action)

        return filtered_actions

    def _apply_decision_tree(
        self, actions: List[Action], analysis: GameStateAnalysis
    ) -> Optional[Action]:
        """Apply decision tree logic to select the best action"""
        if not actions:
            return None

        # Sort actions by priority and confidence
        actions.sort(key=lambda a: (a.priority.value, a.confidence), reverse=True)

        # Decision tree logic
        critical_actions = [a for a in actions if a.priority == ActionPriority.CRITICAL]
        if critical_actions:
            # Always prioritize critical actions (survival)
            return critical_actions[0]

        high_priority_actions = [
            a for a in actions if a.priority == ActionPriority.HIGH
        ]
        if high_priority_actions and len(analysis.threats) > 0:
            # Combat actions when threats are present
            return high_priority_actions[0]

        # For non-critical situations, consider confidence and strategy
        best_action = actions[0]

        # Apply some randomness based on difficulty (lower difficulty = more random)
        randomness_factor = 1.0 - (self.config["strategic_depth"] * 0.2)
        if random.random() < randomness_factor:
            # Sometimes pick a suboptimal action
            action_index = min(len(actions) - 1, random.randint(0, 2))
            best_action = actions[action_index]

        return best_action

    def _apply_difficulty_modifiers(self, action: Action) -> Action:
        """Apply difficulty-based modifiers to the selected action"""
        # Modify confidence based on accuracy
        action.confidence *= self.config["accuracy_modifier"]

        # Apply difficulty-specific modifications
        if action.action_type == "shoot_at_enemy":
            self._apply_shooting_difficulty_modifiers(action)
        elif action.action_type in ["move_left", "move_right", "jump", "evade_threat"]:
            self._apply_movement_difficulty_modifiers(action)

        # Apply reaction time delays for lower difficulties
        if self.difficulty in [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE]:
            # Add slight delay to action execution
            delay_factor = (
                1.0 if self.difficulty == DifficultyLevel.INTERMEDIATE else 2.0
            )
            additional_delay = random.uniform(0, 0.05 * delay_factor)
            if additional_delay > 0:
                self.reaction_delay_buffer.append(
                    asyncio.get_event_loop().time() + additional_delay
                )

        return action

    def _apply_shooting_difficulty_modifiers(self, action: Action):
        """Apply difficulty-specific modifiers to shooting actions"""
        if "x" not in action.parameters or "y" not in action.parameters:
            return

        # Calculate aim error based on difficulty
        base_error = (1.0 - self.config["accuracy_modifier"]) * 30

        if self.difficulty == DifficultyLevel.BEGINNER:
            # Large aim error, sometimes miss completely
            aim_error = base_error + random.uniform(0, 20)
            if random.random() < 0.1:  # 10% chance of complete miss
                aim_error *= 3
        elif self.difficulty == DifficultyLevel.INTERMEDIATE:
            # Moderate aim error with occasional precision
            aim_error = base_error * random.uniform(0.7, 1.3)
        elif self.difficulty == DifficultyLevel.ADVANCED:
            # Small aim error, mostly accurate
            aim_error = base_error * random.uniform(0.3, 0.8)
        else:  # EXPERT
            # Minimal aim error, very precise
            aim_error = base_error * random.uniform(0.1, 0.4)

        # Apply aim error
        action.parameters["x"] += random.uniform(-aim_error, aim_error)
        action.parameters["y"] += random.uniform(-aim_error, aim_error)

        # Adjust confidence based on target distance and difficulty
        if "target_id" in action.parameters and self.last_analysis:
            target_enemy = None
            for enemy in self.last_analysis.enemies:
                if enemy.get("id") == action.parameters["target_id"]:
                    target_enemy = enemy
                    break

            if target_enemy:
                target_pos = target_enemy.get("position", (0, 0))
                distance = math.sqrt(
                    (target_pos[0] - self.last_analysis.player_position[0]) ** 2
                    + (target_pos[1] - self.last_analysis.player_position[1]) ** 2
                )

                # Reduce confidence for distant targets based on difficulty
                distance_factor = min(
                    1.0, 200 / max(1, distance)
                )  # Optimal range is ~200 units
                difficulty_distance_penalty = {
                    DifficultyLevel.BEGINNER: 0.5,
                    DifficultyLevel.INTERMEDIATE: 0.7,
                    DifficultyLevel.ADVANCED: 0.85,
                    DifficultyLevel.EXPERT: 0.95,
                }

                action.confidence *= (
                    distance_factor * difficulty_distance_penalty[self.difficulty]
                )

    def _apply_movement_difficulty_modifiers(self, action: Action):
        """Apply difficulty-specific modifiers to movement actions"""
        # Lower difficulties have less precise movement timing
        if self.difficulty == DifficultyLevel.BEGINNER:
            # Sometimes hold keys too long or too short
            if action.duration:
                duration_error = random.uniform(-0.05, 0.1)  # Tend to hold longer
                action.duration = max(0.05, action.duration + duration_error)

            # Sometimes make suboptimal movement choices
            if random.random() < 0.15:  # 15% chance of suboptimal movement
                action.confidence *= 0.6

        elif self.difficulty == DifficultyLevel.INTERMEDIATE:
            # Slight timing variations
            if action.duration:
                duration_error = random.uniform(-0.02, 0.03)
                action.duration = max(0.05, action.duration + duration_error)

            # Occasional suboptimal choices
            if random.random() < 0.08:  # 8% chance
                action.confidence *= 0.8

    def _get_combat_rules(self):
        """Lazy initialization of combat rules"""
        if self._combat_rules is None:
            from .combat_rules import CombatRules

            self._combat_rules = CombatRules(self.config)
        return self._combat_rules

    def _get_survival_rules(self):
        """Lazy initialization of survival rules"""
        if self._survival_rules is None:
            from .survival_rules import SurvivalRules

            self._survival_rules = SurvivalRules(self.config)
        return self._survival_rules

    def _get_strategic_rules(self):
        """Lazy initialization of strategic rules"""
        if self._strategic_rules is None:
            from .strategic_rules import StrategicRules

            self._strategic_rules = StrategicRules(self.config)
        return self._strategic_rules

    def _adapt_behavior(self):
        """Adapt bot behavior based on recent game outcomes"""
        if (
            len(self.game_history) < 3
        ):  # Need at least 3 games for meaningful adaptation
            return

        recent_games = self.game_history[-5:]  # Look at last 5 games
        recent_win_rate = sum(1 for game in recent_games if game["won"]) / len(
            recent_games
        )
        recent_avg_accuracy = sum(game["accuracy"] for game in recent_games) / len(
            recent_games
        )
        recent_kd_ratio = sum(game["kills"] for game in recent_games) / max(
            1, sum(game["deaths"] for game in recent_games)
        )

        self.logger.info(
            f"Adapting behavior - Recent win rate: {recent_win_rate:.2f}, "
            f"Accuracy: {recent_avg_accuracy:.2f}, K/D: {recent_kd_ratio:.2f}"
        )

        # Adapt strategy based on performance
        if recent_win_rate < 0.3:  # Losing too much
            if self.current_strategy == "aggressive":
                self.current_strategy = "balanced"
                self.logger.info("Switching from aggressive to balanced strategy")
            elif self.current_strategy == "balanced":
                self.current_strategy = "defensive"
                self.logger.info("Switching from balanced to defensive strategy")

            # Also reduce aggression level temporarily
            self.config["aggression_level"] = max(
                0.2, self.config["aggression_level"] * 0.9
            )

        elif recent_win_rate > 0.7:  # Winning too much, can be more aggressive
            if self.current_strategy == "defensive":
                self.current_strategy = "balanced"
                self.logger.info("Switching from defensive to balanced strategy")
            elif self.current_strategy == "balanced":
                self.current_strategy = "aggressive"
                self.logger.info("Switching from balanced to aggressive strategy")

            # Increase aggression level
            self.config["aggression_level"] = min(
                1.0, self.config["aggression_level"] * 1.1
            )

        # Adapt accuracy and reaction time based on performance
        if recent_avg_accuracy < 0.3:  # Poor accuracy
            # Slow down to improve accuracy
            self.config["reaction_time"] = min(
                self.base_config["reaction_time"] * 1.5,
                self.config["reaction_time"] * 1.1,
            )
            self.config["decision_frequency"] = min(
                self.base_config["decision_frequency"] * 1.5,
                self.config["decision_frequency"] * 1.1,
            )
            self.logger.info("Slowing down reactions to improve accuracy")

        elif recent_avg_accuracy > 0.8:  # Very good accuracy
            # Can speed up
            self.config["reaction_time"] = max(
                self.base_config["reaction_time"] * 0.8,
                self.config["reaction_time"] * 0.95,
            )
            self.config["decision_frequency"] = max(
                self.base_config["decision_frequency"] * 0.8,
                self.config["decision_frequency"] * 0.95,
            )
            self.logger.info("Speeding up reactions due to good accuracy")

        # Adapt risk tolerance based on K/D ratio
        if recent_kd_ratio < 0.5:  # Dying too much
            self.config["risk_tolerance"] = max(
                0.1, self.config["risk_tolerance"] * 0.9
            )
            self.logger.info("Reducing risk tolerance due to high death rate")
        elif recent_kd_ratio > 2.0:  # Very good K/D
            self.config["risk_tolerance"] = min(
                1.0, self.config["risk_tolerance"] * 1.1
            )
            self.logger.info("Increasing risk tolerance due to good K/D ratio")

    def reset_adaptation(self):
        """Reset adaptive behavior to base configuration"""
        self.config = self.base_config.copy()
        self.current_strategy = "balanced"
        self.game_history.clear()
        self.recent_performance = {"wins": 0, "losses": 0, "recent_games": 10}
        self.logger.info("Reset adaptive behavior to base configuration")

    def set_adaptation_enabled(self, enabled: bool):
        """Enable or disable adaptive behavior"""
        self.adaptation_enabled = enabled
        self.logger.info(f"Adaptive behavior {'enabled' if enabled else 'disabled'}")

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and metrics"""
        if len(self.game_history) == 0:
            return {
                "adaptation_enabled": self.adaptation_enabled,
                "games_played": 0,
                "current_strategy": self.current_strategy,
                "config_changes": {},
            }

        recent_games = (
            self.game_history[-5:] if len(self.game_history) >= 5 else self.game_history
        )
        recent_win_rate = sum(1 for game in recent_games if game["won"]) / len(
            recent_games
        )

        config_changes = {}
        for key in [
            "aggression_level",
            "reaction_time",
            "risk_tolerance",
            "decision_frequency",
        ]:
            if abs(self.config[key] - self.base_config[key]) > 0.01:
                config_changes[key] = {
                    "base": self.base_config[key],
                    "current": self.config[key],
                    "change": self.config[key] - self.base_config[key],
                }

        return {
            "adaptation_enabled": self.adaptation_enabled,
            "games_played": len(self.game_history),
            "recent_win_rate": recent_win_rate,
            "current_strategy": self.current_strategy,
            "config_changes": config_changes,
        }
