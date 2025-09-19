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
from dataclasses import dataclass


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
    enemies: List[Dict[str, Any]]
    projectiles: List[Dict[str, Any]]
    power_ups: List[Dict[str, Any]]
    game_boundaries: Dict[str, float]
    safe_zones: List[Tuple[float, float]]


class ActionPriority(Enum):
    """Priority levels for different actions"""
    CRITICAL = 1.0    # Immediate survival actions
    HIGH = 0.8        # Important tactical actions
    MEDIUM = 0.6      # Strategic actions
    LOW = 0.4         # Opportunistic actions
    MINIMAL = 0.2     # Background actions


@dataclass
class Action:
    """Represents a potential action the bot can take"""
    action_type: str
    parameters: Dict[str, Any]
    priority: ActionPriority
    confidence: float  # 0.0 to 1.0
    expected_outcome: str
    duration: Optional[float]  # How long to hold the action


class RulesBasedBot:
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
        self.current_game_state = None
        self.last_analysis = None
        self.action_history = []
        self.performance_metrics = {
            'wins': 0,
            'losses': 0,
            'kills': 0,
            'deaths': 0,
            'accuracy': 0.0
        }
        
        # Decision making state
        self.current_target = None
        self.current_strategy = "balanced"
        self.last_action_time = 0
        self.reaction_delay_buffer = []
        
    def _configure_difficulty_parameters(self):
        """Configure bot parameters based on difficulty level"""
        difficulty_configs = {
            DifficultyLevel.BEGINNER: {
                'reaction_time': 0.3,      # 300ms delay
                'accuracy_modifier': 0.6,   # 60% accuracy
                'decision_frequency': 0.2,  # Update decisions every 200ms
                'aggression_level': 0.3,    # Conservative play
                'strategic_depth': 1,       # Simple strategies only
                'prediction_horizon': 0.5,  # Look ahead 0.5 seconds
                'risk_tolerance': 0.2       # Very risk-averse
            },
            DifficultyLevel.INTERMEDIATE: {
                'reaction_time': 0.15,
                'accuracy_modifier': 0.75,
                'decision_frequency': 0.1,
                'aggression_level': 0.5,
                'strategic_depth': 2,
                'prediction_horizon': 1.0,
                'risk_tolerance': 0.4
            },
            DifficultyLevel.ADVANCED: {
                'reaction_time': 0.08,
                'accuracy_modifier': 0.85,
                'decision_frequency': 0.05,
                'aggression_level': 0.7,
                'strategic_depth': 3,
                'prediction_horizon': 1.5,
                'risk_tolerance': 0.6
            },
            DifficultyLevel.EXPERT: {
                'reaction_time': 0.03,
                'accuracy_modifier': 0.95,
                'decision_frequency': 0.02,
                'aggression_level': 0.8,
                'strategic_depth': 4,
                'prediction_horizon': 2.0,
                'risk_tolerance': 0.7
            }
        }
        
        self.config = difficulty_configs[self.difficulty]
        
    def analyze_game_state(self, game_state: Dict[str, Any]) -> GameStateAnalysis:
        """
        Analyze the current game state and extract relevant information.
        
        Args:
            game_state: Raw game state data from the game client
            
        Returns:
            GameStateAnalysis object containing processed information
        """
        self.current_game_state = game_state
        
        # Extract basic player information
        player_info = self._extract_player_info(game_state)
        
        # Detect threats in the environment
        threats = self.detect_threats(game_state, player_info)
        
        # Find opportunities for advancement
        opportunities = self.find_opportunities(game_state, player_info)
        
        # Identify safe zones and escape routes
        safe_zones = self._identify_safe_zones(game_state, threats)
        
        analysis = GameStateAnalysis(
            threats=threats,
            opportunities=opportunities,
            player_position=player_info.get('position', (0, 0)),
            player_velocity=player_info.get('velocity', (0, 0)),
            player_health=player_info.get('health', 100),
            enemies=game_state.get('enemies', []),
            projectiles=game_state.get('projectiles', []),
            power_ups=game_state.get('powerUps', []),
            game_boundaries=game_state.get('boundaries', {}),
            safe_zones=safe_zones
        )
        
        self.last_analysis = analysis
        return analysis
        
    def detect_threats(self, game_state: Dict[str, Any], player_info: Dict[str, Any]) -> List[Threat]:
        """
        Detect and analyze threats in the current game state.
        
        Args:
            game_state: Current game state
            player_info: Information about the bot's player
            
        Returns:
            List of detected threats sorted by severity
        """
        threats = []
        player_pos = player_info.get('position', (0, 0))
        
        # Detect projectile threats
        for projectile in game_state.get('projectiles', []):
            threat = self._analyze_projectile_threat(projectile, player_pos)
            if threat:
                threats.append(threat)
        
        # Detect enemy threats
        for enemy in game_state.get('enemies', []):
            threat = self._analyze_enemy_threat(enemy, player_pos)
            if threat:
                threats.append(threat)
        
        # Detect boundary threats (going out of bounds)
        boundary_threat = self._analyze_boundary_threat(player_pos, game_state.get('boundaries', {}))
        if boundary_threat:
            threats.append(boundary_threat)
        
        # Sort threats by severity (highest first)
        threats.sort(key=lambda t: t.severity, reverse=True)
        
        return threats
        
    def find_opportunities(self, game_state: Dict[str, Any], player_info: Dict[str, Any]) -> List[Opportunity]:
        """
        Identify opportunities for strategic advantage.
        
        Args:
            game_state: Current game state
            player_info: Information about the bot's player
            
        Returns:
            List of opportunities sorted by priority
        """
        opportunities = []
        player_pos = player_info.get('position', (0, 0))
        
        # Find attack opportunities
        for enemy in game_state.get('enemies', []):
            opportunity = self._analyze_attack_opportunity(enemy, player_pos)
            if opportunity:
                opportunities.append(opportunity)
        
        # Find power-up opportunities
        for power_up in game_state.get('powerUps', []):
            opportunity = self._analyze_power_up_opportunity(power_up, player_pos)
            if opportunity:
                opportunities.append(opportunity)
        
        # Find strategic positioning opportunities
        strategic_positions = self._identify_strategic_positions(game_state, player_pos)
        for position in strategic_positions:
            opportunities.append(position)
        
        # Sort opportunities by priority (highest first)
        opportunities.sort(key=lambda o: o.priority, reverse=True)
        
        return opportunities
        
    def select_action(self, analysis: GameStateAnalysis) -> Optional[Action]:
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
            self.action_history.append({
                'action': selected_action,
                'analysis': analysis,
                'timestamp': asyncio.get_event_loop().time()
            })
            
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
        if game_result.get('won'):
            self.performance_metrics['wins'] += 1
        else:
            self.performance_metrics['losses'] += 1
            
        self.performance_metrics['kills'] += game_result.get('kills', 0)
        self.performance_metrics['deaths'] += game_result.get('deaths', 0)
        
        # Update accuracy based on shots fired vs hits
        shots_fired = game_result.get('shots_fired', 0)
        shots_hit = game_result.get('shots_hit', 0)
        if shots_fired > 0:
            game_accuracy = shots_hit / shots_fired
            # Running average of accuracy
            total_games = self.performance_metrics['wins'] + self.performance_metrics['losses']
            self.performance_metrics['accuracy'] = (
                (self.performance_metrics['accuracy'] * (total_games - 1) + game_accuracy) / total_games
            )
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_games = self.performance_metrics['wins'] + self.performance_metrics['losses']
        win_rate = self.performance_metrics['wins'] / total_games if total_games > 0 else 0
        
        return {
            **self.performance_metrics,
            'total_games': total_games,
            'win_rate': win_rate,
            'kd_ratio': (self.performance_metrics['kills'] / 
                        max(1, self.performance_metrics['deaths']))
        }
        
    # Private helper methods
    
    def _extract_player_info(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract information about the bot's player from game state"""
        # This would need to be adapted based on actual game state structure
        player_info = {
            'position': game_state.get('player', {}).get('position', (0, 0)),
            'velocity': game_state.get('player', {}).get('velocity', (0, 0)),
            'health': game_state.get('player', {}).get('health', 100),
            'ammunition': game_state.get('player', {}).get('ammunition', 10),
            'power_ups': game_state.get('player', {}).get('powerUps', [])
        }
        return player_info
        
    def _analyze_projectile_threat(self, projectile: Dict[str, Any], player_pos: Tuple[float, float]) -> Optional[Threat]:
        """Analyze if a projectile poses a threat"""
        proj_pos = projectile.get('position', (0, 0))
        proj_vel = projectile.get('velocity', (0, 0))
        
        # Calculate distance and trajectory
        distance = math.sqrt((proj_pos[0] - player_pos[0])**2 + (proj_pos[1] - player_pos[1])**2)
        
        # Simple trajectory prediction
        if proj_vel[0] != 0 or proj_vel[1] != 0:
            # Calculate if projectile is heading towards player
            direction_to_player = (player_pos[0] - proj_pos[0], player_pos[1] - proj_pos[1])
            vel_magnitude = math.sqrt(proj_vel[0]**2 + proj_vel[1]**2)
            
            if vel_magnitude > 0:
                # Normalize velocity vector
                vel_normalized = (proj_vel[0] / vel_magnitude, proj_vel[1] / vel_magnitude)
                
                # Check if projectile is moving towards player (dot product)
                dot_product = (direction_to_player[0] * vel_normalized[0] + 
                              direction_to_player[1] * vel_normalized[1])
                
                if dot_product > 0 and distance < 200:  # Threat range
                    severity = max(0.1, 1.0 - (distance / 200))  # Closer = more severe
                    time_to_impact = distance / vel_magnitude if vel_magnitude > 0 else None
                    
                    return Threat(
                        threat_type=ThreatType.PROJECTILE,
                        position=proj_pos,
                        velocity=proj_vel,
                        severity=severity,
                        time_to_impact=time_to_impact,
                        source_id=projectile.get('id')
                    )
        
        return None
        
    def _analyze_enemy_threat(self, enemy: Dict[str, Any], player_pos: Tuple[float, float]) -> Optional[Threat]:
        """Analyze if an enemy poses a threat"""
        enemy_pos = enemy.get('position', (0, 0))
        distance = math.sqrt((enemy_pos[0] - player_pos[0])**2 + (enemy_pos[1] - player_pos[1])**2)
        
        # Enemies are more threatening when closer
        if distance < 300:  # Threat range
            severity = max(0.2, 1.0 - (distance / 300))
            
            # Increase severity if enemy is aiming at us or has line of sight
            if enemy.get('hasLineOfSight', False):
                severity *= 1.5
                
            return Threat(
                threat_type=ThreatType.ENEMY,
                position=enemy_pos,
                velocity=enemy.get('velocity', (0, 0)),
                severity=min(1.0, severity),
                time_to_impact=None,
                source_id=enemy.get('id')
            )
        
        return None
        
    def _analyze_boundary_threat(self, player_pos: Tuple[float, float], boundaries: Dict[str, float]) -> Optional[Threat]:
        """Analyze if player is approaching game boundaries"""
        if not boundaries:
            return None
            
        min_distance_to_boundary = float('inf')
        closest_boundary = None
        
        # Check distance to each boundary
        for boundary_name, boundary_value in boundaries.items():
            if boundary_name in ['left', 'right']:
                distance = abs(player_pos[0] - boundary_value)
            elif boundary_name in ['top', 'bottom']:
                distance = abs(player_pos[1] - boundary_value)
            else:
                continue
                
            if distance < min_distance_to_boundary:
                min_distance_to_boundary = distance
                closest_boundary = boundary_name
        
        # If too close to boundary, create threat
        if min_distance_to_boundary < 50:  # Danger zone
            severity = max(0.3, 1.0 - (min_distance_to_boundary / 50))
            
            return Threat(
                threat_type=ThreatType.BOUNDARY,
                position=player_pos,
                velocity=None,
                severity=severity,
                time_to_impact=None,
                source_id=closest_boundary
            )
        
        return None
        
    def _analyze_attack_opportunity(self, enemy: Dict[str, Any], player_pos: Tuple[float, float]) -> Optional[Opportunity]:
        """Analyze attack opportunities against an enemy"""
        enemy_pos = enemy.get('position', (0, 0))
        distance = math.sqrt((enemy_pos[0] - player_pos[0])**2 + (enemy_pos[1] - player_pos[1])**2)
        
        # Good attack range
        if 50 < distance < 250:
            # Higher value for closer enemies (easier to hit)
            value = max(0.3, 1.0 - (distance / 250))
            
            # Increase value if enemy is vulnerable (low health, not moving much)
            if enemy.get('health', 100) < 50:
                value *= 1.3
                
            enemy_vel = enemy.get('velocity', (0, 0))
            if abs(enemy_vel[0]) + abs(enemy_vel[1]) < 10:  # Enemy is relatively stationary
                value *= 1.2
                
            priority = value * self.config['aggression_level']
            
            return Opportunity(
                opportunity_type=OpportunityType.ATTACK,
                position=enemy_pos,
                value=min(1.0, value),
                distance=distance,
                priority=min(1.0, priority)
            )
        
        return None
        
    def _analyze_power_up_opportunity(self, power_up: Dict[str, Any], player_pos: Tuple[float, float]) -> Optional[Opportunity]:
        """Analyze power-up collection opportunities"""
        power_up_pos = power_up.get('position', (0, 0))
        distance = math.sqrt((power_up_pos[0] - player_pos[0])**2 + (power_up_pos[1] - player_pos[1])**2)
        
        # Power-ups are valuable when reasonably close
        if distance < 150:
            # Value decreases with distance
            value = max(0.4, 1.0 - (distance / 150))
            
            # Different power-up types have different values
            power_up_type = power_up.get('type', 'unknown')
            type_multipliers = {
                'health': 1.5,
                'ammunition': 1.2,
                'speed': 1.0,
                'shield': 1.3,
                'unknown': 0.8
            }
            
            value *= type_multipliers.get(power_up_type, 0.8)
            priority = value * 0.7  # Power-ups are generally medium priority
            
            return Opportunity(
                opportunity_type=OpportunityType.POWER_UP,
                position=power_up_pos,
                value=min(1.0, value),
                distance=distance,
                priority=min(1.0, priority)
            )
        
        return None
        
    def _identify_strategic_positions(self, game_state: Dict[str, Any], player_pos: Tuple[float, float]) -> List[Opportunity]:
        """Identify strategic positions on the map"""
        # This is a simplified implementation - would need map knowledge for real strategic positioning
        strategic_positions = []
        
        # High ground positions (if available in game state)
        platforms = game_state.get('platforms', [])
        for platform in platforms:
            platform_pos = platform.get('position', (0, 0))
            distance = math.sqrt((platform_pos[0] - player_pos[0])**2 + (platform_pos[1] - player_pos[1])**2)
            
            if distance < 100 and platform.get('height', 0) > player_pos[1]:
                value = 0.6
                priority = value * 0.5  # Strategic positioning is lower priority than immediate threats
                
                strategic_positions.append(Opportunity(
                    opportunity_type=OpportunityType.STRATEGIC_POSITION,
                    position=platform_pos,
                    value=value,
                    distance=distance,
                    priority=priority
                ))
        
        return strategic_positions
        
    def _identify_safe_zones(self, game_state: Dict[str, Any], threats: List[Threat]) -> List[Tuple[float, float]]:
        """Identify safe zones away from threats"""
        safe_zones = []
        
        # Simple implementation: find areas away from threats
        boundaries = game_state.get('boundaries', {})
        if boundaries:
            # Create a grid of potential safe positions
            x_min = boundaries.get('left', -400)
            x_max = boundaries.get('right', 400)
            y_min = boundaries.get('top', -300)
            y_max = boundaries.get('bottom', 300)
            
            # Sample positions across the map
            for x in range(int(x_min), int(x_max), 50):
                for y in range(int(y_min), int(y_max), 50):
                    position = (x, y)
                    
                    # Check if position is safe from all threats
                    is_safe = True
                    for threat in threats:
                        threat_distance = math.sqrt(
                            (position[0] - threat.position[0])**2 + 
                            (position[1] - threat.position[1])**2
                        )
                        
                        # If too close to any threat, not safe
                        if threat_distance < 100:
                            is_safe = False
                            break
                    
                    if is_safe:
                        safe_zones.append(position)
        
        return safe_zones[:10]  # Limit to top 10 safe zones
        
    def _should_react_now(self) -> bool:
        """Check if bot should react now based on reaction time and difficulty"""
        current_time = asyncio.get_event_loop().time()
        
        # Add reaction delay buffer
        if len(self.reaction_delay_buffer) == 0:
            # First decision, add delay
            delay = self.config['reaction_time'] + random.uniform(-0.02, 0.02)  # Small random variation
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
            ('move_left', {'key': 'a', 'pressed': True}, ActionPriority.MEDIUM),
            ('move_right', {'key': 'd', 'pressed': True}, ActionPriority.MEDIUM),
            ('jump', {'key': 'w', 'pressed': True}, ActionPriority.MEDIUM),
            ('crouch', {'key': 's', 'pressed': True}, ActionPriority.LOW)
        ]
        
        for action_type, params, priority in directions:
            actions.append(Action(
                action_type=action_type,
                parameters=params,
                priority=priority,
                confidence=0.7,
                expected_outcome=f"Move player {action_type.split('_')[1] if '_' in action_type else action_type}",
                duration=0.1
            ))
        
        return actions
        
    def _generate_combat_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate combat-related actions"""
        actions = []
        
        # Shooting actions for each enemy
        for enemy in analysis.enemies:
            enemy_pos = enemy.get('position', (0, 0))
            distance = math.sqrt(
                (enemy_pos[0] - analysis.player_position[0])**2 + 
                (enemy_pos[1] - analysis.player_position[1])**2
            )
            
            if distance < 300:  # Within shooting range
                confidence = max(0.3, 1.0 - (distance / 300)) * self.config['accuracy_modifier']
                
                actions.append(Action(
                    action_type='shoot_at_enemy',
                    parameters={
                        'button': 'left',
                        'pressed': True,
                        'x': enemy_pos[0],
                        'y': enemy_pos[1],
                        'target_id': enemy.get('id')
                    },
                    priority=ActionPriority.HIGH,
                    confidence=confidence,
                    expected_outcome=f"Attack enemy at {enemy_pos}",
                    duration=0.05
                ))
        
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
                    analysis.player_position[1] - threat.position[1]
                )
                
                # Move perpendicular to threat direction
                if abs(threat_to_player[0]) > abs(threat_to_player[1]):
                    # Move vertically
                    evasion_key = 'w' if threat_to_player[1] > 0 else 's'
                else:
                    # Move horizontally
                    evasion_key = 'd' if threat_to_player[0] > 0 else 'a'
                
                actions.append(Action(
                    action_type='evade_threat',
                    parameters={'key': evasion_key, 'pressed': True},
                    priority=ActionPriority.CRITICAL,
                    confidence=0.8,
                    expected_outcome=f"Evade {threat.threat_type.value}",
                    duration=0.2
                ))
        
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
                key = 'd' if direction_x > 0 else 'a'
            else:
                key = 'w' if direction_y > 0 else 's'
            
            actions.append(Action(
                action_type='move_to_opportunity',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.LOW,
                confidence=opportunity.value,
                expected_outcome=f"Move towards {opportunity.opportunity_type.value}",
                duration=0.15
            ))
        
        return actions
        
    def _filter_actions_by_strategy(self, actions: List[Action], analysis: GameStateAnalysis) -> List[Action]:
        """Filter actions based on current strategy and difficulty level"""
        filtered_actions = []
        
        for action in actions:
            # Apply strategy-based filtering
            if self.current_strategy == "aggressive":
                if action.action_type in ['shoot_at_enemy', 'move_to_opportunity']:
                    action.priority = ActionPriority(min(1.0, action.priority.value * 1.2))
            elif self.current_strategy == "defensive":
                if action.action_type in ['evade_threat', 'move_to_opportunity']:
                    action.priority = ActionPriority(min(1.0, action.priority.value * 1.2))
            
            # Apply difficulty-based filtering
            if self.difficulty == DifficultyLevel.BEGINNER:
                # Beginners make simpler decisions
                if action.action_type in ['move_left', 'move_right', 'shoot_at_enemy']:
                    filtered_actions.append(action)
            else:
                # Higher difficulties consider all actions
                filtered_actions.append(action)
        
        return filtered_actions
        
    def _apply_decision_tree(self, actions: List[Action], analysis: GameStateAnalysis) -> Optional[Action]:
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
        
        high_priority_actions = [a for a in actions if a.priority == ActionPriority.HIGH]
        if high_priority_actions and len(analysis.threats) > 0:
            # Combat actions when threats are present
            return high_priority_actions[0]
        
        # For non-critical situations, consider confidence and strategy
        best_action = actions[0]
        
        # Apply some randomness based on difficulty (lower difficulty = more random)
        randomness_factor = 1.0 - (self.config['strategic_depth'] * 0.2)
        if random.random() < randomness_factor:
            # Sometimes pick a suboptimal action
            action_index = min(len(actions) - 1, random.randint(0, 2))
            best_action = actions[action_index]
        
        return best_action
        
    def _apply_difficulty_modifiers(self, action: Action) -> Action:
        """Apply difficulty-based modifiers to the selected action"""
        # Modify confidence based on accuracy
        action.confidence *= self.config['accuracy_modifier']
        
        # Add some randomness to shooting accuracy for lower difficulties
        if action.action_type == 'shoot_at_enemy' and self.difficulty != DifficultyLevel.EXPERT:
            # Add aim error
            aim_error = (1.0 - self.config['accuracy_modifier']) * 20  # Up to 20 pixel error
            if 'x' in action.parameters and 'y' in action.parameters:
                action.parameters['x'] += random.uniform(-aim_error, aim_error)
                action.parameters['y'] += random.uniform(-aim_error, aim_error)
        
        return action