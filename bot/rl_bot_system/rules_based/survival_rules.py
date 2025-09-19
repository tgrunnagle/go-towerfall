"""
Survival rules for the rules-based bot.
Implements survival behaviors, health management, and safety protocols.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from .rules_based_bot import (
    Action, ActionPriority, GameStateAnalysis, 
    Threat, ThreatType, Opportunity, OpportunityType
)


class SurvivalRules:
    """
    Survival rules implementation for the rules-based bot.
    Handles health management, boundary avoidance, and safety protocols.
    """
    
    def __init__(self, bot_config: Dict[str, Any]):
        self.config = bot_config
        
    def evaluate_survival_situation(self, analysis: GameStateAnalysis) -> Dict[str, Any]:
        """
        Evaluate the current survival situation and return assessment.
        
        Args:
            analysis: Current game state analysis
            
        Returns:
            Dictionary containing survival situation assessment
        """
        assessment = {
            'health_status': self._assess_health_status(analysis.player_health),
            'boundary_safety': self._assess_boundary_safety(analysis.player_position, analysis.game_boundaries),
            'immediate_dangers': self._identify_immediate_dangers(analysis.threats),
            'escape_routes': self._identify_escape_routes(analysis),
            'survival_priority': self._calculate_survival_priority(analysis)
        }
        
        return assessment
        
    def generate_survival_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """
        Generate survival-specific actions based on game state analysis.
        
        Args:
            analysis: Current game state analysis
            
        Returns:
            List of survival actions
        """
        actions = []
        survival_assessment = self.evaluate_survival_situation(analysis)
        
        # Handle immediate dangers first
        if survival_assessment['immediate_dangers']:
            actions.extend(self._generate_danger_avoidance_actions(analysis, survival_assessment))
        
        # Handle boundary safety
        if not survival_assessment['boundary_safety']:
            actions.extend(self._generate_boundary_safety_actions(analysis))
        
        # Handle health management
        if survival_assessment['health_status'] == 'critical' or survival_assessment['health_status'] == 'low':
            actions.extend(self._generate_health_management_actions(analysis))
        
        # Generate escape actions if needed
        if survival_assessment['survival_priority'] > 0.7:
            actions.extend(self._generate_escape_actions(analysis, survival_assessment))
        
        return actions
        
    def _assess_health_status(self, health: float) -> str:
        """
        Assess player health status.
        
        Returns:
            Health status: 'critical', 'low', 'medium', 'high'
        """
        if health <= 25:
            return 'critical'
        elif health <= 50:
            return 'low'
        elif health <= 75:
            return 'medium'
        else:
            return 'high'
            
    def _assess_boundary_safety(self, player_pos: Tuple[float, float], boundaries: Dict[str, float]) -> bool:
        """
        Check if player is in a safe distance from boundaries.
        
        Returns:
            True if safe from boundaries, False if too close
        """
        if not boundaries:
            return True
            
        safety_margin = 60  # Minimum safe distance from boundary
        
        for boundary_name, boundary_value in boundaries.items():
            if boundary_name in ['left', 'right']:
                distance = abs(player_pos[0] - boundary_value)
            elif boundary_name in ['top', 'bottom']:
                distance = abs(player_pos[1] - boundary_value)
            else:
                continue
                
            if distance < safety_margin:
                return False
                
        return True
        
    def _identify_immediate_dangers(self, threats: List[Threat]) -> List[Threat]:
        """
        Identify threats that pose immediate danger requiring urgent action.
        
        Returns:
            List of immediate danger threats
        """
        immediate_dangers = []
        
        for threat in threats:
            # Projectiles with short time to impact
            if (threat.threat_type == ThreatType.PROJECTILE and 
                threat.time_to_impact and threat.time_to_impact < 0.5):
                immediate_dangers.append(threat)
                
            # Very close enemies
            elif (threat.threat_type == ThreatType.ENEMY and threat.severity > 0.8):
                immediate_dangers.append(threat)
                
            # Boundary threats with high severity
            elif (threat.threat_type == ThreatType.BOUNDARY and threat.severity > 0.7):
                immediate_dangers.append(threat)
                
        return immediate_dangers
        
    def _identify_escape_routes(self, analysis: GameStateAnalysis) -> List[Tuple[float, float]]:
        """
        Identify potential escape routes from current position.
        
        Returns:
            List of escape route positions
        """
        escape_routes = []
        player_pos = analysis.player_position
        
        # Use safe zones as escape routes
        escape_routes.extend(analysis.safe_zones)
        
        # Generate additional escape routes away from threats
        if analysis.threats:
            # Calculate average threat position
            threat_positions = [t.position for t in analysis.threats if t.threat_type != ThreatType.BOUNDARY]
            if threat_positions:
                avg_threat_x = sum(pos[0] for pos in threat_positions) / len(threat_positions)
                avg_threat_y = sum(pos[1] for pos in threat_positions) / len(threat_positions)
                
                # Generate positions away from average threat location
                escape_directions = [
                    (player_pos[0] - (avg_threat_x - player_pos[0]), player_pos[1]),  # Opposite X
                    (player_pos[0], player_pos[1] - (avg_threat_y - player_pos[1])),  # Opposite Y
                    (player_pos[0] - (avg_threat_x - player_pos[0]), 
                     player_pos[1] - (avg_threat_y - player_pos[1]))  # Diagonal opposite
                ]
                
                # Filter escape routes that are within boundaries
                for escape_pos in escape_directions:
                    if self._is_position_safe(escape_pos, analysis.game_boundaries):
                        escape_routes.append(escape_pos)
                        
        return escape_routes[:5]  # Limit to top 5 escape routes
        
    def _calculate_survival_priority(self, analysis: GameStateAnalysis) -> float:
        """
        Calculate overall survival priority from 0.0 to 1.0.
        Higher values indicate more urgent survival needs.
        """
        priority = 0.0
        
        # Health factor
        health_factor = 1.0 - (analysis.player_health / 100)
        priority += health_factor * 0.4
        
        # Threat factor
        if analysis.threats:
            max_threat_severity = max(t.severity for t in analysis.threats)
            priority += max_threat_severity * 0.4
        
        # Boundary factor
        if not self._assess_boundary_safety(analysis.player_position, analysis.game_boundaries):
            priority += 0.2
            
        return min(1.0, priority)
        
    def _generate_danger_avoidance_actions(self, analysis: GameStateAnalysis, survival_assessment: Dict[str, Any]) -> List[Action]:
        """Generate actions to avoid immediate dangers"""
        actions = []
        
        for danger in survival_assessment['immediate_dangers']:
            if danger.threat_type == ThreatType.PROJECTILE:
                # Quick dodge action
                actions.extend(self._generate_projectile_dodge_actions(danger, analysis))
            elif danger.threat_type == ThreatType.ENEMY:
                # Evasive movement away from enemy
                actions.extend(self._generate_enemy_evasion_actions(danger, analysis))
            elif danger.threat_type == ThreatType.BOUNDARY:
                # Move away from boundary
                actions.extend(self._generate_boundary_avoidance_actions(danger, analysis))
                
        return actions
        
    def _generate_projectile_dodge_actions(self, projectile_threat: Threat, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to dodge incoming projectiles"""
        actions = []
        
        if not projectile_threat.velocity:
            return actions
            
        # Calculate perpendicular movement to projectile path
        proj_vel = projectile_threat.velocity
        player_pos = analysis.player_position
        
        # Normalize projectile velocity
        vel_magnitude = math.sqrt(proj_vel[0]**2 + proj_vel[1]**2)
        if vel_magnitude == 0:
            return actions
            
        vel_normalized = (proj_vel[0] / vel_magnitude, proj_vel[1] / vel_magnitude)
        
        # Calculate perpendicular directions
        perp_directions = [
            (-vel_normalized[1], vel_normalized[0]),   # Perpendicular 1
            (vel_normalized[1], -vel_normalized[0])    # Perpendicular 2
        ]
        
        # Choose the perpendicular direction that keeps us in bounds
        best_direction = None
        for direction in perp_directions:
            new_pos = (player_pos[0] + direction[0] * 50, player_pos[1] + direction[1] * 50)
            if self._is_position_safe(new_pos, analysis.game_boundaries):
                best_direction = direction
                break
                
        if best_direction:
            # Determine key press based on direction
            if abs(best_direction[0]) > abs(best_direction[1]):
                key = 'd' if best_direction[0] > 0 else 'a'
            else:
                key = 's' if best_direction[1] > 0 else 'w'
                
            actions.append(Action(
                action_type='dodge_projectile',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.CRITICAL,
                confidence=0.9,
                expected_outcome="Dodge incoming projectile",
                duration=0.2
            ))
            
        return actions
        
    def _generate_enemy_evasion_actions(self, enemy_threat: Threat, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to evade close enemies"""
        actions = []
        
        player_pos = analysis.player_position
        enemy_pos = enemy_threat.position
        
        # Move away from enemy
        direction_x = player_pos[0] - enemy_pos[0]
        direction_y = player_pos[1] - enemy_pos[1]
        
        # Normalize direction
        distance = math.sqrt(direction_x**2 + direction_y**2)
        if distance > 0:
            direction_x /= distance
            direction_y /= distance
            
            # Choose primary movement direction
            if abs(direction_x) > abs(direction_y):
                key = 'd' if direction_x > 0 else 'a'
            else:
                key = 's' if direction_y > 0 else 'w'
                
            actions.append(Action(
                action_type='evade_enemy',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.CRITICAL,
                confidence=0.85,
                expected_outcome="Evade close enemy",
                duration=0.25
            ))
            
        return actions
        
    def _generate_boundary_avoidance_actions(self, boundary_threat: Threat, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to avoid going out of bounds"""
        actions = []
        
        player_pos = analysis.player_position
        boundaries = analysis.game_boundaries
        
        if not boundaries:
            return actions
            
        # Find which boundary is closest
        closest_boundary = None
        min_distance = float('inf')
        
        for boundary_name, boundary_value in boundaries.items():
            if boundary_name in ['left', 'right']:
                distance = abs(player_pos[0] - boundary_value)
                if distance < min_distance:
                    min_distance = distance
                    closest_boundary = boundary_name
            elif boundary_name in ['top', 'bottom']:
                distance = abs(player_pos[1] - boundary_value)
                if distance < min_distance:
                    min_distance = distance
                    closest_boundary = boundary_name
                    
        # Move away from closest boundary
        if closest_boundary:
            if closest_boundary == 'left':
                key = 'd'  # Move right
            elif closest_boundary == 'right':
                key = 'a'  # Move left
            elif closest_boundary == 'top':
                key = 's'  # Move down
            elif closest_boundary == 'bottom':
                key = 'w'  # Move up
            else:
                return actions
                
            actions.append(Action(
                action_type='avoid_boundary',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.CRITICAL,
                confidence=0.95,
                expected_outcome=f"Move away from {closest_boundary} boundary",
                duration=0.3
            ))
            
        return actions
        
    def _generate_boundary_safety_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to maintain safe distance from boundaries"""
        actions = []
        
        player_pos = analysis.player_position
        boundaries = analysis.game_boundaries
        
        if not boundaries:
            return actions
            
        safety_margin = 80  # Desired safety margin
        
        # Check each boundary and generate corrective actions
        for boundary_name, boundary_value in boundaries.items():
            if boundary_name in ['left', 'right']:
                distance = abs(player_pos[0] - boundary_value)
                if distance < safety_margin:
                    if boundary_name == 'left' and player_pos[0] < boundary_value + safety_margin:
                        key = 'd'  # Move right
                    elif boundary_name == 'right' and player_pos[0] > boundary_value - safety_margin:
                        key = 'a'  # Move left
                    else:
                        continue
                        
                    actions.append(Action(
                        action_type='maintain_boundary_safety',
                        parameters={'key': key, 'pressed': True},
                        priority=ActionPriority.HIGH,
                        confidence=0.8,
                        expected_outcome=f"Maintain safe distance from {boundary_name} boundary",
                        duration=0.15
                    ))
                    
            elif boundary_name in ['top', 'bottom']:
                distance = abs(player_pos[1] - boundary_value)
                if distance < safety_margin:
                    if boundary_name == 'top' and player_pos[1] < boundary_value + safety_margin:
                        key = 's'  # Move down
                    elif boundary_name == 'bottom' and player_pos[1] > boundary_value - safety_margin:
                        key = 'w'  # Move up
                    else:
                        continue
                        
                    actions.append(Action(
                        action_type='maintain_boundary_safety',
                        parameters={'key': key, 'pressed': True},
                        priority=ActionPriority.HIGH,
                        confidence=0.8,
                        expected_outcome=f"Maintain safe distance from {boundary_name} boundary",
                        duration=0.15
                    ))
                    
        return actions
        
    def _generate_health_management_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions for health management (seeking health power-ups)"""
        actions = []
        
        # Look for health power-ups
        health_opportunities = [
            opp for opp in analysis.opportunities 
            if opp.opportunity_type == OpportunityType.POWER_UP
        ]
        
        # Filter for health-related power-ups (would need to check power-up type)
        for opportunity in health_opportunities:
            # Move towards health power-up
            player_pos = analysis.player_position
            powerup_pos = opportunity.position
            
            direction_x = powerup_pos[0] - player_pos[0]
            direction_y = powerup_pos[1] - player_pos[1]
            
            # Choose primary movement direction
            if abs(direction_x) > abs(direction_y):
                key = 'd' if direction_x > 0 else 'a'
            else:
                key = 's' if direction_y > 0 else 'w'
                
            actions.append(Action(
                action_type='seek_health_powerup',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.HIGH,
                confidence=0.7,
                expected_outcome="Move towards health power-up",
                duration=0.2
            ))
            
            break  # Only pursue one health power-up at a time
            
        return actions
        
    def _generate_escape_actions(self, analysis: GameStateAnalysis, survival_assessment: Dict[str, Any]) -> List[Action]:
        """Generate actions for escaping dangerous situations"""
        actions = []
        
        escape_routes = survival_assessment['escape_routes']
        if not escape_routes:
            return actions
            
        # Choose the nearest escape route
        player_pos = analysis.player_position
        nearest_escape = min(escape_routes,
                           key=lambda route: math.sqrt(
                               (route[0] - player_pos[0])**2 + 
                               (route[1] - player_pos[1])**2
                           ))
        
        # Move towards escape route
        direction_x = nearest_escape[0] - player_pos[0]
        direction_y = nearest_escape[1] - player_pos[1]
        
        # Choose primary movement direction
        if abs(direction_x) > abs(direction_y):
            key = 'd' if direction_x > 0 else 'a'
        else:
            key = 's' if direction_y > 0 else 'w'
            
        actions.append(Action(
            action_type='escape_to_safety',
            parameters={'key': key, 'pressed': True},
            priority=ActionPriority.CRITICAL,
            confidence=0.9,
            expected_outcome="Escape to safety",
            duration=0.25
        ))
        
        return actions
        
    def _is_position_safe(self, position: Tuple[float, float], boundaries: Dict[str, float]) -> bool:
        """Check if a position is within safe boundaries"""
        if not boundaries:
            return True
            
        x, y = position
        safety_margin = 40
        
        # Check against each boundary
        if 'left' in boundaries and x < boundaries['left'] + safety_margin:
            return False
        if 'right' in boundaries and x > boundaries['right'] - safety_margin:
            return False
        if 'top' in boundaries and y < boundaries['top'] + safety_margin:
            return False
        if 'bottom' in boundaries and y > boundaries['bottom'] - safety_margin:
            return False
            
        return True