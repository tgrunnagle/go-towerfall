"""
Combat rules for the rules-based bot.
Implements specific combat behaviors and decision making.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .rules_based_bot import (
    Action, ActionPriority, GameStateAnalysis, 
    Threat, ThreatType, Opportunity, OpportunityType
)


class CombatRules:
    """
    Combat rules implementation for the rules-based bot.
    Handles targeting, aiming, and combat decision making.
    """
    
    def __init__(self, bot_config: Dict[str, Any]):
        self.config = bot_config
        
    def evaluate_combat_situation(self, analysis: GameStateAnalysis) -> Dict[str, Any]:
        """
        Evaluate the current combat situation and return tactical assessment.
        
        Args:
            analysis: Current game state analysis
            
        Returns:
            Dictionary containing combat situation assessment
        """
        assessment = {
            'threat_level': self._calculate_threat_level(analysis.threats),
            'target_priority': self._prioritize_targets(analysis.enemies, analysis.player_position),
            'combat_stance': self._determine_combat_stance(analysis),
            'engagement_range': self._calculate_optimal_engagement_range(analysis),
            'retreat_recommended': self._should_retreat(analysis)
        }
        
        return assessment
        
    def generate_combat_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """
        Generate combat-specific actions based on game state analysis.
        
        Args:
            analysis: Current game state analysis
            
        Returns:
            List of combat actions
        """
        actions = []
        combat_assessment = self.evaluate_combat_situation(analysis)
        
        # Target enemies based on priority
        if combat_assessment['target_priority']:
            primary_target = combat_assessment['target_priority'][0]
            actions.extend(self._generate_targeting_actions(primary_target, analysis))
        
        # Generate tactical movement actions
        actions.extend(self._generate_tactical_movement(analysis, combat_assessment))
        
        # Generate defensive actions if under threat
        if combat_assessment['threat_level'] > 0.6:
            actions.extend(self._generate_defensive_combat_actions(analysis))
        
        return actions
        
    def _calculate_threat_level(self, threats: List[Threat]) -> float:
        """Calculate overall threat level from 0.0 to 1.0"""
        if not threats:
            return 0.0
            
        # Weight threats by severity and type
        total_threat = 0.0
        threat_weights = {
            ThreatType.PROJECTILE: 1.0,
            ThreatType.ENEMY: 0.8,
            ThreatType.BOUNDARY: 0.3,
            ThreatType.ENVIRONMENTAL: 0.5
        }
        
        for threat in threats:
            weight = threat_weights.get(threat.threat_type, 0.5)
            total_threat += threat.severity * weight
            
        # Normalize to 0-1 range
        return min(1.0, total_threat)
        
    def _prioritize_targets(self, enemies: List[Dict[str, Any]], player_pos: Tuple[float, float]) -> List[Dict[str, Any]]:
        """
        Prioritize enemy targets based on threat level, distance, and vulnerability.
        
        Returns:
            List of enemies sorted by target priority (highest first)
        """
        if not enemies:
            return []
            
        target_scores = []
        
        for enemy in enemies:
            enemy_pos = enemy.get('position', (0, 0))
            distance = math.sqrt(
                (enemy_pos[0] - player_pos[0])**2 + 
                (enemy_pos[1] - player_pos[1])**2
            )
            
            # Base score inversely related to distance
            score = max(0.1, 1.0 - (distance / 400))  # Max effective range 400
            
            # Bonus for low health enemies (easier kills)
            enemy_health = enemy.get('health', 100)
            if enemy_health < 50:
                score *= 1.5
            elif enemy_health < 25:
                score *= 2.0
                
            # Bonus for stationary enemies (easier to hit)
            enemy_vel = enemy.get('velocity', (0, 0))
            if abs(enemy_vel[0]) + abs(enemy_vel[1]) < 5:
                score *= 1.3
                
            # Penalty for enemies behind cover
            if enemy.get('behind_cover', False):
                score *= 0.5
                
            # Bonus for enemies with line of sight
            if enemy.get('line_of_sight', True):
                score *= 1.2
                
            target_scores.append((score, enemy))
            
        # Sort by score (highest first)
        target_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [enemy for score, enemy in target_scores]
        
    def _determine_combat_stance(self, analysis: GameStateAnalysis) -> str:
        """
        Determine the appropriate combat stance based on situation.
        
        Returns:
            Combat stance: 'aggressive', 'defensive', 'balanced', 'retreat'
        """
        threat_level = self._calculate_threat_level(analysis.threats)
        enemy_count = len(analysis.enemies)
        player_health = analysis.player_health
        
        # Retreat if low health and high threat
        if player_health < 30 and threat_level > 0.7:
            return 'retreat'
            
        # Defensive if outnumbered or low health
        if enemy_count > 2 or player_health < 50:
            return 'defensive'
            
        # Aggressive if healthy and enemies are vulnerable
        if player_health > 70 and enemy_count <= 1:
            return 'aggressive'
            
        # Default to balanced
        return 'balanced'
        
    def _calculate_optimal_engagement_range(self, analysis: GameStateAnalysis) -> float:
        """Calculate optimal engagement range based on situation"""
        base_range = 150  # Base engagement range
        
        # Adjust based on player health
        health_factor = analysis.player_health / 100
        range_modifier = 1.0 + (1.0 - health_factor) * 0.5  # Longer range when low health
        
        # Adjust based on enemy count
        enemy_count = len(analysis.enemies)
        if enemy_count > 1:
            range_modifier *= 1.3  # Stay further back when outnumbered
            
        return base_range * range_modifier
        
    def _should_retreat(self, analysis: GameStateAnalysis) -> bool:
        """Determine if retreat is recommended"""
        threat_level = self._calculate_threat_level(analysis.threats)
        
        # Retreat conditions
        if analysis.player_health < 25:  # Very low health
            return True
            
        if threat_level > 0.8 and len(analysis.enemies) > 2:  # Overwhelmed
            return True
            
        if len(analysis.safe_zones) > 0 and threat_level > 0.6:  # Safe zones available
            return True
            
        return False
        
    def _generate_targeting_actions(self, target: Dict[str, Any], analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions for targeting and attacking an enemy"""
        actions = []
        target_pos = target.get('position', (0, 0))
        player_pos = analysis.player_position
        
        distance = math.sqrt(
            (target_pos[0] - player_pos[0])**2 + 
            (target_pos[1] - player_pos[1])**2
        )
        
        # Calculate aim point with prediction
        aim_point = self._calculate_aim_point(target, analysis)
        
        # Generate shooting action
        if distance < 300:  # Within effective range
            confidence = self._calculate_shot_confidence(target, analysis)
            
            actions.append(Action(
                action_type='shoot_at_target',
                parameters={
                    'button': 'left',
                    'pressed': True,
                    'x': aim_point[0],
                    'y': aim_point[1],
                    'target_id': target.get('id'),
                    'hold_duration': 0.1
                },
                priority=ActionPriority.HIGH,
                confidence=confidence,
                expected_outcome=f"Attack target at {target_pos}",
                duration=0.1
            ))
            
        return actions
        
    def _calculate_aim_point(self, target: Dict[str, Any], analysis: GameStateAnalysis) -> Tuple[float, float]:
        """
        Calculate where to aim when shooting at a target, including prediction.
        
        Args:
            target: Target enemy information
            analysis: Current game state analysis
            
        Returns:
            Aim point coordinates (x, y)
        """
        target_pos = target.get('position', (0, 0))
        target_vel = target.get('velocity', (0, 0))
        
        # Simple prediction: assume projectile speed and calculate intercept
        projectile_speed = 300  # Estimated projectile speed
        player_pos = analysis.player_position
        
        # Calculate time to target
        distance = math.sqrt(
            (target_pos[0] - player_pos[0])**2 + 
            (target_pos[1] - player_pos[1])**2
        )
        time_to_target = distance / projectile_speed
        
        # Predict target position
        predicted_x = target_pos[0] + target_vel[0] * time_to_target
        predicted_y = target_pos[1] + target_vel[1] * time_to_target
        
        # Add some randomness based on accuracy
        accuracy = self.config.get('accuracy_modifier', 1.0)
        if accuracy < 1.0:
            error_range = (1.0 - accuracy) * 30  # Up to 30 pixel error
            import random
            predicted_x += random.uniform(-error_range, error_range)
            predicted_y += random.uniform(-error_range, error_range)
            
        return (predicted_x, predicted_y)
        
    def _calculate_shot_confidence(self, target: Dict[str, Any], analysis: GameStateAnalysis) -> float:
        """Calculate confidence level for a shot"""
        target_pos = target.get('position', (0, 0))
        player_pos = analysis.player_position
        
        distance = math.sqrt(
            (target_pos[0] - player_pos[0])**2 + 
            (target_pos[1] - player_pos[1])**2
        )
        
        # Base confidence inversely related to distance
        confidence = max(0.2, 1.0 - (distance / 300))
        
        # Adjust for target movement
        target_vel = target.get('velocity', (0, 0))
        target_speed = math.sqrt(target_vel[0]**2 + target_vel[1]**2)
        if target_speed > 50:  # Fast moving target
            confidence *= 0.7
        elif target_speed < 10:  # Stationary target
            confidence *= 1.3
            
        # Adjust for line of sight
        if not target.get('line_of_sight', True):
            confidence *= 0.3
            
        # Apply accuracy modifier
        confidence *= self.config.get('accuracy_modifier', 1.0)
        
        return min(1.0, confidence)
        
    def _generate_tactical_movement(self, analysis: GameStateAnalysis, combat_assessment: Dict[str, Any]) -> List[Action]:
        """Generate tactical movement actions for combat"""
        actions = []
        stance = combat_assessment['combat_stance']
        
        if stance == 'aggressive':
            # Move closer to enemies for better shots
            actions.extend(self._generate_aggressive_movement(analysis))
        elif stance == 'defensive':
            # Maintain distance and use cover
            actions.extend(self._generate_defensive_movement(analysis))
        elif stance == 'retreat':
            # Move to safety
            actions.extend(self._generate_retreat_movement(analysis))
        else:  # balanced
            # Maintain optimal engagement range
            actions.extend(self._generate_balanced_movement(analysis, combat_assessment))
            
        return actions
        
    def _generate_aggressive_movement(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate movement actions for aggressive combat stance"""
        actions = []
        
        if analysis.enemies:
            closest_enemy = min(analysis.enemies, 
                              key=lambda e: math.sqrt(
                                  (e.get('position', (0, 0))[0] - analysis.player_position[0])**2 + 
                                  (e.get('position', (0, 0))[1] - analysis.player_position[1])**2
                              ))
            
            enemy_pos = closest_enemy.get('position', (0, 0))
            direction_x = enemy_pos[0] - analysis.player_position[0]
            direction_y = enemy_pos[1] - analysis.player_position[1]
            
            # Move towards enemy but maintain some distance
            distance = math.sqrt(direction_x**2 + direction_y**2)
            if distance > 120:  # Too far, move closer
                if abs(direction_x) > abs(direction_y):
                    key = 'd' if direction_x > 0 else 'a'
                else:
                    key = 'w' if direction_y < 0 else 's'  # Inverted Y for screen coordinates
                    
                actions.append(Action(
                    action_type='aggressive_advance',
                    parameters={'key': key, 'pressed': True},
                    priority=ActionPriority.MEDIUM,
                    confidence=0.8,
                    expected_outcome="Move closer to enemy",
                    duration=0.15
                ))
                
        return actions
        
    def _generate_defensive_movement(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate movement actions for defensive combat stance"""
        actions = []
        
        # Move away from threats
        for threat in analysis.threats[:2]:  # Top 2 threats
            if threat.threat_type in [ThreatType.PROJECTILE, ThreatType.ENEMY]:
                threat_to_player = (
                    analysis.player_position[0] - threat.position[0],
                    analysis.player_position[1] - threat.position[1]
                )
                
                # Move away from threat
                if abs(threat_to_player[0]) > abs(threat_to_player[1]):
                    key = 'd' if threat_to_player[0] > 0 else 'a'
                else:
                    key = 's' if threat_to_player[1] > 0 else 'w'
                    
                actions.append(Action(
                    action_type='defensive_retreat',
                    parameters={'key': key, 'pressed': True},
                    priority=ActionPriority.HIGH,
                    confidence=0.9,
                    expected_outcome=f"Move away from {threat.threat_type.value}",
                    duration=0.2
                ))
                
        return actions
        
    def _generate_retreat_movement(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate movement actions for retreat"""
        actions = []
        
        # Move to nearest safe zone
        if analysis.safe_zones:
            nearest_safe_zone = min(analysis.safe_zones,
                                  key=lambda sz: math.sqrt(
                                      (sz[0] - analysis.player_position[0])**2 + 
                                      (sz[1] - analysis.player_position[1])**2
                                  ))
            
            direction_x = nearest_safe_zone[0] - analysis.player_position[0]
            direction_y = nearest_safe_zone[1] - analysis.player_position[1]
            
            if abs(direction_x) > abs(direction_y):
                key = 'd' if direction_x > 0 else 'a'
            else:
                key = 's' if direction_y > 0 else 'w'
                
            actions.append(Action(
                action_type='retreat_to_safety',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.CRITICAL,
                confidence=0.95,
                expected_outcome="Retreat to safe zone",
                duration=0.3
            ))
            
        return actions
        
    def _generate_balanced_movement(self, analysis: GameStateAnalysis, combat_assessment: Dict[str, Any]) -> List[Action]:
        """Generate movement actions for balanced combat stance"""
        actions = []
        optimal_range = combat_assessment['engagement_range']
        
        if analysis.enemies:
            closest_enemy = min(analysis.enemies, 
                              key=lambda e: math.sqrt(
                                  (e.get('position', (0, 0))[0] - analysis.player_position[0])**2 + 
                                  (e.get('position', (0, 0))[1] - analysis.player_position[1])**2
                              ))
            
            enemy_pos = closest_enemy.get('position', (0, 0))
            current_distance = math.sqrt(
                (enemy_pos[0] - analysis.player_position[0])**2 + 
                (enemy_pos[1] - analysis.player_position[1])**2
            )
            
            # Maintain optimal range
            if current_distance < optimal_range * 0.8:  # Too close
                # Move away
                direction_x = analysis.player_position[0] - enemy_pos[0]
                direction_y = analysis.player_position[1] - enemy_pos[1]
            elif current_distance > optimal_range * 1.2:  # Too far
                # Move closer
                direction_x = enemy_pos[0] - analysis.player_position[0]
                direction_y = enemy_pos[1] - analysis.player_position[1]
            else:
                # Good range, no movement needed
                return actions
                
            if abs(direction_x) > abs(direction_y):
                key = 'd' if direction_x > 0 else 'a'
            else:
                key = 's' if direction_y > 0 else 'w'
                
            actions.append(Action(
                action_type='maintain_range',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.MEDIUM,
                confidence=0.7,
                expected_outcome="Maintain optimal engagement range",
                duration=0.1
            ))
            
        return actions
        
    def _generate_defensive_combat_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate defensive combat actions when under high threat"""
        actions = []
        
        # Evasive maneuvers for projectiles
        projectile_threats = [t for t in analysis.threats if t.threat_type == ThreatType.PROJECTILE]
        
        for threat in projectile_threats[:2]:  # Handle top 2 projectile threats
            if threat.time_to_impact and threat.time_to_impact < 1.0:  # Incoming projectile
                # Quick evasive action
                threat_to_player = (
                    analysis.player_position[0] - threat.position[0],
                    analysis.player_position[1] - threat.position[1]
                )
                
                # Move perpendicular to projectile path
                if threat.velocity:
                    # Move perpendicular to projectile velocity
                    if abs(threat.velocity[0]) > abs(threat.velocity[1]):
                        key = 'w' if threat_to_player[1] > 0 else 's'
                    else:
                        key = 'd' if threat_to_player[0] > 0 else 'a'
                else:
                    # Move away from projectile
                    if abs(threat_to_player[0]) > abs(threat_to_player[1]):
                        key = 'd' if threat_to_player[0] > 0 else 'a'
                    else:
                        key = 's' if threat_to_player[1] > 0 else 'w'
                
                actions.append(Action(
                    action_type='evasive_maneuver',
                    parameters={'key': key, 'pressed': True},
                    priority=ActionPriority.CRITICAL,
                    confidence=0.9,
                    expected_outcome="Evade incoming projectile",
                    duration=0.15
                ))
                
        return actions