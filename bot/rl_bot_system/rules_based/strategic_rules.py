"""
Strategic rules for the rules-based bot.
Implements high-level strategic behaviors, territory control, and tactical planning.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from .rules_based_bot import (
    Action, ActionPriority, GameStateAnalysis, 
    Threat, ThreatType, Opportunity, OpportunityType
)


class StrategicRules:
    """
    Strategic rules implementation for the rules-based bot.
    Handles territory control, power-up collection, and tactical positioning.
    """
    
    def __init__(self, bot_config: Dict[str, Any]):
        self.config = bot_config
        self.territory_map = {}  # Track controlled areas
        self.strategic_objectives = []  # Current strategic goals
        
    def evaluate_strategic_situation(self, analysis: GameStateAnalysis) -> Dict[str, Any]:
        """
        Evaluate the current strategic situation and return assessment.
        
        Args:
            analysis: Current game state analysis
            
        Returns:
            Dictionary containing strategic situation assessment
        """
        assessment = {
            'map_control': self._assess_map_control(analysis),
            'power_up_control': self._assess_power_up_control(analysis),
            'positional_advantage': self._assess_positional_advantage(analysis),
            'strategic_objectives': self._identify_strategic_objectives(analysis),
            'timing_opportunities': self._identify_timing_opportunities(analysis)
        }
        
        return assessment
        
    def generate_strategic_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """
        Generate strategic actions based on game state analysis.
        
        Args:
            analysis: Current game state analysis
            
        Returns:
            List of strategic actions
        """
        actions = []
        strategic_assessment = self.evaluate_strategic_situation(analysis)
        
        # Execute strategic objectives
        for objective in strategic_assessment['strategic_objectives']:
            actions.extend(self._execute_strategic_objective(objective, analysis))
        
        # Control key areas of the map
        if strategic_assessment['map_control']['control_percentage'] < 0.6:
            actions.extend(self._generate_territory_control_actions(analysis))
        
        # Collect strategic power-ups
        actions.extend(self._generate_power_up_collection_actions(analysis, strategic_assessment))
        
        # Maintain positional advantage
        if strategic_assessment['positional_advantage'] < 0.5:
            actions.extend(self._generate_positioning_actions(analysis))
        
        return actions
        
    def _assess_map_control(self, analysis: GameStateAnalysis) -> Dict[str, Any]:
        """Assess how much of the map the bot controls"""
        # Simplified map control assessment
        player_pos = analysis.player_position
        boundaries = analysis.game_boundaries
        
        if not boundaries:
            return {'control_percentage': 0.5, 'controlled_areas': [], 'contested_areas': []}
        
        # Calculate influence zones around player and enemies
        player_influence = self._calculate_influence_zone(player_pos, 100)  # 100 unit radius
        
        enemy_influences = []
        for enemy in analysis.enemies:
            enemy_pos = enemy.get('position', (0, 0))
            enemy_influences.append(self._calculate_influence_zone(enemy_pos, 100))
        
        # Calculate control percentage (simplified)
        total_map_area = self._calculate_map_area(boundaries)
        controlled_area = self._calculate_controlled_area(player_influence, enemy_influences, boundaries)
        
        control_percentage = controlled_area / total_map_area if total_map_area > 0 else 0.5
        
        return {
            'control_percentage': min(1.0, control_percentage),
            'controlled_areas': [player_influence],
            'contested_areas': self._find_contested_areas(player_influence, enemy_influences)
        }
        
    def _assess_power_up_control(self, analysis: GameStateAnalysis) -> Dict[str, Any]:
        """Assess control over power-up spawns and collection"""
        power_up_opportunities = [
            opp for opp in analysis.opportunities 
            if opp.opportunity_type == OpportunityType.POWER_UP
        ]
        
        # Calculate distance to each power-up vs enemy distances
        controlled_power_ups = 0
        contested_power_ups = 0
        
        for opportunity in power_up_opportunities:
            player_distance = opportunity.distance
            
            # Check if any enemy is closer
            enemy_closer = False
            for enemy in analysis.enemies:
                enemy_pos = enemy.get('position', (0, 0))
                enemy_distance = math.sqrt(
                    (opportunity.position[0] - enemy_pos[0])**2 + 
                    (opportunity.position[1] - enemy_pos[1])**2
                )
                
                if enemy_distance < player_distance * 0.8:  # Enemy significantly closer
                    enemy_closer = True
                    break
            
            if not enemy_closer:
                controlled_power_ups += 1
            else:
                contested_power_ups += 1
        
        total_power_ups = len(power_up_opportunities)
        control_ratio = controlled_power_ups / total_power_ups if total_power_ups > 0 else 1.0
        
        return {
            'control_ratio': control_ratio,
            'controlled_count': controlled_power_ups,
            'contested_count': contested_power_ups,
            'total_count': total_power_ups
        }
        
    def _assess_positional_advantage(self, analysis: GameStateAnalysis) -> float:
        """
        Assess positional advantage from 0.0 to 1.0.
        Higher values indicate better positioning.
        """
        advantage = 0.5  # Base neutral position
        
        # Height advantage (if platform data available)
        player_y = analysis.player_position[1]
        enemy_heights = [enemy.get('position', (0, 0))[1] for enemy in analysis.enemies]
        
        if enemy_heights:
            avg_enemy_height = sum(enemy_heights) / len(enemy_heights)
            if player_y < avg_enemy_height:  # Higher position (lower Y in screen coordinates)
                advantage += 0.2
            elif player_y > avg_enemy_height:
                advantage -= 0.1
        
        # Center control advantage
        if analysis.game_boundaries:
            center_x = (analysis.game_boundaries.get('left', 0) + analysis.game_boundaries.get('right', 800)) / 2
            center_y = (analysis.game_boundaries.get('top', 0) + analysis.game_boundaries.get('bottom', 600)) / 2
            
            distance_from_center = math.sqrt(
                (analysis.player_position[0] - center_x)**2 + 
                (analysis.player_position[1] - center_y)**2
            )
            
            # Closer to center is generally better
            max_distance = math.sqrt((center_x)**2 + (center_y)**2)
            center_advantage = 1.0 - (distance_from_center / max_distance)
            advantage += center_advantage * 0.2
        
        # Cover and safety advantage
        if len(analysis.safe_zones) > 0:
            nearest_safe_zone_distance = min(
                math.sqrt(
                    (sz[0] - analysis.player_position[0])**2 + 
                    (sz[1] - analysis.player_position[1])**2
                ) for sz in analysis.safe_zones
            )
            
            if nearest_safe_zone_distance < 50:  # Close to safety
                advantage += 0.1
        
        return min(1.0, max(0.0, advantage))
        
    def _identify_strategic_objectives(self, analysis: GameStateAnalysis) -> List[Dict[str, Any]]:
        """Identify current strategic objectives"""
        objectives = []
        
        # Objective 1: Control high-value power-ups
        high_value_power_ups = [
            opp for opp in analysis.opportunities 
            if opp.opportunity_type == OpportunityType.POWER_UP and opp.value > 0.7
        ]
        
        for power_up in high_value_power_ups[:2]:  # Top 2 high-value power-ups
            objectives.append({
                'type': 'control_power_up',
                'target': power_up.position,
                'priority': power_up.priority,
                'description': f"Control high-value power-up at {power_up.position}"
            })
        
        # Objective 2: Eliminate isolated enemies
        isolated_enemies = self._find_isolated_enemies(analysis)
        for enemy in isolated_enemies[:1]:  # Focus on one isolated enemy
            objectives.append({
                'type': 'eliminate_target',
                'target': enemy.get('position', (0, 0)),
                'priority': 0.8,
                'description': f"Eliminate isolated enemy at {enemy.get('position')}"
            })
        
        # Objective 3: Control strategic positions
        strategic_positions = [
            opp for opp in analysis.opportunities 
            if opp.opportunity_type == OpportunityType.STRATEGIC_POSITION
        ]
        
        for position in strategic_positions[:1]:  # Focus on one strategic position
            objectives.append({
                'type': 'control_position',
                'target': position.position,
                'priority': position.priority,
                'description': f"Control strategic position at {position.position}"
            })
        
        # Sort by priority
        objectives.sort(key=lambda obj: obj['priority'], reverse=True)
        
        return objectives[:3]  # Limit to top 3 objectives
        
    def _identify_timing_opportunities(self, analysis: GameStateAnalysis) -> List[Dict[str, Any]]:
        """Identify timing-based opportunities"""
        opportunities = []
        
        # Power-up spawn timing (simplified - would need game-specific timing data)
        for power_up_opp in analysis.opportunities:
            if power_up_opp.opportunity_type == OpportunityType.POWER_UP:
                # Check if we can reach power-up before enemies
                time_to_reach = power_up_opp.distance / 100  # Assume 100 units/second movement
                
                enemy_times = []
                for enemy in analysis.enemies:
                    enemy_pos = enemy.get('position', (0, 0))
                    enemy_distance = math.sqrt(
                        (power_up_opp.position[0] - enemy_pos[0])**2 + 
                        (power_up_opp.position[1] - enemy_pos[1])**2
                    )
                    enemy_times.append(enemy_distance / 100)
                
                if not enemy_times or time_to_reach < min(enemy_times):
                    opportunities.append({
                        'type': 'power_up_timing',
                        'target': power_up_opp.position,
                        'time_advantage': min(enemy_times) - time_to_reach if enemy_times else 1.0,
                        'priority': 0.7
                    })
        
        return opportunities
        
    def _execute_strategic_objective(self, objective: Dict[str, Any], analysis: GameStateAnalysis) -> List[Action]:
        """Execute a specific strategic objective"""
        actions = []
        
        if objective['type'] == 'control_power_up':
            actions.extend(self._generate_power_up_control_actions(objective, analysis))
        elif objective['type'] == 'eliminate_target':
            actions.extend(self._generate_target_elimination_actions(objective, analysis))
        elif objective['type'] == 'control_position':
            actions.extend(self._generate_position_control_actions(objective, analysis))
        
        return actions
        
    def _generate_territory_control_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to control territory"""
        actions = []
        
        # Move to control key areas of the map
        if analysis.game_boundaries:
            # Identify key control points (corners, center, etc.)
            center_x = (analysis.game_boundaries.get('left', 0) + analysis.game_boundaries.get('right', 800)) / 2
            center_y = (analysis.game_boundaries.get('top', 0) + analysis.game_boundaries.get('bottom', 600)) / 2
            
            # Move towards center if not already there
            distance_to_center = math.sqrt(
                (analysis.player_position[0] - center_x)**2 + 
                (analysis.player_position[1] - center_y)**2
            )
            
            if distance_to_center > 100:  # Not in center area
                direction_x = center_x - analysis.player_position[0]
                direction_y = center_y - analysis.player_position[1]
                
                if abs(direction_x) > abs(direction_y):
                    key = 'd' if direction_x > 0 else 'a'
                else:
                    key = 's' if direction_y > 0 else 'w'
                
                actions.append(Action(
                    action_type='control_territory',
                    parameters={'key': key, 'pressed': True},
                    priority=ActionPriority.MEDIUM,
                    confidence=0.6,
                    expected_outcome="Move to control center territory",
                    duration=0.2
                ))
        
        return actions
        
    def _generate_power_up_collection_actions(self, analysis: GameStateAnalysis, strategic_assessment: Dict[str, Any]) -> List[Action]:
        """Generate actions for strategic power-up collection"""
        actions = []
        
        # Focus on power-ups we can control
        power_up_control = strategic_assessment['power_up_control']
        
        if power_up_control['controlled_count'] > 0:
            # Move to collect controlled power-ups
            controlled_power_ups = []
            
            for opportunity in analysis.opportunities:
                if opportunity.opportunity_type == OpportunityType.POWER_UP:
                    # Check if this power-up is controlled by us
                    player_distance = opportunity.distance
                    
                    enemy_closer = False
                    for enemy in analysis.enemies:
                        enemy_pos = enemy.get('position', (0, 0))
                        enemy_distance = math.sqrt(
                            (opportunity.position[0] - enemy_pos[0])**2 + 
                            (opportunity.position[1] - enemy_pos[1])**2
                        )
                        
                        if enemy_distance < player_distance * 0.8:
                            enemy_closer = True
                            break
                    
                    if not enemy_closer:
                        controlled_power_ups.append(opportunity)
            
            # Move to nearest controlled power-up
            if controlled_power_ups:
                nearest_power_up = min(controlled_power_ups, key=lambda opp: opp.distance)
                
                direction_x = nearest_power_up.position[0] - analysis.player_position[0]
                direction_y = nearest_power_up.position[1] - analysis.player_position[1]
                
                if abs(direction_x) > abs(direction_y):
                    key = 'd' if direction_x > 0 else 'a'
                else:
                    key = 's' if direction_y > 0 else 'w'
                
                actions.append(Action(
                    action_type='collect_strategic_powerup',
                    parameters={'key': key, 'pressed': True},
                    priority=ActionPriority.MEDIUM,
                    confidence=0.8,
                    expected_outcome="Collect strategic power-up",
                    duration=0.15
                ))
        
        return actions
        
    def _generate_positioning_actions(self, analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to improve positioning"""
        actions = []
        
        # Move to high ground if available
        strategic_positions = [
            opp for opp in analysis.opportunities 
            if opp.opportunity_type == OpportunityType.STRATEGIC_POSITION
        ]
        
        if strategic_positions:
            best_position = max(strategic_positions, key=lambda opp: opp.value)
            
            direction_x = best_position.position[0] - analysis.player_position[0]
            direction_y = best_position.position[1] - analysis.player_position[1]
            
            if abs(direction_x) > abs(direction_y):
                key = 'd' if direction_x > 0 else 'a'
            else:
                key = 's' if direction_y > 0 else 'w'
            
            actions.append(Action(
                action_type='improve_positioning',
                parameters={'key': key, 'pressed': True},
                priority=ActionPriority.LOW,
                confidence=0.7,
                expected_outcome="Move to strategic position",
                duration=0.2
            ))
        
        return actions
        
    def _generate_power_up_control_actions(self, objective: Dict[str, Any], analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to control a specific power-up"""
        actions = []
        
        target_pos = objective['target']
        player_pos = analysis.player_position
        
        # Move towards power-up
        direction_x = target_pos[0] - player_pos[0]
        direction_y = target_pos[1] - player_pos[1]
        
        if abs(direction_x) > abs(direction_y):
            key = 'd' if direction_x > 0 else 'a'
        else:
            key = 's' if direction_y > 0 else 'w'
        
        actions.append(Action(
            action_type='control_powerup_objective',
            parameters={'key': key, 'pressed': True},
            priority=ActionPriority.MEDIUM,
            confidence=0.8,
            expected_outcome=f"Move to control power-up at {target_pos}",
            duration=0.15
        ))
        
        return actions
        
    def _generate_target_elimination_actions(self, objective: Dict[str, Any], analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to eliminate a specific target"""
        actions = []
        
        target_pos = objective['target']
        player_pos = analysis.player_position
        
        # Move to optimal attack position
        distance = math.sqrt(
            (target_pos[0] - player_pos[0])**2 + 
            (target_pos[1] - player_pos[1])**2
        )
        
        optimal_range = 120  # Optimal attack range
        
        if distance > optimal_range * 1.2:  # Too far
            # Move closer
            direction_x = target_pos[0] - player_pos[0]
            direction_y = target_pos[1] - player_pos[1]
        elif distance < optimal_range * 0.8:  # Too close
            # Move away
            direction_x = player_pos[0] - target_pos[0]
            direction_y = player_pos[1] - target_pos[1]
        else:
            # Good range, no movement needed
            return actions
        
        if abs(direction_x) > abs(direction_y):
            key = 'd' if direction_x > 0 else 'a'
        else:
            key = 's' if direction_y > 0 else 'w'
        
        actions.append(Action(
            action_type='position_for_elimination',
            parameters={'key': key, 'pressed': True},
            priority=ActionPriority.HIGH,
            confidence=0.7,
            expected_outcome=f"Position to eliminate target at {target_pos}",
            duration=0.1
        ))
        
        return actions
        
    def _generate_position_control_actions(self, objective: Dict[str, Any], analysis: GameStateAnalysis) -> List[Action]:
        """Generate actions to control a strategic position"""
        actions = []
        
        target_pos = objective['target']
        player_pos = analysis.player_position
        
        # Move to strategic position
        direction_x = target_pos[0] - player_pos[0]
        direction_y = target_pos[1] - player_pos[1]
        
        if abs(direction_x) > abs(direction_y):
            key = 'd' if direction_x > 0 else 'a'
        else:
            key = 's' if direction_y > 0 else 'w'
        
        actions.append(Action(
            action_type='control_strategic_position',
            parameters={'key': key, 'pressed': True},
            priority=ActionPriority.MEDIUM,
            confidence=0.75,
            expected_outcome=f"Move to control strategic position at {target_pos}",
            duration=0.2
        ))
        
        return actions
        
    # Helper methods
    
    def _calculate_influence_zone(self, position: Tuple[float, float], radius: float) -> Dict[str, Any]:
        """Calculate influence zone around a position"""
        return {
            'center': position,
            'radius': radius,
            'area': math.pi * radius**2
        }
        
    def _calculate_map_area(self, boundaries: Dict[str, float]) -> float:
        """Calculate total map area"""
        if not boundaries:
            return 800 * 600  # Default area
            
        width = boundaries.get('right', 800) - boundaries.get('left', 0)
        height = boundaries.get('bottom', 600) - boundaries.get('top', 0)
        
        return width * height
        
    def _calculate_controlled_area(self, player_influence: Dict[str, Any], enemy_influences: List[Dict[str, Any]], boundaries: Dict[str, float]) -> float:
        """Calculate area controlled by player (simplified)"""
        # Simplified calculation - just return player influence area
        # In a real implementation, this would account for overlapping zones
        return player_influence['area']
        
    def _find_contested_areas(self, player_influence: Dict[str, Any], enemy_influences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find areas where player and enemy influences overlap"""
        contested = []
        
        player_pos = player_influence['center']
        player_radius = player_influence['radius']
        
        for enemy_influence in enemy_influences:
            enemy_pos = enemy_influence['center']
            enemy_radius = enemy_influence['radius']
            
            # Check if zones overlap
            distance = math.sqrt(
                (player_pos[0] - enemy_pos[0])**2 + 
                (player_pos[1] - enemy_pos[1])**2
            )
            
            if distance < (player_radius + enemy_radius):
                # Zones overlap - contested area
                contested.append({
                    'center': ((player_pos[0] + enemy_pos[0]) / 2, (player_pos[1] + enemy_pos[1]) / 2),
                    'radius': min(player_radius, enemy_radius),
                    'area': math.pi * min(player_radius, enemy_radius)**2
                })
        
        return contested
        
    def _find_isolated_enemies(self, analysis: GameStateAnalysis) -> List[Dict[str, Any]]:
        """Find enemies that are isolated from other enemies"""
        isolated = []
        
        for i, enemy in enumerate(analysis.enemies):
            enemy_pos = enemy.get('position', (0, 0))
            
            # Check distance to other enemies
            min_distance_to_ally = float('inf')
            for j, other_enemy in enumerate(analysis.enemies):
                if i == j:
                    continue
                    
                other_pos = other_enemy.get('position', (0, 0))
                distance = math.sqrt(
                    (enemy_pos[0] - other_pos[0])**2 + 
                    (enemy_pos[1] - other_pos[1])**2
                )
                
                min_distance_to_ally = min(min_distance_to_ally, distance)
            
            # If enemy is far from allies, consider isolated
            if min_distance_to_ally > 150:  # Isolation threshold
                isolated.append(enemy)
        
        return isolated