"""
Reward function implementations for different reward strategies.

This module provides comprehensive reward function systems with plugin architecture
for different reward calculation strategies including sparse, dense, and shaped rewards.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from enum import Enum


class RewardType(Enum):
    """Types of reward functions available."""
    SPARSE = "sparse"
    DENSE = "dense"
    SHAPED = "shaped"
    MULTI_OBJECTIVE = "multi_objective"
    CURRICULUM = "curriculum"


class TimeHorizon(Enum):
    """Time horizons for reward calculation."""
    SHORT_TERM = "short_term"      # 1-5 steps
    MEDIUM_TERM = "medium_term"    # 10-50 steps
    LONG_TERM = "long_term"        # 100+ steps


class RewardFunction(ABC):
    """Base class for reward functions with plugin architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reward function.
        
        Args:
            config: Configuration dictionary for the reward function
        """
        self.config = config or {}
        self.reward_type = self._get_reward_type()
        self.reward_history = []
        self.episode_rewards = []
    
    @abstractmethod
    def calculate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate reward for the current step."""
        pass
    
    @abstractmethod
    def _get_reward_type(self) -> RewardType:
        """Get the reward type for this function."""
        pass
    
    def reset_episode(self) -> None:
        """Reset for a new episode."""
        if self.reward_history:
            self.episode_rewards.append(sum(self.reward_history))
        self.reward_history = []
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """Get statistics about reward distribution."""
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        return {
            'total_reward': float(np.sum(rewards)),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'positive_rewards': int(np.sum(rewards > 0)),
            'negative_rewards': int(np.sum(rewards < 0)),
            'zero_rewards': int(np.sum(rewards == 0))
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get reward function configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update reward function configuration."""
        self.config.update(new_config)


class SparseRewardFunction(RewardFunction):
    """
    Sparse reward function that only gives rewards for major events.
    
    Provides rewards only for significant game events like wins, losses,
    kills, deaths, and objective completions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sparse reward function.
        
        Config options:
            - win_reward: Reward for winning (default: 10.0)
            - loss_penalty: Penalty for losing (default: -10.0)
            - kill_reward: Reward for killing an enemy (default: 1.0)
            - death_penalty: Penalty for dying (default: -1.0)
            - survival_reward: Small reward for surviving each step (default: 0.01)
            - objective_reward: Reward for completing objectives (default: 2.0)
        """
        default_config = {
            'win_reward': 10.0,
            'loss_penalty': -10.0,
            'kill_reward': 1.0,
            'death_penalty': -1.0,
            'survival_reward': 0.01,
            'objective_reward': 2.0
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _get_reward_type(self) -> RewardType:
        """Get reward type."""
        return RewardType.SPARSE
    
    def calculate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate sparse reward based on major events."""
        reward = 0.0
        
        player = state.get('player', {})
        prev_player = previous_state.get('player', {})
        
        # Death penalty
        if not player.get('is_alive', True) and prev_player.get('is_alive', True):
            reward += self.config['death_penalty']
        
        # Kill reward (check if kills increased)
        prev_kills = getattr(self, '_prev_kills', 0)
        current_kills = episode_stats.get('kills', 0)
        if current_kills > prev_kills:
            reward += self.config['kill_reward'] * (current_kills - prev_kills)
        self._prev_kills = current_kills
        
        # Game end rewards
        if not state.get('round_active', True):
            # Determine if won or lost based on survival and kills
            if player.get('is_alive', True):
                reward += self.config['win_reward']
            else:
                reward += self.config['loss_penalty']
        
        # Small survival reward
        if player.get('is_alive', True):
            reward += self.config['survival_reward']
        
        # Objective-based rewards (if applicable)
        objectives_completed = self._count_objectives_completed(state, previous_state)
        if objectives_completed > 0:
            reward += self.config['objective_reward'] * objectives_completed
        
        self.reward_history.append(reward)
        return reward
    
    def _count_kills_from_state(self, state: Dict[str, Any], episode_stats: Dict[str, Any]) -> int:
        """Count kills from current state (placeholder implementation)."""
        return episode_stats.get('kills', 0)
    
    def _count_objectives_completed(self, state: Dict[str, Any], previous_state: Dict[str, Any]) -> int:
        """Count objectives completed this step."""
        # Placeholder - could be power-ups collected, areas controlled, etc.
        return 0
    
    def reset_episode(self) -> None:
        """Reset for a new episode."""
        super().reset_episode()
        self._prev_kills = 0


class DenseRewardFunction(RewardFunction):
    """
    Dense reward function that provides frequent feedback.
    
    Gives rewards for many actions including damage dealt/taken,
    positioning, movement efficiency, and tactical decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dense reward function.
        
        Config options:
            - damage_dealt_scale: Scale for damage dealt rewards (default: 0.1)
            - damage_taken_scale: Scale for damage taken penalties (default: -0.1)
            - health_differential_scale: Scale for health advantage rewards (default: 0.05)
            - positioning_scale: Scale for positioning rewards (default: 0.02)
            - movement_efficiency_scale: Scale for movement rewards (default: 0.01)
            - ammo_conservation_scale: Scale for ammo conservation (default: 0.005)
        """
        default_config = {
            'damage_dealt_scale': 0.1,
            'damage_taken_scale': -0.1,
            'health_differential_scale': 0.05,
            'positioning_scale': 0.02,
            'movement_efficiency_scale': 0.01,
            'ammo_conservation_scale': 0.005
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _get_reward_type(self) -> RewardType:
        """Get reward type."""
        return RewardType.DENSE
    
    def calculate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate dense reward with frequent feedback."""
        reward = 0.0
        
        player = state.get('player', {})
        prev_player = previous_state.get('player', {})
        enemies = state.get('enemies', [])
        
        # Damage dealt reward
        damage_dealt = episode_stats.get('damage_dealt', 0)
        prev_damage_dealt = getattr(self, '_prev_damage_dealt', 0)
        if damage_dealt > prev_damage_dealt:
            reward += (damage_dealt - prev_damage_dealt) * self.config['damage_dealt_scale']
        self._prev_damage_dealt = damage_dealt
        
        # Damage taken penalty
        damage_taken = episode_stats.get('damage_taken', 0)
        prev_damage_taken = getattr(self, '_prev_damage_taken', 0)
        if damage_taken > prev_damage_taken:
            reward += (damage_taken - prev_damage_taken) * self.config['damage_taken_scale']
        self._prev_damage_taken = damage_taken
        
        # Health differential reward
        player_health = player.get('health', 100)
        if enemies:
            avg_enemy_health = np.mean([e.get('health', 100) for e in enemies])
            health_advantage = (player_health - avg_enemy_health) / 100.0
            reward += health_advantage * self.config['health_differential_scale']
        
        # Positioning reward
        positioning_reward = self._calculate_positioning_reward(state)
        reward += positioning_reward * self.config['positioning_scale']
        
        # Movement efficiency reward
        movement_reward = self._calculate_movement_efficiency(state, previous_state, action)
        reward += movement_reward * self.config['movement_efficiency_scale']
        
        # Ammo conservation reward
        ammo_reward = self._calculate_ammo_conservation(state, previous_state, action)
        reward += ammo_reward * self.config['ammo_conservation_scale']
        
        self.reward_history.append(reward)
        return reward
    
    def _calculate_positioning_reward(self, state: Dict[str, Any]) -> float:
        """Calculate reward for good positioning."""
        player = state.get('player', {})
        enemies = state.get('enemies', [])
        boundaries = state.get('boundaries', {})
        
        if not enemies:
            return 0.0
        
        player_pos = player.get('position', {'x': 0, 'y': 0})
        
        # Reward for maintaining good distance from enemies
        distances = []
        for enemy in enemies:
            enemy_pos = enemy.get('position', {'x': 0, 'y': 0})
            distance = self._calculate_distance(player_pos, enemy_pos)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        min_distance = min(distances)
        
        # Optimal distance is neither too close nor too far
        optimal_distance = 150.0
        distance_reward = 1.0 - abs(avg_distance - optimal_distance) / optimal_distance
        
        # Penalty for being too close to any enemy
        if min_distance < 50.0:
            distance_reward -= 0.5
        
        # Reward for staying away from boundaries
        boundary_reward = self._calculate_boundary_safety(player_pos, boundaries)
        
        return (distance_reward + boundary_reward) / 2.0
    
    def _calculate_movement_efficiency(
        self, 
        state: Dict[str, Any], 
        previous_state: Dict[str, Any], 
        action: Any
    ) -> float:
        """Calculate reward for efficient movement."""
        player = state.get('player', {})
        prev_player = previous_state.get('player', {})
        
        player_pos = player.get('position', {'x': 0, 'y': 0})
        prev_pos = prev_player.get('position', {'x': 0, 'y': 0})
        
        # Calculate movement distance
        movement_distance = self._calculate_distance(player_pos, prev_pos)
        
        # Reward purposeful movement, penalize excessive movement
        if movement_distance > 0:
            # Check if movement is towards objectives or away from threats
            enemies = state.get('enemies', [])
            if enemies:
                nearest_enemy = min(enemies, key=lambda e: self._calculate_distance(
                    player_pos, e.get('position', {'x': 0, 'y': 0})
                ))
                enemy_pos = nearest_enemy.get('position', {'x': 0, 'y': 0})
                
                # Reward moving away from very close enemies
                enemy_distance = self._calculate_distance(player_pos, enemy_pos)
                prev_enemy_distance = self._calculate_distance(prev_pos, enemy_pos)
                
                if prev_enemy_distance < 75.0 and enemy_distance > prev_enemy_distance:
                    return 1.0  # Good evasive movement
                elif enemy_distance < 200.0 and enemy_distance < prev_enemy_distance:
                    return 0.5  # Good aggressive movement
        
        # Penalize excessive movement without purpose
        if movement_distance > 100.0:
            return -0.5
        
        return 0.0
    
    def _calculate_ammo_conservation(
        self, 
        state: Dict[str, Any], 
        previous_state: Dict[str, Any], 
        action: Any
    ) -> float:
        """Calculate reward for ammo conservation."""
        player = state.get('player', {})
        prev_player = previous_state.get('player', {})
        
        current_ammo = player.get('ammunition', 10)
        prev_ammo = prev_player.get('ammunition', 10)
        
        # If ammo decreased (shot fired)
        if current_ammo < prev_ammo:
            shots_fired = prev_ammo - current_ammo
            
            # Check if shots were effective (hit enemies)
            shots_hit = episode_stats.get('shots_hit', 0) - getattr(self, '_prev_shots_hit', 0)
            self._prev_shots_hit = episode_stats.get('shots_hit', 0)
            
            if shots_hit > 0:
                return 1.0  # Reward accurate shooting
            else:
                return -0.5 * shots_fired  # Penalize missed shots
        
        return 0.0
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def _calculate_boundary_safety(self, position: Dict[str, float], boundaries: Dict[str, float]) -> float:
        """Calculate safety score based on distance from boundaries."""
        if not boundaries:
            return 0.0
        
        x, y = position['x'], position['y']
        left = boundaries.get('left', -400)
        right = boundaries.get('right', 400)
        top = boundaries.get('top', -300)
        bottom = boundaries.get('bottom', 300)
        
        # Distance to each boundary
        distances = [
            abs(x - left),
            abs(x - right),
            abs(y - top),
            abs(y - bottom)
        ]
        
        min_distance = min(distances)
        safe_distance = 100.0  # Minimum safe distance
        
        if min_distance < safe_distance:
            return -1.0 + (min_distance / safe_distance)
        else:
            return 0.5  # Bonus for being safely away from boundaries


class ShapedRewardFunction(RewardFunction):
    """
    Shaped reward function that guides learning with intermediate rewards.
    
    Provides carefully designed rewards to guide the agent towards
    desired behaviors and strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize shaped reward function.
        
        Config options:
            - aim_accuracy_scale: Scale for aim accuracy rewards (default: 0.2)
            - tactical_positioning_scale: Scale for tactical positioning (default: 0.15)
            - threat_avoidance_scale: Scale for threat avoidance (default: 0.1)
            - resource_management_scale: Scale for resource management (default: 0.05)
            - exploration_bonus_scale: Scale for exploration bonuses (default: 0.02)
        """
        default_config = {
            'aim_accuracy_scale': 0.2,
            'tactical_positioning_scale': 0.15,
            'threat_avoidance_scale': 0.1,
            'resource_management_scale': 0.05,
            'exploration_bonus_scale': 0.02
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.visited_positions = set()
    
    def _get_reward_type(self) -> RewardType:
        """Get reward type."""
        return RewardType.SHAPED
    
    def calculate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate shaped reward to guide learning."""
        reward = 0.0
        
        # Aim accuracy reward
        aim_reward = self._calculate_aim_accuracy_reward(state, action, episode_stats)
        reward += aim_reward * self.config['aim_accuracy_scale']
        
        # Tactical positioning reward
        tactical_reward = self._calculate_tactical_positioning_reward(state)
        reward += tactical_reward * self.config['tactical_positioning_scale']
        
        # Threat avoidance reward
        threat_reward = self._calculate_threat_avoidance_reward(state, previous_state)
        reward += threat_reward * self.config['threat_avoidance_scale']
        
        # Resource management reward
        resource_reward = self._calculate_resource_management_reward(state, previous_state)
        reward += resource_reward * self.config['resource_management_scale']
        
        # Exploration bonus
        exploration_reward = self._calculate_exploration_bonus(state)
        reward += exploration_reward * self.config['exploration_bonus_scale']
        
        self.reward_history.append(reward)
        return reward
    
    def _calculate_aim_accuracy_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate reward for aim accuracy."""
        # This would analyze if the agent is aiming towards enemies
        # For now, use hit rate as a proxy
        shots_fired = episode_stats.get('shots_fired', 0)
        shots_hit = episode_stats.get('shots_hit', 0)
        
        if shots_fired > 0:
            accuracy = shots_hit / shots_fired
            return accuracy * 2.0 - 1.0  # Scale to [-1, 1]
        
        return 0.0
    
    def _calculate_tactical_positioning_reward(self, state: Dict[str, Any]) -> float:
        """Calculate reward for tactical positioning."""
        player = state.get('player', {})
        enemies = state.get('enemies', [])
        power_ups = state.get('power_ups', [])
        
        if not enemies:
            return 0.0
        
        player_pos = player.get('position', {'x': 0, 'y': 0})
        reward = 0.0
        
        # Reward for flanking positions
        for enemy in enemies:
            enemy_pos = enemy.get('position', {'x': 0, 'y': 0})
            enemy_vel = enemy.get('velocity', {'x': 0, 'y': 0})
            
            # Calculate if player is in a flanking position
            flank_score = self._calculate_flank_score(player_pos, enemy_pos, enemy_vel)
            reward += flank_score
        
        # Reward for controlling high-value areas
        if power_ups:
            nearest_powerup = min(power_ups, key=lambda p: self._calculate_distance(
                player_pos, p.get('position', {'x': 0, 'y': 0})
            ))
            powerup_distance = self._calculate_distance(
                player_pos, nearest_powerup.get('position', {'x': 0, 'y': 0})
            )
            
            # Reward being near power-ups
            if powerup_distance < 100.0:
                reward += 0.5
        
        return np.clip(reward, -1.0, 1.0)
    
    def _calculate_threat_avoidance_reward(
        self, 
        state: Dict[str, Any], 
        previous_state: Dict[str, Any]
    ) -> float:
        """Calculate reward for avoiding threats."""
        player = state.get('player', {})
        prev_player = previous_state.get('player', {})
        projectiles = state.get('projectiles', [])
        
        player_pos = player.get('position', {'x': 0, 'y': 0})
        prev_pos = prev_player.get('position', {'x': 0, 'y': 0})
        
        reward = 0.0
        
        # Reward for avoiding incoming projectiles
        for projectile in projectiles:
            proj_pos = projectile.get('position', {'x': 0, 'y': 0})
            proj_vel = projectile.get('velocity', {'x': 0, 'y': 0})
            
            # Check if projectile is threatening
            if self._is_projectile_threatening(prev_pos, proj_pos, proj_vel):
                # Check if player moved away from threat
                prev_threat_distance = self._calculate_distance(prev_pos, proj_pos)
                current_threat_distance = self._calculate_distance(player_pos, proj_pos)
                
                if current_threat_distance > prev_threat_distance:
                    reward += 1.0  # Good evasion
                else:
                    reward -= 0.5  # Failed to evade
        
        return reward
    
    def _calculate_resource_management_reward(
        self, 
        state: Dict[str, Any], 
        previous_state: Dict[str, Any]
    ) -> float:
        """Calculate reward for resource management."""
        player = state.get('player', {})
        prev_player = previous_state.get('player', {})
        
        reward = 0.0
        
        # Health management
        current_health = player.get('health', 100)
        prev_health = prev_player.get('health', 100)
        
        if current_health > prev_health:
            reward += 0.5  # Reward for healing/health gain
        
        # Ammo management
        current_ammo = player.get('ammunition', 10)
        prev_ammo = prev_player.get('ammunition', 10)
        
        if current_ammo > prev_ammo:
            reward += 0.3  # Reward for ammo pickup
        
        # Penalize low resources
        if current_health < 30:
            reward -= 0.3
        if current_ammo < 3:
            reward -= 0.2
        
        return reward
    
    def _calculate_exploration_bonus(self, state: Dict[str, Any]) -> float:
        """Calculate bonus for exploring new areas."""
        player = state.get('player', {})
        player_pos = player.get('position', {'x': 0, 'y': 0})
        
        # Discretize position for exploration tracking
        grid_x = int(player_pos['x'] / 50)  # 50-unit grid
        grid_y = int(player_pos['y'] / 50)
        grid_pos = (grid_x, grid_y)
        
        if grid_pos not in self.visited_positions:
            self.visited_positions.add(grid_pos)
            return 1.0  # Exploration bonus
        
        return 0.0
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def _calculate_flank_score(
        self, 
        player_pos: Dict[str, float], 
        enemy_pos: Dict[str, float], 
        enemy_vel: Dict[str, float]
    ) -> float:
        """Calculate flanking position score."""
        # Simple flanking detection based on relative positions
        dx = player_pos['x'] - enemy_pos['x']
        dy = player_pos['y'] - enemy_pos['y']
        
        # Enemy's facing direction (based on velocity)
        enemy_facing_x = enemy_vel['x']
        enemy_facing_y = enemy_vel['y']
        
        if enemy_facing_x == 0 and enemy_facing_y == 0:
            return 0.0  # Can't determine facing direction
        
        # Dot product to determine if player is behind enemy
        dot_product = dx * enemy_facing_x + dy * enemy_facing_y
        
        if dot_product < 0:
            return 0.5  # Player is behind enemy (good flanking position)
        else:
            return -0.2  # Player is in front of enemy
    
    def _is_projectile_threatening(
        self, 
        player_pos: Dict[str, float], 
        proj_pos: Dict[str, float], 
        proj_vel: Dict[str, float]
    ) -> bool:
        """Check if a projectile is threatening to the player."""
        # Vector from projectile to player
        dx = player_pos['x'] - proj_pos['x']
        dy = player_pos['y'] - proj_pos['y']
        
        # Dot product to check if projectile is moving towards player
        dot_product = dx * proj_vel['x'] + dy * proj_vel['y']
        
        return dot_product > 0 and self._calculate_distance(player_pos, proj_pos) < 200


class MultiObjectiveRewardFunction(RewardFunction):
    """
    Multi-objective reward function that combines multiple reward signals.
    
    Allows combining different reward functions with configurable weights
    and provides methods for multi-objective optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-objective reward function.
        
        Config options:
            - reward_functions: List of reward function configs
            - weights: Weights for each reward function
            - normalization: Normalization method ('none', 'z_score', 'min_max')
            - combination_method: How to combine rewards ('weighted_sum', 'pareto')
        """
        default_config = {
            'reward_functions': [
                {'type': 'sparse', 'weight': 0.4},
                {'type': 'dense', 'weight': 0.4},
                {'type': 'shaped', 'weight': 0.2}
            ],
            'normalization': 'z_score',
            'combination_method': 'weighted_sum'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Initialize component reward functions
        self.component_functions = []
        self.component_weights = []
        
        for rf_config in self.config['reward_functions']:
            rf_type = rf_config['type']
            rf_weight = rf_config.get('weight', 1.0)
            rf_params = rf_config.get('params', {})
            
            if rf_type == 'sparse':
                rf = SparseRewardFunction(rf_params)
            elif rf_type == 'dense':
                rf = DenseRewardFunction(rf_params)
            elif rf_type == 'shaped':
                rf = ShapedRewardFunction(rf_params)
            else:
                continue
            
            self.component_functions.append(rf)
            self.component_weights.append(rf_weight)
        
        # Normalization statistics
        self.reward_stats = {i: {'mean': 0.0, 'std': 1.0} for i in range(len(self.component_functions))}
    
    def _get_reward_type(self) -> RewardType:
        """Get reward type."""
        return RewardType.MULTI_OBJECTIVE
    
    def calculate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate multi-objective reward."""
        component_rewards = []
        
        # Calculate rewards from each component function
        for rf in self.component_functions:
            reward = rf.calculate_reward(state, action, previous_state, episode_stats)
            component_rewards.append(reward)
        
        # Apply normalization if configured
        if self.config['normalization'] != 'none':
            component_rewards = self._normalize_rewards(component_rewards)
        
        # Combine rewards
        if self.config['combination_method'] == 'weighted_sum':
            final_reward = sum(w * r for w, r in zip(self.component_weights, component_rewards))
        else:
            # For other combination methods, use weighted sum as default
            final_reward = sum(w * r for w, r in zip(self.component_weights, component_rewards))
        
        self.reward_history.append(final_reward)
        return final_reward
    
    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """Normalize component rewards."""
        if self.config['normalization'] == 'z_score':
            normalized = []
            for i, reward in enumerate(rewards):
                stats = self.reward_stats[i]
                normalized_reward = (reward - stats['mean']) / max(stats['std'], 1e-8)
                normalized.append(normalized_reward)
            return normalized
        elif self.config['normalization'] == 'min_max':
            # Simple min-max normalization (would need more sophisticated implementation)
            return rewards
        else:
            return rewards
    
    def update_normalization_stats(self) -> None:
        """Update normalization statistics based on reward history."""
        if not self.reward_history:
            return
        
        # This is a simplified version - in practice, you'd track component rewards separately
        for i in range(len(self.component_functions)):
            component_rewards = [rf.reward_history for rf in self.component_functions]
            if component_rewards[i]:
                rewards = np.array(component_rewards[i])
                self.reward_stats[i]['mean'] = float(np.mean(rewards))
                self.reward_stats[i]['std'] = float(np.std(rewards))
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each component reward function."""
        stats = {}
        for i, rf in enumerate(self.component_functions):
            stats[f'component_{i}'] = rf.get_episode_statistics()
        return stats


class HorizonBasedRewardFunction(RewardFunction):
    """
    Horizon-based reward function with different time scales.
    
    Provides rewards at different time horizons to balance
    short-term tactics with long-term strategy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize horizon-based reward function.
        
        Config options:
            - short_term_weight: Weight for short-term rewards (default: 0.5)
            - medium_term_weight: Weight for medium-term rewards (default: 0.3)
            - long_term_weight: Weight for long-term rewards (default: 0.2)
            - horizon_lengths: Step counts for each horizon (default: [5, 25, 100])
        """
        default_config = {
            'short_term_weight': 0.5,
            'medium_term_weight': 0.3,
            'long_term_weight': 0.2,
            'horizon_lengths': [5, 25, 100]
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        self.short_term_history = []
        self.medium_term_history = []
        self.long_term_history = []
        self.step_count = 0
    
    def _get_reward_type(self) -> RewardType:
        """Get reward type."""
        return RewardType.MULTI_OBJECTIVE  # Uses multiple time horizons
    
    def calculate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate horizon-based reward."""
        self.step_count += 1
        
        # Calculate immediate reward
        immediate_reward = self._calculate_immediate_reward(state, action, previous_state, episode_stats)
        
        # Store in history
        self.short_term_history.append(immediate_reward)
        self.medium_term_history.append(immediate_reward)
        self.long_term_history.append(immediate_reward)
        
        # Calculate horizon rewards
        short_term_reward = self._calculate_horizon_reward(
            self.short_term_history, self.config['horizon_lengths'][0]
        )
        medium_term_reward = self._calculate_horizon_reward(
            self.medium_term_history, self.config['horizon_lengths'][1]
        )
        long_term_reward = self._calculate_horizon_reward(
            self.long_term_history, self.config['horizon_lengths'][2]
        )
        
        # Combine with weights
        final_reward = (
            self.config['short_term_weight'] * short_term_reward +
            self.config['medium_term_weight'] * medium_term_reward +
            self.config['long_term_weight'] * long_term_reward
        )
        
        self.reward_history.append(final_reward)
        return final_reward
    
    def _calculate_immediate_reward(
        self, 
        state: Dict[str, Any], 
        action: Any, 
        previous_state: Dict[str, Any],
        episode_stats: Dict[str, Any]
    ) -> float:
        """Calculate immediate step reward."""
        # Use a combination of sparse and dense rewards
        reward = 0.0
        
        player = state.get('player', {})
        
        # Basic survival
        if player.get('is_alive', True):
            reward += 0.01
        else:
            reward -= 1.0
        
        # Damage events
        damage_dealt = episode_stats.get('damage_dealt', 0) - getattr(self, '_prev_damage_dealt', 0)
        damage_taken = episode_stats.get('damage_taken', 0) - getattr(self, '_prev_damage_taken', 0)
        
        reward += damage_dealt * 0.1
        reward -= damage_taken * 0.1
        
        self._prev_damage_dealt = episode_stats.get('damage_dealt', 0)
        self._prev_damage_taken = episode_stats.get('damage_taken', 0)
        
        return reward
    
    def _calculate_horizon_reward(self, history: List[float], horizon_length: int) -> float:
        """Calculate reward for a specific time horizon."""
        if len(history) < horizon_length:
            return 0.0
        
        # Use recent history for this horizon
        recent_rewards = history[-horizon_length:]
        
        # Different aggregation methods for different horizons
        if horizon_length <= 10:  # Short term - sum recent rewards
            return sum(recent_rewards)
        elif horizon_length <= 50:  # Medium term - average with trend
            avg_reward = np.mean(recent_rewards)
            trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5]) if len(recent_rewards) >= 10 else 0
            return avg_reward + 0.1 * trend
        else:  # Long term - focus on overall performance
            return np.mean(recent_rewards) * horizon_length * 0.01
    
    def reset_episode(self) -> None:
        """Reset for new episode."""
        super().reset_episode()
        self.short_term_history = []
        self.medium_term_history = []
        self.long_term_history = []
        self.step_count = 0


class RewardFunctionFactory:
    """Factory for creating reward functions."""
    
    @staticmethod
    def create_reward_function(
        reward_type: RewardType,
        config: Optional[Dict[str, Any]] = None
    ) -> RewardFunction:
        """
        Create a reward function of the specified type.
        
        Args:
            reward_type: Type of reward function
            config: Configuration for the reward function
            
        Returns:
            RewardFunction instance
        """
        if reward_type == RewardType.SPARSE:
            return SparseRewardFunction(config)
        elif reward_type == RewardType.DENSE:
            return DenseRewardFunction(config)
        elif reward_type == RewardType.SHAPED:
            return ShapedRewardFunction(config)
        elif reward_type == RewardType.MULTI_OBJECTIVE:
            return MultiObjectiveRewardFunction(config)
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
    
    @staticmethod
    def get_available_types() -> List[RewardType]:
        """Get list of available reward types."""
        return list(RewardType)


class RewardTuningFramework:
    """Framework for tuning and optimizing reward functions."""
    
    def __init__(self):
        self.tuning_results = {}
        self.current_experiments = {}
    
    def create_experiment(
        self,
        experiment_name: str,
        base_reward_type: RewardType,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_trials: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Create a reward tuning experiment.
        
        Args:
            experiment_name: Name of the experiment
            base_reward_type: Base reward function type
            parameter_ranges: Ranges for parameters to tune
            num_trials: Number of parameter combinations to try
            
        Returns:
            List of configurations to test
        """
        configs = []
        
        # Generate parameter combinations (simplified random sampling)
        for _ in range(num_trials):
            config = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                config[param] = np.random.uniform(min_val, max_val)
            configs.append(config)
        
        self.current_experiments[experiment_name] = {
            'base_type': base_reward_type,
            'configs': configs,
            'results': []
        }
        
        return configs
    
    def record_experiment_result(
        self,
        experiment_name: str,
        config_index: int,
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Record results for an experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            config_index: Index of the configuration
            performance_metrics: Performance metrics achieved
        """
        if experiment_name in self.current_experiments:
            experiment = self.current_experiments[experiment_name]
            if config_index < len(experiment['configs']):
                result = {
                    'config': experiment['configs'][config_index],
                    'metrics': performance_metrics
                }
                experiment['results'].append(result)
    
    def get_best_configuration(self, experiment_name: str, metric: str = 'total_reward') -> Optional[Dict[str, Any]]:
        """
        Get the best configuration from an experiment.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize for
            
        Returns:
            Best configuration or None if no results
        """
        if experiment_name not in self.current_experiments:
            return None
        
        experiment = self.current_experiments[experiment_name]
        results = experiment['results']
        
        if not results:
            return None
        
        best_result = max(results, key=lambda r: r['metrics'].get(metric, float('-inf')))
        return best_result['config']