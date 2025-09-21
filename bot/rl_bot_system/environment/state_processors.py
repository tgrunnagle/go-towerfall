"""
State processor implementations for different state representations.

This module provides flexible state representation systems with plugin architecture
for different ways of encoding game state for RL models.
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
import math


class StateRepresentationType(Enum):
    """Types of state representations available."""
    RAW_COORDINATE = "raw_coordinate"
    GRID_BASED = "grid_based"
    FEATURE_VECTOR = "feature_vector"
    HYBRID = "hybrid"


class StateProcessor(ABC):
    """Base class for state processors with plugin architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize state processor.
        
        Args:
            config: Configuration dictionary for the processor
        """
        self.config = config or {}
        self.representation_type = self._get_representation_type()
    
    @abstractmethod
    def process_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Process raw game state into model input."""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Get the observation space for this processor."""
        pass
    
    @abstractmethod
    def _get_representation_type(self) -> StateRepresentationType:
        """Get the representation type for this processor."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update processor configuration."""
        self.config.update(new_config)


class RawCoordinateProcessor(StateProcessor):
    """
    Raw coordinate state processor.
    
    Encodes game state as raw position and velocity coordinates with
    additional game information like health, ammunition, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize raw coordinate processor.
        
        Config options:
            - max_enemies: Maximum number of enemies to include (default: 5)
            - max_projectiles: Maximum number of projectiles to include (default: 10)
            - max_power_ups: Maximum number of power-ups to include (default: 3)
            - normalize_coordinates: Whether to normalize coordinates (default: True)
            - coordinate_bounds: Bounds for coordinate normalization (default: [-400, 400, -300, 300])
        """
        default_config = {
            'max_enemies': 5,
            'max_projectiles': 10,
            'max_power_ups': 3,
            'normalize_coordinates': True,
            'coordinate_bounds': [-400, 400, -300, 300]  # [left, right, top, bottom]
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Calculate feature dimensions
        self.player_features = 6  # x, y, vx, vy, health, ammo
        self.enemy_features = 5   # x, y, vx, vy, health
        self.projectile_features = 4  # x, y, vx, vy
        self.power_up_features = 3    # x, y, type
        self.game_features = 3        # game_time, round_active, boundaries_info
        
        self.total_features = (
            self.player_features +
            self.enemy_features * self.config['max_enemies'] +
            self.projectile_features * self.config['max_projectiles'] +
            self.power_up_features * self.config['max_power_ups'] +
            self.game_features
        )
    
    def _get_representation_type(self) -> StateRepresentationType:
        """Get representation type."""
        return StateRepresentationType.RAW_COORDINATE
    
    def process_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Process state into raw coordinates."""
        features = []
        
        # Player features
        player = raw_state.get('player', {})
        player_pos = player.get('position', {'x': 0, 'y': 0})
        player_vel = player.get('velocity', {'x': 0, 'y': 0})
        
        player_features = [
            player_pos['x'],
            player_pos['y'],
            player_vel['x'],
            player_vel['y'],
            player.get('health', 100) / 100.0,  # Normalize health
            player.get('ammunition', 10) / 10.0  # Normalize ammo
        ]
        
        if self.config['normalize_coordinates']:
            player_features = self._normalize_coordinates(player_features)
        
        features.extend(player_features)
        
        # Enemy features
        enemies = raw_state.get('enemies', [])
        for i in range(self.config['max_enemies']):
            if i < len(enemies):
                enemy = enemies[i]
                enemy_pos = enemy.get('position', {'x': 0, 'y': 0})
                enemy_vel = enemy.get('velocity', {'x': 0, 'y': 0})
                
                enemy_features = [
                    enemy_pos['x'],
                    enemy_pos['y'],
                    enemy_vel['x'],
                    enemy_vel['y'],
                    enemy.get('health', 100) / 100.0
                ]
                
                if self.config['normalize_coordinates']:
                    enemy_features = self._normalize_coordinates(enemy_features)
            else:
                # Pad with zeros if not enough enemies
                enemy_features = [0.0] * self.enemy_features
            
            features.extend(enemy_features)
        
        # Projectile features
        projectiles = raw_state.get('projectiles', [])
        for i in range(self.config['max_projectiles']):
            if i < len(projectiles):
                proj = projectiles[i]
                proj_pos = proj.get('position', {'x': 0, 'y': 0})
                proj_vel = proj.get('velocity', {'x': 0, 'y': 0})
                
                proj_features = [
                    proj_pos['x'],
                    proj_pos['y'],
                    proj_vel['x'],
                    proj_vel['y']
                ]
                
                if self.config['normalize_coordinates']:
                    proj_features = self._normalize_coordinates(proj_features)
            else:
                proj_features = [0.0] * self.projectile_features
            
            features.extend(proj_features)
        
        # Power-up features
        power_ups = raw_state.get('power_ups', [])
        for i in range(self.config['max_power_ups']):
            if i < len(power_ups):
                power_up = power_ups[i]
                power_up_pos = power_up.get('position', {'x': 0, 'y': 0})
                power_up_type = self._encode_power_up_type(power_up.get('type', 'health'))
                
                power_up_features = [
                    power_up_pos['x'],
                    power_up_pos['y'],
                    power_up_type
                ]
                
                if self.config['normalize_coordinates']:
                    power_up_features = self._normalize_coordinates(power_up_features)
            else:
                power_up_features = [0.0] * self.power_up_features
            
            features.extend(power_up_features)
        
        # Game features
        game_features = [
            raw_state.get('game_time', 0) / 300.0,  # Normalize assuming max 5 min games
            1.0 if raw_state.get('round_active', True) else 0.0,
            self._calculate_boundary_distance(player_pos, raw_state.get('boundaries', {}))
        ]
        
        features.extend(game_features)
        
        return np.array(features, dtype=np.float32)
    
    def get_observation_space(self) -> spaces.Space:
        """Get observation space."""
        if self.config['normalize_coordinates']:
            low = -2.0  # Allow some margin beyond normalized bounds
            high = 2.0
        else:
            bounds = self.config['coordinate_bounds']
            low = min(bounds) - 100  # Add margin
            high = max(bounds) + 100
        
        return spaces.Box(
            low=low, high=high,
            shape=(self.total_features,),
            dtype=np.float32
        )
    
    def _normalize_coordinates(self, coords: List[float]) -> List[float]:
        """Normalize coordinates to [-1, 1] range."""
        bounds = self.config['coordinate_bounds']
        left, right, top, bottom = bounds
        
        normalized = []
        for i, coord in enumerate(coords):
            if i % 2 == 0:  # x coordinates
                normalized.append(2 * (coord - left) / (right - left) - 1)
            else:  # y coordinates
                normalized.append(2 * (coord - top) / (bottom - top) - 1)
        
        return normalized
    
    def _encode_power_up_type(self, power_up_type: str) -> float:
        """Encode power-up type as numeric value."""
        type_mapping = {
            'health': 1.0,
            'ammo': 2.0,
            'speed': 3.0,
            'shield': 4.0
        }
        return type_mapping.get(power_up_type, 0.0)
    
    def _calculate_boundary_distance(self, position: Dict[str, float], boundaries: Dict[str, float]) -> float:
        """Calculate normalized distance to nearest boundary."""
        if not boundaries:
            return 0.0
        
        x, y = position.get('x', 0), position.get('y', 0)
        left = boundaries.get('left', -400)
        right = boundaries.get('right', 400)
        top = boundaries.get('top', -300)
        bottom = boundaries.get('bottom', 300)
        
        # Distance to each boundary
        dist_left = abs(x - left)
        dist_right = abs(x - right)
        dist_top = abs(y - top)
        dist_bottom = abs(y - bottom)
        
        min_distance = min(dist_left, dist_right, dist_top, dist_bottom)
        max_possible_distance = max(right - left, bottom - top) / 2
        
        return min_distance / max_possible_distance


class GridBasedProcessor(StateProcessor):
    """
    Grid-based state processor.
    
    Converts the game world into a discretized grid where each cell
    contains information about entities present in that location.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize grid-based processor.
        
        Config options:
            - grid_width: Width of the grid (default: 32)
            - grid_height: Height of the grid (default: 24)
            - world_bounds: World coordinate bounds (default: [-400, 400, -300, 300])
            - channels: List of channels to include (default: ['player', 'enemies', 'projectiles', 'power_ups'])
            - multi_channel: Whether to use multi-channel representation (default: True)
        """
        default_config = {
            'grid_width': 32,
            'grid_height': 24,
            'world_bounds': [-400, 400, -300, 300],
            'channels': ['player', 'enemies', 'projectiles', 'power_ups', 'boundaries'],
            'multi_channel': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        self.grid_width = self.config['grid_width']
        self.grid_height = self.config['grid_height']
        self.world_bounds = self.config['world_bounds']
        self.channels = self.config['channels']
        self.num_channels = len(self.channels) if self.config['multi_channel'] else 1
        
        # Calculate grid cell size
        world_width = self.world_bounds[1] - self.world_bounds[0]
        world_height = self.world_bounds[3] - self.world_bounds[2]
        self.cell_width = world_width / self.grid_width
        self.cell_height = world_height / self.grid_height
    
    def _get_representation_type(self) -> StateRepresentationType:
        """Get representation type."""
        return StateRepresentationType.GRID_BASED
    
    def process_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Process state into grid representation."""
        if self.config['multi_channel']:
            grid = np.zeros((self.num_channels, self.grid_height, self.grid_width), dtype=np.float32)
        else:
            grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        channel_idx = 0
        
        # Player channel
        if 'player' in self.channels:
            player = raw_state.get('player', {})
            player_pos = player.get('position', {'x': 0, 'y': 0})
            grid_x, grid_y = self._world_to_grid(player_pos['x'], player_pos['y'])
            
            if self._is_valid_grid_pos(grid_x, grid_y):
                if self.config['multi_channel']:
                    grid[channel_idx, grid_y, grid_x] = player.get('health', 100) / 100.0
                else:
                    grid[grid_y, grid_x] = 1.0  # Player presence
            
            if self.config['multi_channel']:
                channel_idx += 1
        
        # Enemies channel
        if 'enemies' in self.channels:
            enemies = raw_state.get('enemies', [])
            for enemy in enemies:
                enemy_pos = enemy.get('position', {'x': 0, 'y': 0})
                grid_x, grid_y = self._world_to_grid(enemy_pos['x'], enemy_pos['y'])
                
                if self._is_valid_grid_pos(grid_x, grid_y):
                    enemy_value = enemy.get('health', 100) / 100.0
                    if self.config['multi_channel']:
                        grid[channel_idx, grid_y, grid_x] = max(
                            grid[channel_idx, grid_y, grid_x], enemy_value
                        )
                    else:
                        grid[grid_y, grid_x] = max(grid[grid_y, grid_x], 0.5)
            
            if self.config['multi_channel']:
                channel_idx += 1
        
        # Projectiles channel
        if 'projectiles' in self.channels:
            projectiles = raw_state.get('projectiles', [])
            for proj in projectiles:
                proj_pos = proj.get('position', {'x': 0, 'y': 0})
                grid_x, grid_y = self._world_to_grid(proj_pos['x'], proj_pos['y'])
                
                if self._is_valid_grid_pos(grid_x, grid_y):
                    if self.config['multi_channel']:
                        grid[channel_idx, grid_y, grid_x] = 1.0
                    else:
                        grid[grid_y, grid_x] = max(grid[grid_y, grid_x], 0.3)
            
            if self.config['multi_channel']:
                channel_idx += 1
        
        # Power-ups channel
        if 'power_ups' in self.channels:
            power_ups = raw_state.get('power_ups', [])
            for power_up in power_ups:
                power_up_pos = power_up.get('position', {'x': 0, 'y': 0})
                grid_x, grid_y = self._world_to_grid(power_up_pos['x'], power_up_pos['y'])
                
                if self._is_valid_grid_pos(grid_x, grid_y):
                    power_up_value = self._encode_power_up_type(power_up.get('type', 'health')) / 4.0
                    if self.config['multi_channel']:
                        grid[channel_idx, grid_y, grid_x] = power_up_value
                    else:
                        grid[grid_y, grid_x] = max(grid[grid_y, grid_x], 0.2)
            
            if self.config['multi_channel']:
                channel_idx += 1
        
        # Boundaries channel
        if 'boundaries' in self.channels:
            boundaries = raw_state.get('boundaries', {})
            if boundaries:
                self._add_boundaries_to_grid(grid, channel_idx if self.config['multi_channel'] else None)
            
            if self.config['multi_channel']:
                channel_idx += 1
        
        return grid
    
    def get_observation_space(self) -> spaces.Space:
        """Get observation space."""
        if self.config['multi_channel']:
            shape = (self.num_channels, self.grid_height, self.grid_width)
        else:
            shape = (self.grid_height, self.grid_width)
        
        return spaces.Box(
            low=0.0, high=1.0,
            shape=shape,
            dtype=np.float32
        )
    
    def _world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        # Normalize to [0, 1]
        norm_x = (world_x - self.world_bounds[0]) / (self.world_bounds[1] - self.world_bounds[0])
        norm_y = (world_y - self.world_bounds[2]) / (self.world_bounds[3] - self.world_bounds[2])
        
        # Convert to grid coordinates
        grid_x = int(norm_x * self.grid_width)
        grid_y = int(norm_y * self.grid_height)
        
        # Clamp to valid range
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))
        
        return grid_x, grid_y
    
    def _is_valid_grid_pos(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid position is valid."""
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height
    
    def _encode_power_up_type(self, power_up_type: str) -> float:
        """Encode power-up type as numeric value."""
        type_mapping = {
            'health': 1.0,
            'ammo': 2.0,
            'speed': 3.0,
            'shield': 4.0
        }
        return type_mapping.get(power_up_type, 1.0)
    
    def _add_boundaries_to_grid(self, grid: np.ndarray, channel_idx: Optional[int] = None) -> None:
        """Add boundary information to grid."""
        if channel_idx is not None:
            # Multi-channel: mark boundary cells
            grid[channel_idx, 0, :] = 1.0  # Top boundary
            grid[channel_idx, -1, :] = 1.0  # Bottom boundary
            grid[channel_idx, :, 0] = 1.0  # Left boundary
            grid[channel_idx, :, -1] = 1.0  # Right boundary
        else:
            # Single channel: add boundary markers
            grid[0, :] = np.maximum(grid[0, :], 0.1)
            grid[-1, :] = np.maximum(grid[-1, :], 0.1)
            grid[:, 0] = np.maximum(grid[:, 0], 0.1)
            grid[:, -1] = np.maximum(grid[:, -1], 0.1)


class FeatureVectorProcessor(StateProcessor):
    """
    Feature vector state processor.
    
    Extracts tactical and strategic features from the game state,
    focusing on high-level information rather than raw coordinates.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature vector processor.
        
        Config options:
            - include_distances: Include distance features (default: True)
            - include_angles: Include angle features (default: True)
            - include_tactical: Include tactical features (default: True)
            - include_health_ratios: Include health ratio features (default: True)
            - max_enemies_for_features: Max enemies to consider for features (default: 3)
        """
        default_config = {
            'include_distances': True,
            'include_angles': True,
            'include_tactical': True,
            'include_health_ratios': True,
            'max_enemies_for_features': 3
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Calculate feature dimensions
        self._calculate_feature_dimensions()
    
    def _get_representation_type(self) -> StateRepresentationType:
        """Get representation type."""
        return StateRepresentationType.FEATURE_VECTOR
    
    def process_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Process state into feature vector."""
        features = []
        
        player = raw_state.get('player', {})
        enemies = raw_state.get('enemies', [])
        projectiles = raw_state.get('projectiles', [])
        power_ups = raw_state.get('power_ups', [])
        boundaries = raw_state.get('boundaries', {})
        
        player_pos = player.get('position', {'x': 0, 'y': 0})
        player_vel = player.get('velocity', {'x': 0, 'y': 0})
        
        # Basic player features
        features.extend([
            player.get('health', 100) / 100.0,
            player.get('ammunition', 10) / 10.0,
            np.linalg.norm([player_vel['x'], player_vel['y']]) / 50.0,  # Normalized speed
        ])
        
        # Distance features
        if self.config['include_distances']:
            # Distance to nearest enemy
            if enemies:
                enemy_distances = [
                    self._calculate_distance(player_pos, enemy.get('position', {'x': 0, 'y': 0}))
                    for enemy in enemies
                ]
                features.append(min(enemy_distances) / 800.0)  # Normalize by max possible distance
                features.append(np.mean(enemy_distances) / 800.0)
            else:
                features.extend([1.0, 1.0])  # Max distance if no enemies
            
            # Distance to nearest projectile
            if projectiles:
                proj_distances = [
                    self._calculate_distance(player_pos, proj.get('position', {'x': 0, 'y': 0}))
                    for proj in projectiles
                ]
                features.append(min(proj_distances) / 800.0)
            else:
                features.append(1.0)
            
            # Distance to nearest power-up
            if power_ups:
                power_up_distances = [
                    self._calculate_distance(player_pos, pu.get('position', {'x': 0, 'y': 0}))
                    for pu in power_ups
                ]
                features.append(min(power_up_distances) / 800.0)
            else:
                features.append(1.0)
            
            # Distance to boundaries
            boundary_distance = self._calculate_boundary_distance(player_pos, boundaries)
            features.append(boundary_distance)
        
        # Angle features
        if self.config['include_angles']:
            # Angle to nearest enemy
            if enemies:
                nearest_enemy = min(enemies, key=lambda e: self._calculate_distance(
                    player_pos, e.get('position', {'x': 0, 'y': 0})
                ))
                enemy_angle = self._calculate_angle(player_pos, nearest_enemy.get('position', {'x': 0, 'y': 0}))
                features.extend([np.cos(enemy_angle), np.sin(enemy_angle)])
            else:
                features.extend([0.0, 0.0])
            
            # Angle to nearest projectile
            if projectiles:
                nearest_proj = min(projectiles, key=lambda p: self._calculate_distance(
                    player_pos, p.get('position', {'x': 0, 'y': 0})
                ))
                proj_angle = self._calculate_angle(player_pos, nearest_proj.get('position', {'x': 0, 'y': 0}))
                features.extend([np.cos(proj_angle), np.sin(proj_angle)])
            else:
                features.extend([0.0, 0.0])
        
        # Tactical features
        if self.config['include_tactical']:
            # Number of enemies in different ranges
            close_enemies = sum(1 for e in enemies if self._calculate_distance(
                player_pos, e.get('position', {'x': 0, 'y': 0})) < 100)
            medium_enemies = sum(1 for e in enemies if 100 <= self._calculate_distance(
                player_pos, e.get('position', {'x': 0, 'y': 0})) < 200)
            far_enemies = sum(1 for e in enemies if self._calculate_distance(
                player_pos, e.get('position', {'x': 0, 'y': 0})) >= 200)
            
            features.extend([
                close_enemies / 5.0,  # Normalize by max possible
                medium_enemies / 5.0,
                far_enemies / 5.0
            ])
            
            # Projectile threat level
            incoming_projectiles = sum(1 for p in projectiles if self._is_projectile_threatening(
                player_pos, p))
            features.append(incoming_projectiles / 10.0)
            
            # Cover availability (simplified)
            cover_score = self._calculate_cover_score(player_pos, boundaries)
            features.append(cover_score)
        
        # Health ratios
        if self.config['include_health_ratios']:
            if enemies:
                enemy_healths = [e.get('health', 100) for e in enemies]
                avg_enemy_health = np.mean(enemy_healths) / 100.0
                min_enemy_health = min(enemy_healths) / 100.0
                features.extend([avg_enemy_health, min_enemy_health])
            else:
                features.extend([0.0, 0.0])
        
        # Enemy-specific features (for top N enemies)
        max_enemies = min(len(enemies), self.config['max_enemies_for_features'])
        sorted_enemies = sorted(enemies, key=lambda e: self._calculate_distance(
            player_pos, e.get('position', {'x': 0, 'y': 0})))
        
        for i in range(self.config['max_enemies_for_features']):
            if i < max_enemies:
                enemy = sorted_enemies[i]
                enemy_pos = enemy.get('position', {'x': 0, 'y': 0})
                enemy_vel = enemy.get('velocity', {'x': 0, 'y': 0})
                
                # Relative position (normalized)
                rel_x = (enemy_pos['x'] - player_pos['x']) / 400.0
                rel_y = (enemy_pos['y'] - player_pos['y']) / 300.0
                
                # Relative velocity
                rel_vx = (enemy_vel['x'] - player_vel['x']) / 50.0
                rel_vy = (enemy_vel['y'] - player_vel['y']) / 50.0
                
                # Enemy health
                enemy_health = enemy.get('health', 100) / 100.0
                
                features.extend([rel_x, rel_y, rel_vx, rel_vy, enemy_health])
            else:
                # Pad with zeros
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def get_observation_space(self) -> spaces.Space:
        """Get observation space."""
        return spaces.Box(
            low=-2.0, high=2.0,
            shape=(self.total_features,),
            dtype=np.float32
        )
    
    def _calculate_feature_dimensions(self) -> None:
        """Calculate total number of features."""
        self.total_features = 3  # Basic player features
        
        if self.config['include_distances']:
            self.total_features += 5  # Enemy, projectile, power-up, boundary distances
        
        if self.config['include_angles']:
            self.total_features += 4  # Enemy and projectile angles (cos, sin)
        
        if self.config['include_tactical']:
            self.total_features += 5  # Range counts, projectile threat, cover
        
        if self.config['include_health_ratios']:
            self.total_features += 2  # Average and min enemy health
        
        # Enemy-specific features
        self.total_features += self.config['max_enemies_for_features'] * 5
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def _calculate_angle(self, from_pos: Dict[str, float], to_pos: Dict[str, float]) -> float:
        """Calculate angle from one position to another."""
        dx = to_pos['x'] - from_pos['x']
        dy = to_pos['y'] - from_pos['y']
        return math.atan2(dy, dx)
    
    def _calculate_boundary_distance(self, position: Dict[str, float], boundaries: Dict[str, float]) -> float:
        """Calculate normalized distance to nearest boundary."""
        if not boundaries:
            return 0.5
        
        x, y = position['x'], position['y']
        left = boundaries.get('left', -400)
        right = boundaries.get('right', 400)
        top = boundaries.get('top', -300)
        bottom = boundaries.get('bottom', 300)
        
        distances = [
            abs(x - left),
            abs(x - right),
            abs(y - top),
            abs(y - bottom)
        ]
        
        min_distance = min(distances)
        max_possible = max(right - left, bottom - top) / 2
        
        return min_distance / max_possible
    
    def _is_projectile_threatening(self, player_pos: Dict[str, float], projectile: Dict[str, Any]) -> bool:
        """Check if a projectile is threatening to the player."""
        proj_pos = projectile.get('position', {'x': 0, 'y': 0})
        proj_vel = projectile.get('velocity', {'x': 0, 'y': 0})
        
        # Simple threat detection: projectile moving towards player
        dx = player_pos['x'] - proj_pos['x']
        dy = player_pos['y'] - proj_pos['y']
        
        # Dot product to check if velocity is towards player
        dot_product = dx * proj_vel['x'] + dy * proj_vel['y']
        
        return dot_product > 0 and self._calculate_distance(player_pos, proj_pos) < 150
    
    def _calculate_cover_score(self, position: Dict[str, float], boundaries: Dict[str, float]) -> float:
        """Calculate a simple cover score based on position."""
        if not boundaries:
            return 0.5
        
        # Simple heuristic: closer to boundaries = more cover
        boundary_distance = self._calculate_boundary_distance(position, boundaries)
        return 1.0 - boundary_distance


class StateProcessorFactory:
    """Factory for creating state processors with A/B testing support."""
    
    @staticmethod
    def create_processor(
        representation_type: StateRepresentationType,
        config: Optional[Dict[str, Any]] = None
    ) -> StateProcessor:
        """
        Create a state processor of the specified type.
        
        Args:
            representation_type: Type of state representation
            config: Configuration for the processor
            
        Returns:
            StateProcessor instance
        """
        if representation_type == StateRepresentationType.RAW_COORDINATE:
            return RawCoordinateProcessor(config)
        elif representation_type == StateRepresentationType.GRID_BASED:
            return GridBasedProcessor(config)
        elif representation_type == StateRepresentationType.FEATURE_VECTOR:
            return FeatureVectorProcessor(config)
        else:
            raise ValueError(f"Unsupported representation type: {representation_type}")
    
    @staticmethod
    def get_available_types() -> List[StateRepresentationType]:
        """Get list of available representation types."""
        return list(StateRepresentationType)


class ABTestingFramework:
    """Framework for A/B testing different state representations."""
    
    def __init__(self):
        self.test_results = {}
        self.current_tests = {}
    
    def create_test(
        self,
        test_name: str,
        representation_a: StateRepresentationType,
        representation_b: StateRepresentationType,
        config_a: Optional[Dict[str, Any]] = None,
        config_b: Optional[Dict[str, Any]] = None
    ) -> Dict[str, StateProcessor]:
        """
        Create an A/B test between two state representations.
        
        Args:
            test_name: Name of the test
            representation_a: First representation type
            representation_b: Second representation type
            config_a: Configuration for first representation
            config_b: Configuration for second representation
            
        Returns:
            Dictionary with 'A' and 'B' processors
        """
        processor_a = StateProcessorFactory.create_processor(representation_a, config_a)
        processor_b = StateProcessorFactory.create_processor(representation_b, config_b)
        
        test_setup = {
            'A': processor_a,
            'B': processor_b,
            'representation_a': representation_a,
            'representation_b': representation_b,
            'config_a': config_a or {},
            'config_b': config_b or {}
        }
        
        self.current_tests[test_name] = test_setup
        
        return {'A': processor_a, 'B': processor_b}
    
    def record_result(
        self,
        test_name: str,
        variant: str,
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Record performance results for a test variant.
        
        Args:
            test_name: Name of the test
            variant: 'A' or 'B'
            performance_metrics: Performance metrics to record
        """
        if test_name not in self.test_results:
            self.test_results[test_name] = {'A': [], 'B': []}
        
        self.test_results[test_name][variant].append(performance_metrics)
    
    def get_test_summary(self, test_name: str) -> Dict[str, Any]:
        """
        Get summary of test results.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test summary with statistics
        """
        if test_name not in self.test_results:
            return {}
        
        results = self.test_results[test_name]
        summary = {}
        
        for variant in ['A', 'B']:
            if results[variant]:
                metrics = results[variant]
                summary[variant] = {
                    'count': len(metrics),
                    'avg_reward': np.mean([m.get('total_reward', 0) for m in metrics]),
                    'avg_win_rate': np.mean([m.get('win_rate', 0) for m in metrics]),
                    'avg_episode_length': np.mean([m.get('episode_length', 0) for m in metrics])
                }
        
        return summary