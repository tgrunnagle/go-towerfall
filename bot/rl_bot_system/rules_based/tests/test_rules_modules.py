"""
Unit tests for the rules modules (combat, survival, strategic).
"""

import unittest
from unittest.mock import Mock, patch
import math

# Path setup handled by test runner via PYTHONPATH

from rl_bot_system.rules_based.combat_rules import CombatRules
from rl_bot_system.rules_based.survival_rules import SurvivalRules
from rl_bot_system.rules_based.strategic_rules import StrategicRules
from rl_bot_system.rules_based.rules_based_bot import (
    DifficultyLevel,
    ThreatType,
    OpportunityType,
    Threat,
    Opportunity,
    GameStateAnalysis,
    ActionPriority
)


class TestSurvivalRules(unittest.TestCase):
    """Test cases for the SurvivalRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'accuracy_modifier': 0.8,
            'reaction_time': 0.1
        }
        self.survival_rules = SurvivalRules(self.config)
        
    def test_survival_rules_initialization(self):
        """Test that survival rules initialize correctly"""
        self.assertEqual(self.survival_rules.config, self.config)
        
    def test_health_status_assessment(self):
        """Test health status assessment"""
        # Test critical health
        self.assertEqual(self.survival_rules._assess_health_status(20), 'critical')
        
        # Test low health
        self.assertEqual(self.survival_rules._assess_health_status(40), 'low')
        
        # Test medium health
        self.assertEqual(self.survival_rules._assess_health_status(60), 'medium')
        
        # Test high health
        self.assertEqual(self.survival_rules._assess_health_status(90), 'high')
        
    def test_boundary_safety_assessment(self):
        """Test boundary safety assessment"""
        boundaries = {
            'left': 0,
            'right': 800,
            'top': 0,
            'bottom': 600
        }
        
        # Safe position in center
        self.assertTrue(self.survival_rules._assess_boundary_safety((400, 300), boundaries))
        
        # Unsafe position near left boundary
        self.assertFalse(self.survival_rules._assess_boundary_safety((30, 300), boundaries))
        
        # Unsafe position near top boundary
        self.assertFalse(self.survival_rules._assess_boundary_safety((400, 30), boundaries))
        
    def test_immediate_danger_identification(self):
        """Test identification of immediate dangers"""
        threats = [
            Threat(ThreatType.PROJECTILE, (100, 100), (10, 0), 0.9, 0.3, 'proj1'),  # Immediate
            Threat(ThreatType.PROJECTILE, (200, 200), (5, 0), 0.5, 1.0, 'proj2'),   # Not immediate
            Threat(ThreatType.ENEMY, (150, 150), (0, 0), 0.9, None, 'enemy1'),      # Immediate
            Threat(ThreatType.BOUNDARY, (50, 50), None, 0.8, None, 'boundary')      # Immediate
        ]
        
        immediate_dangers = self.survival_rules._identify_immediate_dangers(threats)
        
        # Should identify 3 immediate dangers
        self.assertEqual(len(immediate_dangers), 3)
        
        # Check that the right threats are identified
        threat_ids = [t.source_id for t in immediate_dangers]
        self.assertIn('proj1', threat_ids)
        self.assertIn('enemy1', threat_ids)
        self.assertIn('boundary', threat_ids)
        self.assertNotIn('proj2', threat_ids)
        
    def test_survival_priority_calculation(self):
        """Test survival priority calculation"""
        # Create analysis with high threat situation
        high_threat_analysis = GameStateAnalysis(
            threats=[Threat(ThreatType.PROJECTILE, (100, 100), (10, 0), 0.9, 0.3, 'proj1')],
            opportunities=[],
            player_position=(120, 120),
            player_velocity=(0, 0),
            player_health=25,  # Low health
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        priority = self.survival_rules._calculate_survival_priority(high_threat_analysis)
        self.assertGreater(priority, 0.6)  # Should be high priority
        
        # Create analysis with safe situation
        safe_analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(400, 300),
            player_velocity=(0, 0),
            player_health=100,  # Full health
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        priority = self.survival_rules._calculate_survival_priority(safe_analysis)
        self.assertLess(priority, 0.3)  # Should be low priority
        
    def test_projectile_dodge_actions(self):
        """Test projectile dodge action generation"""
        projectile_threat = Threat(
            ThreatType.PROJECTILE, 
            (100, 100), 
            (10, 0),  # Moving right
            0.9, 
            0.3, 
            'proj1'
        )
        
        analysis = GameStateAnalysis(
            threats=[projectile_threat],
            opportunities=[],
            player_position=(120, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        actions = self.survival_rules._generate_projectile_dodge_actions(projectile_threat, analysis)
        
        # Should generate dodge action
        self.assertGreater(len(actions), 0)
        
        # Action should be critical priority
        self.assertEqual(actions[0].priority, ActionPriority.CRITICAL)
        
        # Should be a movement key
        self.assertIn(actions[0].parameters['key'], ['w', 'a', 's', 'd'])
        
    def test_boundary_avoidance_actions(self):
        """Test boundary avoidance action generation"""
        boundary_threat = Threat(
            ThreatType.BOUNDARY,
            (30, 300),  # Near left boundary
            None,
            0.8,
            None,
            'left'
        )
        
        analysis = GameStateAnalysis(
            threats=[boundary_threat],
            opportunities=[],
            player_position=(30, 300),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        actions = self.survival_rules._generate_boundary_avoidance_actions(boundary_threat, analysis)
        
        # Should generate avoidance action
        self.assertGreater(len(actions), 0)
        
        # Should move away from left boundary (move right)
        self.assertEqual(actions[0].parameters['key'], 'd')
        
        # Should be critical priority
        self.assertEqual(actions[0].priority, ActionPriority.CRITICAL)


class TestCombatRules(unittest.TestCase):
    """Test cases for the CombatRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'accuracy_modifier': 0.8,
            'aggression_level': 0.6,
            'reaction_time': 0.1
        }
        self.combat_rules = CombatRules(self.config)
        
    def test_combat_rules_initialization(self):
        """Test that combat rules initialize correctly"""
        self.assertEqual(self.combat_rules.config, self.config)
        
    def test_threat_level_calculation(self):
        """Test threat level calculation"""
        # High threat scenario
        high_threats = [
            Threat(ThreatType.PROJECTILE, (100, 100), (10, 0), 0.9, 0.3, 'proj1'),
            Threat(ThreatType.ENEMY, (150, 150), (0, 0), 0.8, None, 'enemy1')
        ]
        
        threat_level = self.combat_rules._calculate_threat_level(high_threats)
        self.assertGreater(threat_level, 0.8)
        
        # Low threat scenario
        low_threats = [
            Threat(ThreatType.BOUNDARY, (50, 50), None, 0.3, None, 'boundary')
        ]
        
        threat_level = self.combat_rules._calculate_threat_level(low_threats)
        self.assertLess(threat_level, 0.3)
        
        # No threats
        threat_level = self.combat_rules._calculate_threat_level([])
        self.assertEqual(threat_level, 0.0)
        
    def test_target_prioritization(self):
        """Test enemy target prioritization"""
        enemies = [
            {
                'id': 'enemy1',
                'position': (200, 200),  # Far enemy
                'velocity': (0, 0),
                'health': 100,
                'behind_cover': False,
                'line_of_sight': True
            },
            {
                'id': 'enemy2', 
                'position': (120, 120),  # Close enemy
                'velocity': (0, 0),
                'health': 30,  # Low health
                'behind_cover': False,
                'line_of_sight': True
            },
            {
                'id': 'enemy3',
                'position': (150, 150),  # Medium distance
                'velocity': (10, 10),  # Moving
                'health': 100,
                'behind_cover': True,  # Behind cover
                'line_of_sight': False
            }
        ]
        
        player_pos = (100, 100)
        prioritized = self.combat_rules._prioritize_targets(enemies, player_pos)
        
        # Should prioritize close, low-health enemy first
        self.assertEqual(prioritized[0]['id'], 'enemy2')
        
        # Enemy behind cover should be lower priority
        self.assertEqual(prioritized[-1]['id'], 'enemy3')
        
    def test_combat_stance_determination(self):
        """Test combat stance determination"""
        # Healthy player, few enemies -> aggressive
        healthy_analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=90,
            enemies=[{'id': 'enemy1', 'position': (150, 150)}],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        stance = self.combat_rules._determine_combat_stance(healthy_analysis)
        self.assertEqual(stance, 'aggressive')
        
        # Low health, high threat -> retreat
        danger_analysis = GameStateAnalysis(
            threats=[Threat(ThreatType.PROJECTILE, (100, 100), (10, 0), 0.9, 0.3, 'proj1')],
            opportunities=[],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=20,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        stance = self.combat_rules._determine_combat_stance(danger_analysis)
        self.assertEqual(stance, 'retreat')
        
    def test_aim_point_calculation(self):
        """Test aim point calculation with prediction"""
        target = {
            'position': (150, 150),
            'velocity': (10, 0)  # Moving right
        }
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        aim_point = self.combat_rules._calculate_aim_point(target, analysis)
        
        # Should predict ahead of target's current position (allowing for accuracy modifier)
        # The aim point should be reasonably close to the predicted position
        self.assertIsInstance(aim_point, tuple)
        self.assertEqual(len(aim_point), 2)
        
        # Check that prediction is reasonable (within accuracy bounds)
        distance_from_target = abs(aim_point[0] - target['position'][0])
        self.assertLess(distance_from_target, 50)  # Should be within reasonable range
        
    def test_shot_confidence_calculation(self):
        """Test shot confidence calculation"""
        # Close, stationary target
        close_target = {
            'position': (120, 120),
            'velocity': (0, 0),
            'line_of_sight': True
        }
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        confidence = self.combat_rules._calculate_shot_confidence(close_target, analysis)
        self.assertGreater(confidence, 0.7)
        
        # Far, moving target with no line of sight
        difficult_target = {
            'position': (400, 400),
            'velocity': (20, 20),
            'line_of_sight': False
        }
        
        confidence = self.combat_rules._calculate_shot_confidence(difficult_target, analysis)
        self.assertLess(confidence, 0.3)
        
    def test_targeting_actions(self):
        """Test targeting action generation"""
        target = {
            'id': 'enemy1',
            'position': (150, 150),
            'velocity': (0, 0)
        }
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[target],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        actions = self.combat_rules._generate_targeting_actions(target, analysis)
        
        # Should generate shooting action
        self.assertGreater(len(actions), 0)
        
        # Should be a shooting action
        self.assertEqual(actions[0].action_type, 'shoot_at_target')
        self.assertEqual(actions[0].parameters['button'], 'left')
        self.assertEqual(actions[0].parameters['pressed'], True)
        
        # Should have target coordinates
        self.assertIn('x', actions[0].parameters)
        self.assertIn('y', actions[0].parameters)


class TestStrategicRules(unittest.TestCase):
    """Test cases for the StrategicRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategic_depth': 3,
            'aggression_level': 0.6
        }
        self.strategic_rules = StrategicRules(self.config)
        
    def test_strategic_rules_initialization(self):
        """Test that strategic rules initialize correctly"""
        self.assertEqual(self.strategic_rules.config, self.config)
        
    def test_map_control_assessment(self):
        """Test map control assessment"""
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(400, 300),  # Center position
            player_velocity=(0, 0),
            player_health=100,
            enemies=[
                {'id': 'enemy1', 'position': (100, 100)},
                {'id': 'enemy2', 'position': (700, 500)}
            ],
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        map_control = self.strategic_rules._assess_map_control(analysis)
        
        # Should return control assessment
        self.assertIn('control_percentage', map_control)
        self.assertIn('controlled_areas', map_control)
        self.assertIn('contested_areas', map_control)
        
        # Control percentage should be between 0 and 1
        self.assertGreaterEqual(map_control['control_percentage'], 0.0)
        self.assertLessEqual(map_control['control_percentage'], 1.0)
        
    def test_power_up_control_assessment(self):
        """Test power-up control assessment"""
        power_up_opportunities = [
            Opportunity(OpportunityType.POWER_UP, (120, 120), 0.8, 30, 0.7),  # Close to player
            Opportunity(OpportunityType.POWER_UP, (700, 500), 0.6, 200, 0.5)  # Far from player
        ]
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=power_up_opportunities,
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[
                {'id': 'enemy1', 'position': (600, 400)}  # Closer to far power-up
            ],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        power_up_control = self.strategic_rules._assess_power_up_control(analysis)
        
        # Should assess power-up control
        self.assertIn('control_ratio', power_up_control)
        self.assertIn('controlled_count', power_up_control)
        self.assertIn('contested_count', power_up_control)
        self.assertIn('total_count', power_up_control)
        
        # Should control at least one power-up (the close one)
        self.assertGreater(power_up_control['controlled_count'], 0)
        
    def test_positional_advantage_assessment(self):
        """Test positional advantage assessment"""
        # Center position should have good advantage
        center_analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(400, 300),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[{'id': 'enemy1', 'position': (100, 500)}],  # Lower position
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[(350, 250)]  # Close safe zone
        )
        
        advantage = self.strategic_rules._assess_positional_advantage(center_analysis)
        self.assertGreater(advantage, 0.5)
        
        # Corner position should have lower advantage
        corner_analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(50, 50),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[{'id': 'enemy1', 'position': (400, 300)}],  # Better position
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        advantage = self.strategic_rules._assess_positional_advantage(corner_analysis)
        self.assertLess(advantage, 0.75)
        
    def test_strategic_objective_identification(self):
        """Test strategic objective identification"""
        high_value_power_up = Opportunity(OpportunityType.POWER_UP, (200, 200), 0.9, 100, 0.8)
        strategic_position = Opportunity(OpportunityType.STRATEGIC_POSITION, (300, 150), 0.7, 150, 0.6)
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[high_value_power_up, strategic_position],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[
                {'id': 'enemy1', 'position': (600, 600)}  # Isolated enemy
            ],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        objectives = self.strategic_rules._identify_strategic_objectives(analysis)
        
        # Should identify objectives
        self.assertGreater(len(objectives), 0)
        
        # Should be sorted by priority
        for i in range(len(objectives) - 1):
            self.assertGreaterEqual(objectives[i]['priority'], objectives[i + 1]['priority'])
        
        # Should include different objective types
        objective_types = [obj['type'] for obj in objectives]
        self.assertIn('control_power_up', objective_types)
        
    def test_isolated_enemy_detection(self):
        """Test isolated enemy detection"""
        enemies = [
            {'id': 'enemy1', 'position': (100, 100)},  # Isolated
            {'id': 'enemy2', 'position': (500, 500)},  # Close to enemy3
            {'id': 'enemy3', 'position': (520, 480)}   # Close to enemy2
        ]
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(300, 300),
            player_velocity=(0, 0),
            player_health=100,
            enemies=enemies,
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        isolated = self.strategic_rules._find_isolated_enemies(analysis)
        
        # Should find the isolated enemy
        self.assertEqual(len(isolated), 1)
        self.assertEqual(isolated[0]['id'], 'enemy1')
        
    def test_territory_control_actions(self):
        """Test territory control action generation"""
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[],
            player_position=(100, 100),  # Off-center
            player_velocity=(0, 0),
            player_health=100,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        actions = self.strategic_rules._generate_territory_control_actions(analysis)
        
        # Should generate movement towards center
        self.assertGreater(len(actions), 0)
        
        # Should be a movement action
        self.assertEqual(actions[0].action_type, 'control_territory')
        self.assertIn(actions[0].parameters['key'], ['w', 'a', 's', 'd'])
        
    def test_power_up_collection_actions(self):
        """Test power-up collection action generation"""
        strategic_assessment = {
            'power_up_control': {
                'controlled_count': 1,
                'contested_count': 0,
                'total_count': 1,
                'control_ratio': 1.0
            }
        }
        
        power_up_opportunity = Opportunity(OpportunityType.POWER_UP, (200, 200), 0.8, 100, 0.7)
        
        analysis = GameStateAnalysis(
            threats=[],
            opportunities=[power_up_opportunity],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[],
            projectiles=[],
            power_ups=[],
            game_boundaries={},
            safe_zones=[]
        )
        
        actions = self.strategic_rules._generate_power_up_collection_actions(analysis, strategic_assessment)
        
        # Should generate collection action
        self.assertGreater(len(actions), 0)
        
        # Should be a movement action towards power-up
        self.assertEqual(actions[0].action_type, 'collect_strategic_powerup')
        self.assertIn(actions[0].parameters['key'], ['w', 'a', 's', 'd'])


if __name__ == '__main__':
    unittest.main()