"""
Basic tests for the rules-based bot framework.
"""

import unittest
from unittest.mock import Mock, patch
import asyncio

# Path setup handled by test runner via PYTHONPATH

from rl_bot_system.rules_based.rules_based_bot import (
    RulesBasedBot, 
    DifficultyLevel, 
    ThreatType, 
    OpportunityType,
    Threat,
    Opportunity,
    GameStateAnalysis
)


class TestRulesBasedBot(unittest.TestCase):
    """Test cases for the RulesBasedBot class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        
    def test_bot_initialization(self):
        """Test that bot initializes correctly with different difficulty levels"""
        # Test default initialization
        bot = RulesBasedBot()
        self.assertEqual(bot.difficulty, DifficultyLevel.INTERMEDIATE)
        
        # Test specific difficulty levels
        for difficulty in DifficultyLevel:
            bot = RulesBasedBot(difficulty)
            self.assertEqual(bot.difficulty, difficulty)
            self.assertIsNotNone(bot.config)
            
    def test_difficulty_configuration(self):
        """Test that difficulty levels have correct parameter ranges"""
        # Test beginner configuration
        beginner_bot = RulesBasedBot(DifficultyLevel.BEGINNER)
        self.assertGreater(beginner_bot.config['reaction_time'], 0.2)
        self.assertLess(beginner_bot.config['accuracy_modifier'], 0.7)
        
        # Test expert configuration  
        expert_bot = RulesBasedBot(DifficultyLevel.EXPERT)
        self.assertLess(expert_bot.config['reaction_time'], 0.05)
        self.assertGreater(expert_bot.config['accuracy_modifier'], 0.9)
        
    def test_difficulty_level_change(self):
        """Test changing difficulty level updates configuration"""
        original_reaction_time = self.bot.config['reaction_time']
        
        self.bot.set_difficulty_level(DifficultyLevel.EXPERT)
        self.assertEqual(self.bot.difficulty, DifficultyLevel.EXPERT)
        self.assertNotEqual(self.bot.config['reaction_time'], original_reaction_time)
        
    def test_game_state_analysis(self):
        """Test game state analysis functionality"""
        # Mock game state
        mock_game_state = {
            'player': {
                'position': (100, 100),
                'velocity': (0, 0),
                'health': 100
            },
            'enemies': [
                {
                    'id': 'enemy1',
                    'position': (150, 100),
                    'velocity': (0, 0),
                    'health': 100
                }
            ],
            'projectiles': [
                {
                    'id': 'proj1',
                    'position': (120, 100),
                    'velocity': (10, 0)
                }
            ],
            'powerUps': [],
            'boundaries': {
                'left': 0,
                'right': 800,
                'top': 0,
                'bottom': 600
            }
        }
        
        analysis = self.bot.analyze_game_state(mock_game_state)
        
        # Verify analysis structure
        self.assertIsInstance(analysis, GameStateAnalysis)
        self.assertEqual(analysis.player_position, (100, 100))
        self.assertEqual(len(analysis.enemies), 1)
        self.assertEqual(len(analysis.projectiles), 1)
        
    def test_threat_detection(self):
        """Test threat detection functionality"""
        mock_game_state = {
            'player': {'position': (100, 100), 'velocity': (0, 0), 'health': 100},
            'enemies': [{'id': 'enemy1', 'position': (150, 100), 'velocity': (0, 0)}],
            'projectiles': [{'id': 'proj1', 'position': (120, 100), 'velocity': (-10, 0)}],
            'boundaries': {'left': 0, 'right': 800, 'top': 0, 'bottom': 600}
        }
        
        player_info = {'position': (100, 100), 'velocity': (0, 0), 'health': 100}
        threats = self.bot.detect_threats(mock_game_state, player_info)
        
        # Should detect both projectile and enemy threats
        self.assertGreater(len(threats), 0)
        
        # Verify threat types
        threat_types = [threat.threat_type for threat in threats]
        self.assertIn(ThreatType.PROJECTILE, threat_types)
        self.assertIn(ThreatType.ENEMY, threat_types)
        
    def test_opportunity_detection(self):
        """Test opportunity detection functionality"""
        mock_game_state = {
            'player': {'position': (100, 100), 'velocity': (0, 0), 'health': 100},
            'enemies': [{'id': 'enemy1', 'position': (150, 100), 'velocity': (0, 0)}],
            'powerUps': [{'type': 'health', 'position': (120, 120)}],
            'platforms': [{'position': (200, 80), 'height': 80}]
        }
        
        player_info = {'position': (100, 100), 'velocity': (0, 0), 'health': 100}
        opportunities = self.bot.find_opportunities(mock_game_state, player_info)
        
        # Should find opportunities
        self.assertGreater(len(opportunities), 0)
        
        opportunity_types = [opp.opportunity_type for opp in opportunities]
        # At minimum should find power-up opportunities
        self.assertIn(OpportunityType.POWER_UP, opportunity_types)
        # Attack opportunities may depend on specific game state conditions
        
    @patch('asyncio.get_event_loop')
    def test_action_selection(self, mock_get_loop):
        """Test action selection logic"""
        # Mock the event loop time
        mock_loop = Mock()
        mock_loop.time.return_value = 1.0
        mock_get_loop.return_value = mock_loop
        
        # Create a mock analysis with threats and opportunities
        analysis = GameStateAnalysis(
            threats=[
                Threat(ThreatType.PROJECTILE, (120, 100), (10, 0), 0.8, 0.5, 'proj1')
            ],
            opportunities=[
                Opportunity(OpportunityType.ATTACK, (150, 100), 0.7, 50, 0.6)
            ],
            player_position=(100, 100),
            player_velocity=(0, 0),
            player_health=100,
            enemies=[{'id': 'enemy1', 'position': (150, 100)}],
            projectiles=[{'id': 'proj1', 'position': (120, 100)}],
            power_ups=[],
            game_boundaries={'left': 0, 'right': 800, 'top': 0, 'bottom': 600},
            safe_zones=[]
        )
        
        # Force reaction time to pass
        self.bot.reaction_delay_buffer = [0.5]  # Past time
        
        action = self.bot.select_action(analysis)
        
        # Should select some action
        self.assertIsNotNone(action)
        self.assertIsNotNone(action.action_type)
        self.assertIsNotNone(action.parameters)
        
    def test_performance_metrics_update(self):
        """Test performance metrics tracking"""
        initial_metrics = self.bot.get_performance_metrics()
        self.assertEqual(initial_metrics['wins'], 0)
        self.assertEqual(initial_metrics['losses'], 0)
        
        # Simulate a win
        game_result = {
            'won': True,
            'kills': 2,
            'deaths': 1,
            'shots_fired': 10,
            'shots_hit': 7
        }
        
        self.bot.update_performance_metrics(game_result)
        
        updated_metrics = self.bot.get_performance_metrics()
        self.assertEqual(updated_metrics['wins'], 1)
        self.assertEqual(updated_metrics['kills'], 2)
        self.assertEqual(updated_metrics['deaths'], 1)
        self.assertGreater(updated_metrics['accuracy'], 0)
        self.assertGreater(updated_metrics['win_rate'], 0)
        
    def test_projectile_threat_analysis(self):
        """Test projectile threat analysis"""
        projectile = {
            'id': 'proj1',
            'position': (120, 100),
            'velocity': (-10, 0)  # Moving towards player (negative x velocity)
        }
        player_pos = (100, 100)
        
        threat = self.bot._analyze_projectile_threat(projectile, player_pos)
        
        self.assertIsNotNone(threat)
        self.assertEqual(threat.threat_type, ThreatType.PROJECTILE)
        self.assertGreater(threat.severity, 0)
        self.assertIsNotNone(threat.time_to_impact)
        
    def test_enemy_threat_analysis(self):
        """Test enemy threat analysis"""
        enemy = {
            'id': 'enemy1',
            'position': (150, 100),
            'velocity': (0, 0),
            'hasLineOfSight': True
        }
        player_pos = (100, 100)
        
        threat = self.bot._analyze_enemy_threat(enemy, player_pos)
        
        self.assertIsNotNone(threat)
        self.assertEqual(threat.threat_type, ThreatType.ENEMY)
        self.assertGreater(threat.severity, 0)
        
    def test_boundary_threat_analysis(self):
        """Test boundary threat analysis"""
        boundaries = {
            'left': 0,
            'right': 800,
            'top': 0,
            'bottom': 600
        }
        
        # Player close to left boundary
        player_pos = (20, 300)
        threat = self.bot._analyze_boundary_threat(player_pos, boundaries)
        
        self.assertIsNotNone(threat)
        self.assertEqual(threat.threat_type, ThreatType.BOUNDARY)
        self.assertGreater(threat.severity, 0)
        
        # Player in safe position
        player_pos = (400, 300)
        threat = self.bot._analyze_boundary_threat(player_pos, boundaries)
        
        self.assertIsNone(threat)


if __name__ == '__main__':
    unittest.main()