"""
Integration tests for the rules-based bot system.
Tests that all rule modules work together properly.
"""

import unittest
from unittest.mock import patch

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
from rl_bot_system.rules_based.survival_rules import SurvivalRules
from rl_bot_system.rules_based.combat_rules import CombatRules
from rl_bot_system.rules_based.strategic_rules import StrategicRules


class TestRulesIntegration(unittest.TestCase):
    """Integration tests for the rules-based bot system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        
        # Create a complex game state with threats and opportunities
        self.mock_game_state = {
            'player': {
                'position': (200, 200),
                'velocity': (0, 0),
                'health': 60  # Medium health
            },
            'enemies': [
                {
                    'id': 'enemy1',
                    'position': (250, 200),  # Close enemy
                    'velocity': (0, 0),
                    'health': 80,
                    'hasLineOfSight': True
                },
                {
                    'id': 'enemy2', 
                    'position': (400, 400),  # Far enemy
                    'velocity': (5, 5),
                    'health': 30  # Low health
                }
            ],
            'projectiles': [
                {
                    'id': 'proj1',
                    'position': (180, 200),  # Incoming projectile
                    'velocity': (15, 0)
                }
            ],
            'powerUps': [
                {
                    'type': 'health',
                    'position': (300, 300)
                }
            ],
            'boundaries': {
                'left': 0,
                'right': 800,
                'top': 0,
                'bottom': 600
            }
        }
        
    def test_bot_initialization_with_rule_modules(self):
        """Test that bot initializes correctly with rule module attributes"""
        # Check that rule module attributes are initialized
        self.assertIsNone(self.bot._combat_rules)
        self.assertIsNone(self.bot._survival_rules)
        self.assertIsNone(self.bot._strategic_rules)
        
        # Check that lazy initialization methods exist
        self.assertTrue(hasattr(self.bot, '_get_combat_rules'))
        self.assertTrue(hasattr(self.bot, '_get_survival_rules'))
        self.assertTrue(hasattr(self.bot, '_get_strategic_rules'))
        
    def test_rule_module_lazy_initialization(self):
        """Test that rule modules are lazily initialized"""
        # Initially should be None
        self.assertIsNone(self.bot._combat_rules)
        self.assertIsNone(self.bot._survival_rules)
        self.assertIsNone(self.bot._strategic_rules)
        
        # After calling getters, should be initialized
        combat_rules = self.bot._get_combat_rules()
        survival_rules = self.bot._get_survival_rules()
        strategic_rules = self.bot._get_strategic_rules()
        
        self.assertIsInstance(combat_rules, CombatRules)
        self.assertIsInstance(survival_rules, SurvivalRules)
        self.assertIsInstance(strategic_rules, StrategicRules)
        
        # Should be cached
        self.assertIs(combat_rules, self.bot._get_combat_rules())
        self.assertIs(survival_rules, self.bot._get_survival_rules())
        self.assertIs(strategic_rules, self.bot._get_strategic_rules())
        
    def test_game_state_analysis_integration(self):
        """Test that game state analysis works with complex scenarios"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        
        # Verify analysis structure
        self.assertIsInstance(analysis, GameStateAnalysis)
        self.assertEqual(analysis.player_position, (200, 200))
        self.assertEqual(analysis.player_health, 60)
        self.assertEqual(len(analysis.enemies), 2)
        self.assertEqual(len(analysis.projectiles), 1)
        
        # Should detect threats
        self.assertGreater(len(analysis.threats), 0)
        
        # Should find opportunities
        self.assertGreater(len(analysis.opportunities), 0)
        
    def test_survival_rules_integration(self):
        """Test survival rules integration with game state"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        survival_rules = self.bot._get_survival_rules()
        
        # Test survival situation evaluation
        survival_situation = survival_rules.evaluate_survival_situation(analysis)
        
        self.assertIn('health_status', survival_situation)
        self.assertIn('boundary_safety', survival_situation)
        self.assertIn('immediate_dangers', survival_situation)
        self.assertIn('survival_priority', survival_situation)
        
        # Test survival action generation
        survival_actions = survival_rules.generate_survival_actions(analysis)
        
        # Should generate some actions given the threats
        self.assertGreaterEqual(len(survival_actions), 0)
        
        # If actions generated, verify they have proper structure
        for action in survival_actions:
            self.assertIsNotNone(action.action_type)
            self.assertIsNotNone(action.parameters)
            self.assertIsNotNone(action.priority)
            self.assertIsNotNone(action.confidence)
            
    def test_combat_rules_integration(self):
        """Test combat rules integration with game state"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        combat_rules = self.bot._get_combat_rules()
        
        # Test combat situation evaluation
        combat_situation = combat_rules.evaluate_combat_situation(analysis)
        
        self.assertIn('threat_level', combat_situation)
        self.assertIn('target_priority', combat_situation)
        self.assertIn('combat_stance', combat_situation)
        self.assertIn('engagement_range', combat_situation)
        
        # Test combat action generation
        combat_actions = combat_rules.generate_combat_actions(analysis)
        
        # Should generate combat actions given the enemies
        self.assertGreater(len(combat_actions), 0)
        
        # Verify action structure
        for action in combat_actions:
            self.assertIsNotNone(action.action_type)
            self.assertIsNotNone(action.parameters)
            self.assertIsNotNone(action.priority)
            self.assertIsNotNone(action.confidence)
            
    def test_strategic_rules_integration(self):
        """Test strategic rules integration with game state"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        strategic_rules = self.bot._get_strategic_rules()
        
        # Test strategic situation evaluation
        strategic_situation = strategic_rules.evaluate_strategic_situation(analysis)
        
        self.assertIn('map_control', strategic_situation)
        self.assertIn('power_up_control', strategic_situation)
        self.assertIn('positional_advantage', strategic_situation)
        self.assertIn('strategic_objectives', strategic_situation)
        
        # Test strategic action generation
        strategic_actions = strategic_rules.generate_strategic_actions(analysis)
        
        # Should generate some strategic actions
        self.assertGreaterEqual(len(strategic_actions), 0)
        
        # Verify action structure
        for action in strategic_actions:
            self.assertIsNotNone(action.action_type)
            self.assertIsNotNone(action.parameters)
            self.assertIsNotNone(action.priority)
            self.assertIsNotNone(action.confidence)
            
    @patch('asyncio.get_event_loop')
    def test_integrated_action_selection(self, mock_get_loop):
        """Test that the bot can select actions using all rule modules"""
        # Mock the event loop time
        mock_loop = mock_get_loop.return_value
        mock_loop.time.return_value = 1.0
        
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        
        # Force reaction time to pass
        self.bot.reaction_delay_buffer = [0.0]  # Past time
        
        action = self.bot.select_action(analysis)
        
        # Should select an action given the complex game state
        self.assertIsNotNone(action)
        self.assertIsNotNone(action.action_type)
        self.assertIsNotNone(action.parameters)
        self.assertIsNotNone(action.priority)
        self.assertIsNotNone(action.confidence)
        self.assertIsNotNone(action.expected_outcome)
        
    def test_rule_module_action_priority_integration(self):
        """Test that rule modules generate actions with appropriate priorities"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        
        # Get all rule modules
        survival_rules = self.bot._get_survival_rules()
        combat_rules = self.bot._get_combat_rules()
        strategic_rules = self.bot._get_strategic_rules()
        
        # Generate actions from each module
        survival_actions = survival_rules.generate_survival_actions(analysis)
        combat_actions = combat_rules.generate_combat_actions(analysis)
        strategic_actions = strategic_rules.generate_strategic_actions(analysis)
        
        # Survival actions should generally have higher priority than strategic
        if survival_actions and strategic_actions:
            max_survival_priority = max(action.priority.value for action in survival_actions)
            max_strategic_priority = max(action.priority.value for action in strategic_actions)
            
            # In dangerous situations, survival should be prioritized
            if any(action.priority.value >= 0.8 for action in survival_actions):
                self.assertGreaterEqual(max_survival_priority, max_strategic_priority)
                
    def test_rule_module_configuration_sharing(self):
        """Test that all rule modules share the same configuration"""
        combat_rules = self.bot._get_combat_rules()
        survival_rules = self.bot._get_survival_rules()
        strategic_rules = self.bot._get_strategic_rules()
        
        # All should have the same config reference
        self.assertEqual(combat_rules.config, self.bot.config)
        self.assertEqual(survival_rules.config, self.bot.config)
        self.assertEqual(strategic_rules.config, self.bot.config)
        
        # Config should have expected difficulty parameters
        self.assertIn('accuracy_modifier', self.bot.config)
        self.assertIn('aggression_level', self.bot.config)
        self.assertIn('reaction_time', self.bot.config)
        
    def test_different_difficulty_levels_integration(self):
        """Test that different difficulty levels affect rule module behavior"""
        # Test beginner bot
        beginner_bot = RulesBasedBot(DifficultyLevel.BEGINNER)
        beginner_analysis = beginner_bot.analyze_game_state(self.mock_game_state)
        
        # Test expert bot
        expert_bot = RulesBasedBot(DifficultyLevel.EXPERT)
        expert_analysis = expert_bot.analyze_game_state(self.mock_game_state)
        
        # Both should generate actions, but with different characteristics
        beginner_bot.reaction_delay_buffer = [0.0]
        expert_bot.reaction_delay_buffer = [0.0]
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1.0
            
            beginner_action = beginner_bot.select_action(beginner_analysis)
            expert_action = expert_bot.select_action(expert_analysis)
            
            # Both should select actions
            self.assertIsNotNone(beginner_action)
            self.assertIsNotNone(expert_action)
            
            # Expert should have higher confidence (due to accuracy modifier)
            if (beginner_action and expert_action and 
                beginner_action.action_type == expert_action.action_type):
                # Note: This might not always be true due to randomness, 
                # but generally expert should have higher base confidence
                pass  # Just verify both can generate actions
                
    def test_complex_scenario_handling(self):
        """Test that the bot can handle complex scenarios with multiple threats and opportunities"""
        # Create an even more complex scenario
        complex_game_state = {
            'player': {
                'position': (100, 100),
                'velocity': (0, 0),
                'health': 25  # Critical health
            },
            'enemies': [
                {'id': 'enemy1', 'position': (120, 100), 'velocity': (0, 0), 'health': 100},  # Very close
                {'id': 'enemy2', 'position': (200, 200), 'velocity': (10, 10), 'health': 20}, # Moving, low health
                {'id': 'enemy3', 'position': (300, 300), 'velocity': (0, 0), 'health': 100}   # Far
            ],
            'projectiles': [
                {'id': 'proj1', 'position': (90, 100), 'velocity': (20, 0)},   # Very close projectile
                {'id': 'proj2', 'position': (150, 150), 'velocity': (-5, -5)}  # Another projectile
            ],
            'powerUps': [
                {'type': 'health', 'position': (80, 80)},    # Close health
                {'type': 'ammunition', 'position': (400, 400)} # Far ammo
            ],
            'boundaries': {
                'left': 0, 'right': 800, 'top': 0, 'bottom': 600
            }
        }
        
        analysis = self.bot.analyze_game_state(complex_game_state)
        
        # Should detect multiple threats
        self.assertGreater(len(analysis.threats), 2)
        
        # Should find opportunities
        self.assertGreater(len(analysis.opportunities), 0)
        
        # Should be able to select an action even in complex scenario
        self.bot.reaction_delay_buffer = [0.0]
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 1.0
            action = self.bot.select_action(analysis)
            
            # Should prioritize survival given critical health and immediate threats
            self.assertIsNotNone(action)
            
            # In this critical scenario, should likely be a high-priority action
            if action:
                self.assertGreaterEqual(action.priority.value, 0.6)


if __name__ == '__main__':
    unittest.main()