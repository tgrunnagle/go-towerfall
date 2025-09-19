#!/usr/bin/env python3
"""
Unit tests for difficulty scaling and adaptive behavior in the rules-based bot.
This script tests the bot's behavior without requiring a game server connection.
"""

import unittest
import asyncio
import random
from unittest.mock import Mock, patch

from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel, Action, ActionPriority


class TestDifficultyScaling(unittest.TestCase):
    """Test difficulty scaling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_game_state = {
            'player': {
                'position': (100, 100),
                'velocity': (0, 0),
                'health': 100,
                'ammunition': 10
            },
            'enemies': [
                {
                    'id': 'enemy1',
                    'position': (200, 150),
                    'velocity': (5, 0),
                    'health': 80
                }
            ],
            'projectiles': [
                {
                    'id': 'proj1',
                    'position': (150, 120),
                    'velocity': (20, 5)
                }
            ],
            'powerUps': [
                {
                    'type': 'health',
                    'position': (250, 200)
                }
            ],
            'boundaries': {
                'left': -400,
                'right': 400,
                'top': -300,
                'bottom': 300
            }
        }
    
    def test_difficulty_level_configuration(self):
        """Test that different difficulty levels have different configurations"""
        difficulties = [
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ]
        
        bots = [RulesBasedBot(diff) for diff in difficulties]
        
        # Test that reaction times decrease with higher difficulty
        reaction_times = [bot.config['reaction_time'] for bot in bots]
        self.assertEqual(reaction_times, sorted(reaction_times, reverse=True))
        
        # Test that accuracy increases with higher difficulty
        accuracies = [bot.config['accuracy_modifier'] for bot in bots]
        self.assertEqual(accuracies, sorted(accuracies))
        
        # Test that aggression levels generally increase
        aggressions = [bot.config['aggression_level'] for bot in bots]
        self.assertLess(aggressions[0], aggressions[-1])  # Beginner < Expert
        
        print("✓ Difficulty level configurations are properly scaled")
    
    def test_difficulty_level_switching(self):
        """Test that bots can switch difficulty levels"""
        bot = RulesBasedBot(DifficultyLevel.BEGINNER)
        original_reaction_time = bot.config['reaction_time']
        
        # Switch to expert
        bot.set_difficulty_level(DifficultyLevel.EXPERT)
        new_reaction_time = bot.config['reaction_time']
        
        self.assertEqual(bot.difficulty, DifficultyLevel.EXPERT)
        self.assertLess(new_reaction_time, original_reaction_time)
        
        print("✓ Difficulty level switching works correctly")
    
    def test_shooting_accuracy_by_difficulty(self):
        """Test that shooting accuracy varies by difficulty"""
        difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.EXPERT]
        
        for difficulty in difficulties:
            bot = RulesBasedBot(difficulty)
            analysis = bot.analyze_game_state(self.mock_game_state)
            
            # Create a shooting action
            action = Action(
                action_type='shoot_at_enemy',
                parameters={'x': 200, 'y': 150, 'target_id': 'enemy1'},
                priority=ActionPriority.HIGH,
                confidence=0.8,
                expected_outcome="Attack enemy",
                duration=0.05
            )
            
            # Apply difficulty modifiers multiple times to see variation
            original_x, original_y = action.parameters['x'], action.parameters['y']
            errors = []
            
            for _ in range(10):
                test_action = Action(
                    action_type='shoot_at_enemy',
                    parameters={'x': original_x, 'y': original_y, 'target_id': 'enemy1'},
                    priority=ActionPriority.HIGH,
                    confidence=0.8,
                    expected_outcome="Attack enemy",
                    duration=0.05
                )
                
                modified_action = bot._apply_difficulty_modifiers(test_action)
                error = abs(modified_action.parameters['x'] - original_x) + abs(modified_action.parameters['y'] - original_y)
                errors.append(error)
            
            avg_error = sum(errors) / len(errors)
            
            if difficulty == DifficultyLevel.BEGINNER:
                beginner_error = avg_error
            else:
                expert_error = avg_error
        
        # Expert should have lower average error than beginner
        self.assertLess(expert_error, beginner_error)
        print(f"✓ Shooting accuracy scales with difficulty (Beginner avg error: {beginner_error:.1f}, Expert: {expert_error:.1f})")
    
    def test_reaction_time_delays(self):
        """Test that reaction time delays work correctly"""
        bot = RulesBasedBot(DifficultyLevel.BEGINNER)
        
        # Mock asyncio event loop time
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 0.0
            
            # First call should return False (reaction delay)
            self.assertFalse(bot._should_react_now())
            
            # Should have added delay to buffer
            self.assertGreater(len(bot.reaction_delay_buffer), 0)
            
            # Simulate time passing
            mock_loop.return_value.time.return_value = 1.0
            
            # Now should return True
            self.assertTrue(bot._should_react_now())
        
        print("✓ Reaction time delays work correctly")


class TestAdaptiveBehavior(unittest.TestCase):
    """Test adaptive behavior functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        self.bot.set_adaptation_enabled(True)
    
    def test_adaptation_enabled_disabled(self):
        """Test enabling and disabling adaptation"""
        self.assertTrue(self.bot.adaptation_enabled)
        
        self.bot.set_adaptation_enabled(False)
        self.assertFalse(self.bot.adaptation_enabled)
        
        print("✓ Adaptation can be enabled/disabled")
    
    def test_performance_tracking(self):
        """Test that performance metrics are tracked correctly"""
        initial_wins = self.bot.performance_metrics['wins']
        initial_losses = self.bot.performance_metrics['losses']
        
        # Simulate a win
        self.bot.update_performance_metrics({
            'won': True,
            'kills': 2,
            'deaths': 1,
            'shots_fired': 10,
            'shots_hit': 6
        })
        
        self.assertEqual(self.bot.performance_metrics['wins'], initial_wins + 1)
        self.assertEqual(self.bot.performance_metrics['kills'], 2)
        self.assertEqual(len(self.bot.game_history), 1)
        
        print("✓ Performance tracking works correctly")
    
    def test_strategy_adaptation(self):
        """Test that strategy adapts based on performance"""
        original_strategy = self.bot.current_strategy
        
        # Simulate multiple losses to trigger defensive adaptation
        for _ in range(5):
            self.bot.update_performance_metrics({
                'won': False,
                'kills': 0,
                'deaths': 2,
                'shots_fired': 5,
                'shots_hit': 1
            })
        
        # Strategy should have changed to defensive
        self.assertNotEqual(self.bot.current_strategy, original_strategy)
        print(f"✓ Strategy adapted from {original_strategy} to {self.bot.current_strategy} after losses")
    
    def test_config_adaptation(self):
        """Test that configuration parameters adapt"""
        original_aggression = self.bot.config['aggression_level']
        original_reaction_time = self.bot.config['reaction_time']
        
        # Simulate poor performance
        for _ in range(5):
            self.bot.update_performance_metrics({
                'won': False,
                'kills': 0,
                'deaths': 2,
                'shots_fired': 10,
                'shots_hit': 2  # Poor accuracy
            })
        
        # Configuration should have adapted
        new_aggression = self.bot.config['aggression_level']
        new_reaction_time = self.bot.config['reaction_time']
        
        self.assertLess(new_aggression, original_aggression)  # Should be less aggressive
        self.assertGreater(new_reaction_time, original_reaction_time)  # Should be slower
        
        print("✓ Configuration parameters adapt based on performance")
    
    def test_adaptation_reset(self):
        """Test that adaptation can be reset"""
        # Make some changes
        self.bot.current_strategy = "aggressive"
        self.bot.config['aggression_level'] = 0.9
        self.bot.game_history.append({'won': True})
        
        # Reset adaptation
        self.bot.reset_adaptation()
        
        self.assertEqual(self.bot.current_strategy, "balanced")
        self.assertEqual(len(self.bot.game_history), 0)
        self.assertEqual(self.bot.config['aggression_level'], self.bot.base_config['aggression_level'])
        
        print("✓ Adaptation reset works correctly")
    
    def test_adaptation_status(self):
        """Test adaptation status reporting"""
        # Add some game history
        for i in range(3):
            self.bot.update_performance_metrics({
                'won': i % 2 == 0,  # Win every other game
                'kills': 1,
                'deaths': 1,
                'shots_fired': 5,
                'shots_hit': 3
            })
        
        status = self.bot.get_adaptation_status()
        
        self.assertTrue(status['adaptation_enabled'])
        self.assertEqual(status['games_played'], 3)
        self.assertIn('recent_win_rate', status)
        self.assertIn('current_strategy', status)
        self.assertIn('config_changes', status)
        
        print("✓ Adaptation status reporting works correctly")


class TestGameStateAnalysis(unittest.TestCase):
    """Test game state analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        self.mock_game_state = {
            'player': {
                'position': (100, 100),
                'velocity': (0, 0),
                'health': 100,
                'ammunition': 10
            },
            'enemies': [
                {
                    'id': 'enemy1',
                    'position': (200, 150),
                    'velocity': (5, 0),
                    'health': 80
                }
            ],
            'projectiles': [
                {
                    'id': 'proj1',
                    'position': (150, 120),
                    'velocity': (-20, -5)  # Moving towards player
                }
            ],
            'powerUps': [
                {
                    'type': 'health',
                    'position': (120, 110)  # Close to player
                }
            ],
            'boundaries': {
                'left': -400,
                'right': 400,
                'top': -300,
                'bottom': 300
            }
        }
    
    def test_threat_detection(self):
        """Test that threats are detected correctly"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        
        # Should detect projectile and enemy threats
        self.assertGreater(len(analysis.threats), 0)
        
        threat_types = [threat.threat_type.value for threat in analysis.threats]
        self.assertIn('projectile', threat_types)
        self.assertIn('enemy', threat_types)
        
        print(f"✓ Detected {len(analysis.threats)} threats: {threat_types}")
    
    def test_opportunity_detection(self):
        """Test that opportunities are detected correctly"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        
        # Should detect attack and power-up opportunities
        self.assertGreater(len(analysis.opportunities), 0)
        
        opportunity_types = [opp.opportunity_type.value for opp in analysis.opportunities]
        self.assertIn('attack', opportunity_types)
        self.assertIn('power_up', opportunity_types)
        
        print(f"✓ Detected {len(analysis.opportunities)} opportunities: {opportunity_types}")
    
    def test_action_selection(self):
        """Test that actions are selected appropriately"""
        analysis = self.bot.analyze_game_state(self.mock_game_state)
        
        # Should select an action based on the analysis
        action = self.bot.select_action(analysis)
        
        if action:  # Might be None due to reaction time delays
            self.assertIsNotNone(action.action_type)
            self.assertIsNotNone(action.priority)
            self.assertGreater(action.confidence, 0)
            
            print(f"✓ Selected action: {action.action_type} (priority: {action.priority.value}, confidence: {action.confidence:.2f})")
        else:
            print("✓ No action selected (reaction time delay)")


def run_performance_comparison():
    """Run a performance comparison between difficulty levels"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    difficulties = [
        DifficultyLevel.BEGINNER,
        DifficultyLevel.INTERMEDIATE,
        DifficultyLevel.ADVANCED,
        DifficultyLevel.EXPERT
    ]
    
    mock_game_state = {
        'player': {'position': (100, 100), 'velocity': (0, 0), 'health': 100, 'ammunition': 10},
        'enemies': [{'id': 'enemy1', 'position': (200, 150), 'velocity': (5, 0), 'health': 80}],
        'projectiles': [],
        'powerUps': [],
        'boundaries': {'left': -400, 'right': 400, 'top': -300, 'bottom': 300}
    }
    
    for difficulty in difficulties:
        bot = RulesBasedBot(difficulty)
        
        # Simulate decision making speed
        start_time = 0
        decisions_made = 0
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = start_time
            
            for i in range(100):
                mock_loop.return_value.time.return_value = start_time + (i * 0.01)
                
                analysis = bot.analyze_game_state(mock_game_state)
                action = bot.select_action(analysis)
                
                if action:
                    decisions_made += 1
        
        decision_rate = decisions_made / 1.0  # decisions per second
        
        print(f"{difficulty.value.upper():>12}: "
              f"Reaction: {bot.config['reaction_time']*1000:>3.0f}ms, "
              f"Accuracy: {bot.config['accuracy_modifier']:>4.1%}, "
              f"Decisions/sec: {decision_rate:>4.1f}")


if __name__ == "__main__":
    import os
    print("Testing Rules-Based Bot Difficulty Scaling and Adaptive Behavior")
    print("="*70)
    
    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestDifficultyScaling))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestAdaptiveBehavior))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestGameStateAnalysis))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(test_suite)
    
    # Print results
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for failure in result.failures:
            print(f"  FAIL: {failure[0]}")
        for error in result.errors:
            print(f"  ERROR: {error[0]}")
    
    # Run performance comparison
    run_performance_comparison()
    
    print("\n" + "="*70)
    print("Testing completed!")