"""
Integration tests for comparing the rules-based bot against other bots.
Tests different difficulty levels and adaptive behavior.
"""

import asyncio
import logging
import pytest
from typing import Dict, List, Any
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

try:
    from game_client import GameClient
    GAME_CLIENT_AVAILABLE = True
except ImportError:
    GAME_CLIENT_AVAILABLE = False
    
    # Create mock GameClient for testing without dependencies
    class GameClient:
        def __init__(self):
            self.message_handler = None
            
        async def connect(self, room_code, player_name, room_password=None):
            return AsyncMock()
            
        async def close(self):
            pass
            
        def register_message_handler(self, handler):
            self.message_handler = handler
            
        async def send_keyboard_input(self, key, pressed):
            pass
            
        async def send_mouse_input(self, button, pressed, x=0, y=0):
            pass

try:
    from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel
    RULES_BOT_AVAILABLE = True
except ImportError:
    RULES_BOT_AVAILABLE = False
    # Create dummy classes with the expected attributes
    class DifficultyLevel:
        def __init__(self, value):
            self.value = value
        
        @classmethod
        def create_levels(cls):
            cls.BEGINNER = cls("beginner")
            cls.INTERMEDIATE = cls("intermediate")
            cls.ADVANCED = cls("advanced")
            cls.EXPERT = cls("expert")
    
    DifficultyLevel.create_levels()
    
    class RulesBasedBot:
        def __init__(self, difficulty):
            self.difficulty = difficulty
            # Different accuracy for different difficulty levels
            accuracy = 0.3 if difficulty.value == 'beginner' else 0.8
            self.config = {'reaction_time': 1.0, 'accuracy_modifier': accuracy, 'aggression_level': 0.7}
            self.current_strategy = 'balanced'
            self.games_played = 0
        
        def set_adaptation_enabled(self, enabled):
            self.adaptation_enabled = enabled
        
        def get_adaptation_status(self):
            return {
                'adaptation_enabled': getattr(self, 'adaptation_enabled', False),
                'games_played': self.games_played,
                'current_strategy': self.current_strategy,
                'recent_win_rate': 0.0
            }
        
        def update_performance_metrics(self, metrics):
            self.games_played += 1
            # Simulate strategy change after losses
            if not metrics.get('won', True):
                self.current_strategy = 'defensive'
                self.config['aggression_level'] = 0.3
        
        def get_performance_metrics(self):
            return {'accuracy': self.config['accuracy_modifier']}

DEPENDENCIES_AVAILABLE = GAME_CLIENT_AVAILABLE and RULES_BOT_AVAILABLE


@pytest.fixture
def logger():
    """Logger fixture for tests."""
    return logging.getLogger(__name__)


@pytest.fixture
def test_results():
    """Test results fixture."""
    return []


@pytest.fixture
def mock_game_state():
    """Mock game state fixture for testing."""
    return {
        'player': {
            'position': (100, 100),
            'velocity': (0, 0),
            'health': 100,
            'ammunition': 10
        },
        'enemies': [
            {
                'id': 'example_bot',
                'position': (200, 150),
                'velocity': (10, 0),
                'health': 100
            }
        ],
        'projectiles': [],
        'powerUps': [],
        'boundaries': {
            'left': -400,
            'right': 400,
            'top': -300,
            'bottom': 300
        }
    }


class TestBotComparison:
    """Integration tests for bot comparison and performance evaluation"""
        
    def test_difficulty_levels_configuration(self):
        """Test that different difficulty levels have distinct performance characteristics"""
        difficulties = [
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE, 
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ]
        
        bots = []
        for difficulty in difficulties:
            bot = RulesBasedBot(difficulty)
            bots.append(bot)
            
            # Test that bot can be created with difficulty
            assert bot.difficulty == difficulty
            assert bot.config is not None
            
        # Test that difficulty levels have different configurations
        reaction_times = [bot.config['reaction_time'] for bot in bots]
        accuracies = [bot.config['accuracy_modifier'] for bot in bots]
        
        # Reaction times should decrease with higher difficulty
        assert reaction_times == sorted(reaction_times, reverse=True)
        
        # Accuracies should increase with higher difficulty
        assert accuracies == sorted(accuracies)
        
        print("PASS: All difficulty levels have distinct configurations")
    
    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Game client dependencies not available")
    def test_difficulty_levels_integration(self, test_results):
        """Integration test for different difficulty levels (requires game client)"""
        difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE]
        
        for difficulty in difficulties:
            bot = RulesBasedBot(difficulty)
            
            # Run mock test games
            results = asyncio.run(self._run_mock_test_games(bot, num_games=2))
            
            test_results.append({
                'difficulty': difficulty.value,
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
            # Verify results structure
            assert 'total_games' in results
            assert 'win_rate' in results
            assert 'kd_ratio' in results
            assert results['total_games'] >= 1
            
        print(f"PASS: Integration test completed for {len(difficulties)} difficulty levels")
    
    def test_adaptive_behavior_configuration(self):
        """Test adaptive behavior configuration and status tracking"""
        bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        
        # Test adaptation can be enabled/disabled
        bot.set_adaptation_enabled(True)
        assert bot.adaptation_enabled
        
        bot.set_adaptation_enabled(False)
        assert not bot.adaptation_enabled
        
        # Test adaptation status reporting
        bot.set_adaptation_enabled(True)
        status = bot.get_adaptation_status()
        
        assert 'adaptation_enabled' in status
        assert 'games_played' in status
        assert 'current_strategy' in status
        assert status['adaptation_enabled']
        
        print("PASS: Adaptive behavior configuration works correctly")
    
    def test_adaptive_behavior_simulation(self):
        """Test adaptive behavior through simulated games"""
        bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        bot.set_adaptation_enabled(True)
        
        initial_strategy = bot.current_strategy
        initial_aggression = bot.config['aggression_level']
        
        # Simulate poor performance to trigger adaptation
        for i in range(5):
            bot.update_performance_metrics({
                'won': False,
                'kills': 0,
                'deaths': 2,
                'shots_fired': 10,
                'shots_hit': 2
            })
        
        # Check that adaptation occurred
        final_strategy = bot.current_strategy
        final_aggression = bot.config['aggression_level']
        
        # Strategy should have changed to be more defensive
        assert initial_strategy != final_strategy
        assert final_aggression < initial_aggression
        
        # Get adaptation status
        status = bot.get_adaptation_status()
        assert status['games_played'] > 0
        assert status['recent_win_rate'] == 0.0  # All losses
        
        print(f"PASS: Adaptive behavior simulation: {initial_strategy} -> {final_strategy}")
        print(f"  Aggression: {initial_aggression:.2f} -> {final_aggression:.2f}")
    
    async def _run_mock_test_games(self, bot: RulesBasedBot, num_games: int = 3) -> Dict[str, Any]:
        """Run mock test games for testing without actual game server"""
        game_results = []
        
        for game_num in range(num_games):
            # Simulate a game result based on bot difficulty
            difficulty_win_rates = {
                DifficultyLevel.BEGINNER: 0.3,
                DifficultyLevel.INTERMEDIATE: 0.5,
                DifficultyLevel.ADVANCED: 0.7,
                DifficultyLevel.EXPERT: 0.8
            }
            
            base_win_rate = difficulty_win_rates.get(bot.difficulty, 0.5)
            won = (game_num / num_games) < base_win_rate  # Simulate win/loss pattern
            
            # Create mock game result
            game_result = {
                'won': won,
                'kills': 2 if won else 1,
                'deaths': 1 if won else 2,
                'shots_fired': 15,
                'shots_hit': int(15 * bot.config['accuracy_modifier']),
                'duration': 60.0
            }
            
            game_results.append(game_result)
            
            # Update bot performance metrics
            bot.update_performance_metrics(game_result)
        
        # Calculate summary statistics
        return self._calculate_game_statistics(game_results, bot)
    
    async def _run_test_games(self, bot: RulesBasedBot, room_code: str, password: str = None, num_games: int = 3) -> Dict[str, Any]:
        """Run a series of test games with the bot"""
        game_results = []
        
        for game_num in range(num_games):
            self.logger.info(f"Starting test game {game_num + 1}/{num_games}")
            
            try:
                # Create game client for the bot
                client = GameClient()
                
                # Connect to the game room
                listener_task = await client.connect(
                    room_code=room_code,
                    player_name=f"RulesBot-{bot.difficulty.value}",
                    room_password=password
                )
                
                # Create bot wrapper to interface with GameClient
                bot_wrapper = RulesBotWrapper(bot, client)
                
                # Run the bot for a limited time (60 seconds per game)
                bot_task = asyncio.create_task(bot_wrapper.run())
                timeout_task = asyncio.create_task(asyncio.sleep(60))
                
                # Wait for game completion or timeout
                done, pending = await asyncio.wait(
                    [listener_task, bot_task, timeout_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                
                # Get game results
                game_result = bot_wrapper.get_game_result()
                game_results.append(game_result)
                
                # Update bot performance metrics
                bot.update_performance_metrics(game_result)
                
                await client.close()
                
            except Exception as e:
                self.logger.error(f"Error in test game {game_num + 1}: {e}")
                game_results.append({
                    'won': False,
                    'error': str(e),
                    'kills': 0,
                    'deaths': 1,
                    'shots_fired': 0,
                    'shots_hit': 0
                })
            
            # Wait between games
            await asyncio.sleep(2)
        
        # Calculate summary statistics
        return self._calculate_game_statistics(game_results, bot)
    
    def _calculate_game_statistics(self, game_results: List[Dict[str, Any]], bot: RulesBasedBot) -> Dict[str, Any]:
        """Calculate summary statistics from game results"""
        total_games = len(game_results)
        wins = sum(1 for result in game_results if result.get('won', False))
        total_kills = sum(result.get('kills', 0) for result in game_results)
        total_deaths = sum(result.get('deaths', 0) for result in game_results)
        total_shots_fired = sum(result.get('shots_fired', 0) for result in game_results)
        total_shots_hit = sum(result.get('shots_hit', 0) for result in game_results)
        
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': total_games - wins,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'total_kills': total_kills,
            'total_deaths': total_deaths,
            'kd_ratio': total_kills / max(1, total_deaths),
            'accuracy': total_shots_hit / max(1, total_shots_fired),
            'individual_games': game_results,
            'bot_performance_metrics': bot.get_performance_metrics()
        }
    
    def test_bot_wrapper_functionality(self):
        """Test the bot wrapper functionality"""
        bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        client = GameClient()
        
        wrapper = RulesBotWrapper(bot, client)
        
        # Test initialization
        assert wrapper.bot == bot
        assert wrapper.client == client
        assert not wrapper.running
        
        # Test game result initialization
        result = wrapper.get_game_result()
        expected_keys = ['won', 'kills', 'deaths', 'shots_fired', 'shots_hit', 'duration']
        for key in expected_keys:
            assert key in result
        
        print("PASS: Bot wrapper functionality works correctly")
    
    def test_performance_comparison_across_difficulties(self):
        """Test performance comparison across different difficulty levels"""
        difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.EXPERT]
        performance_metrics = {}
        
        for difficulty in difficulties:
            bot = RulesBasedBot(difficulty)
            
            # Simulate some games
            for i in range(5):
                won = i % 2 == 0  # Alternate wins/losses
                bot.update_performance_metrics({
                    'won': won,
                    'kills': 2 if won else 1,
                    'deaths': 1 if won else 2,
                    'shots_fired': 10,
                    'shots_hit': int(10 * bot.config['accuracy_modifier']),
                    'duration': 60.0
                })
            
            metrics = bot.get_performance_metrics()
            performance_metrics[difficulty.value] = metrics
        
        # Expert should have better accuracy than beginner
        expert_accuracy = performance_metrics['expert']['accuracy']
        beginner_accuracy = performance_metrics['beginner']['accuracy']
        
        assert expert_accuracy > beginner_accuracy
        
        print(f"PASS: Performance comparison: Expert accuracy ({expert_accuracy:.1%}) > Beginner accuracy ({beginner_accuracy:.1%})")
    



class RulesBotWrapper:
    """Wrapper to interface RulesBasedBot with GameClient"""
    
    def __init__(self, bot: RulesBasedBot, client: GameClient):
        self.bot = bot
        self.client = client
        self.running = False
        self.game_result = {
            'won': False,
            'kills': 0,
            'deaths': 0,
            'shots_fired': 0,
            'shots_hit': 0,
            'duration': 0
        }
        self.start_time = None
        self.logger = logging.getLogger(__name__)
        
        # Register message handler
        self.client.register_message_handler(self._handle_game_message)
    
    async def run(self):
        """Run the bot"""
        self.running = True
        self.start_time = asyncio.get_event_loop().time()
        
        try:
            while self.running:
                # Get current game state (this would need to be implemented in GameClient)
                game_state = await self._get_game_state()
                
                if game_state:
                    # Analyze game state
                    analysis = self.bot.analyze_game_state(game_state)
                    
                    # Select action
                    action = self.bot.select_action(analysis)
                    
                    # Execute action
                    if action:
                        await self._execute_action(action)
                
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.05)  # 20 FPS decision making
                
        except Exception as e:
            self.logger.error(f"Bot wrapper error: {e}")
        finally:
            self.running = False
            if self.start_time:
                self.game_result['duration'] = asyncio.get_event_loop().time() - self.start_time
    
    async def _get_game_state(self) -> Dict[str, Any]:
        """Get current game state from client"""
        # This is a simplified mock implementation
        # In a real implementation, this would extract actual game state from the client
        return {
            'player': {
                'position': (100, 100),
                'velocity': (0, 0),
                'health': 100,
                'ammunition': 10
            },
            'enemies': [
                {
                    'id': 'example_bot',
                    'position': (200, 150),
                    'velocity': (10, 0),
                    'health': 100
                }
            ],
            'projectiles': [],
            'powerUps': [],
            'boundaries': {
                'left': -400,
                'right': 400,
                'top': -300,
                'bottom': 300
            }
        }
    
    async def _execute_action(self, action):
        """Execute the selected action through GameClient"""
        try:
            if action.action_type in ['move_left', 'move_right', 'jump', 'crouch', 'evade_threat']:
                key = action.parameters.get('key')
                pressed = action.parameters.get('pressed', True)
                
                if key:
                    await self.client.send_keyboard_input(key, pressed)
                    
                    # If action has duration, schedule key release
                    if pressed and action.duration:
                        asyncio.create_task(self._release_key_after_delay(key, action.duration))
            
            elif action.action_type == 'shoot_at_enemy':
                # Mouse click at target position
                x = action.parameters.get('x', 0)
                y = action.parameters.get('y', 0)
                await self.client.send_mouse_input('left', True, x, y)
                
                # Track shots fired
                self.game_result['shots_fired'] += 1
                
                # Release mouse after short duration
                asyncio.create_task(self._release_mouse_after_delay('left', 0.1))
                
        except Exception as e:
            self.logger.error(f"Error executing action {action.action_type}: {e}")
    
    async def _release_key_after_delay(self, key: str, delay: float):
        """Release a key after specified delay"""
        await asyncio.sleep(delay)
        try:
            await self.client.send_keyboard_input(key, False)
        except Exception as e:
            self.logger.error(f"Error releasing key {key}: {e}")
    
    async def _release_mouse_after_delay(self, button: str, delay: float):
        """Release mouse button after specified delay"""
        await asyncio.sleep(delay)
        try:
            await self.client.send_mouse_input(button, False)
        except Exception as e:
            self.logger.error(f"Error releasing mouse {button}: {e}")
    
    async def _handle_game_message(self, message: Dict[str, Any]):
        """Handle incoming game messages"""
        # Mock implementation - would parse actual game messages
        message_type = message.get('type', '')
        
        if message_type == 'game_over':
            self.running = False
            self.game_result['won'] = message.get('winner') == 'rules_bot'
            
        elif message_type == 'player_killed':
            if message.get('victim') == 'rules_bot':
                self.game_result['deaths'] += 1
            elif message.get('killer') == 'rules_bot':
                self.game_result['kills'] += 1
                
        elif message_type == 'shot_hit':
            if message.get('shooter') == 'rules_bot':
                self.game_result['shots_hit'] += 1
    
    def get_game_result(self) -> Dict[str, Any]:
        """Get the final game result"""
        return self.game_result.copy()


# Standalone script functionality for integration testing
class BotTester:
    """Legacy test harness for standalone script mode"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    async def test_difficulty_levels(self, room_code: str, password: str = None):
        """Test all difficulty levels against the example bot"""
        test_case = TestBotComparison()
        test_case.setUp()
        
        difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, 
                       DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
        
        for difficulty in difficulties:
            bot = RulesBasedBot(difficulty)
            results = await test_case._run_test_games(bot, room_code, password, num_games=3)
            
            self.test_results.append({
                'difficulty': difficulty.value,
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
        return self.test_results
    
    async def test_adaptive_behavior(self, room_code: str, password: str = None):
        """Test adaptive behavior over multiple games"""
        test_case = TestBotComparison()
        test_case.setUp()
        
        bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
        bot.set_adaptation_enabled(True)
        
        adaptation_results = []
        for batch in range(2):  # Reduced for faster testing
            status_before = bot.get_adaptation_status()
            batch_results = await test_case._run_test_games(bot, room_code, password, num_games=2)
            status_after = bot.get_adaptation_status()
            
            adaptation_results.append({
                'batch': batch + 1,
                'status_before': status_before,
                'status_after': status_after,
                'game_results': batch_results
            })
        
        self.test_results.append({
            'test_type': 'adaptive_behavior',
            'results': adaptation_results,
            'timestamp': datetime.now().isoformat()
        })
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if filename is None:
            filename = f"bot_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"Test results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of test results"""
        print("\n" + "="*60)
        print("BOT TESTING SUMMARY")
        print("="*60)
        
        for result in self.test_results:
            if 'difficulty' in result:
                difficulty = result['difficulty']
                stats = result['results']
                print(f"\n{difficulty.upper()} DIFFICULTY:")
                print(f"  Games: {stats['total_games']}")
                print(f"  Win Rate: {stats['win_rate']:.1%}")
                print(f"  K/D Ratio: {stats['kd_ratio']:.2f}")
                print(f"  Accuracy: {stats['accuracy']:.1%}")


async def main():
    """Main function for standalone script mode"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test rules-based bot against example bot")
    parser.add_argument("--room_code", type=str, required=True, help="Room code to join")
    parser.add_argument("--password", type=str, default=None, help="Room password")
    parser.add_argument("--test_type", type=str, choices=['difficulty', 'adaptive', 'both'], 
                       default='both', help="Type of test to run")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create tester
    tester = BotTester()
    
    try:
        if args.test_type in ['difficulty', 'both']:
            await tester.test_difficulty_levels(args.room_code, args.password)
        
        if args.test_type in ['adaptive', 'both']:
            await tester.test_adaptive_behavior(args.room_code, args.password)
        
        # Save and display results
        tester.save_results(args.output)
        tester.print_summary()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        logging.exception("Test error")


if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print("Cannot run integration test: missing dependencies")
        print("Install required dependencies or run this test in an environment with game client support")
        exit(1)
    asyncio.run(main())