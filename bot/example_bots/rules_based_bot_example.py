#!/usr/bin/env python3
"""
Example implementation showing how to use the rules-based bot with the game client.
This demonstrates the bot working in a similar way to the existing example_bot.py.
"""

import asyncio
import argparse
import logging
import sys
import websockets
from core.game_client import GameClient
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel


class RulesBasedBotRunner:
    """Runner class that integrates RulesBasedBot with GameClient"""
    
    def __init__(self, client: GameClient, difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE):
        self.client = client
        self.bot = RulesBasedBot(difficulty)
        self.running = False
        self._logger = logging.getLogger(__name__)
        
        # Register message handler
        self.client.register_message_handler(self._handle_message)
        
        # Game state tracking
        self.current_game_state = None
        self.game_stats = {
            'shots_fired': 0,
            'shots_hit': 0,
            'kills': 0,
            'deaths': 0,
            'game_start_time': None
        }
    
    async def run(self):
        """Main bot loop"""
        self.running = True
        self.game_stats['game_start_time'] = asyncio.get_event_loop().time()
        
        self._logger.info(f"Starting rules-based bot with {self.bot.difficulty.value} difficulty")
        
        try:
            while self.running:
                # Get current game state
                game_state = self._get_current_game_state()
                
                if game_state:
                    # Analyze game state using the rules-based bot
                    analysis = self.bot.analyze_game_state(game_state)
                    
                    # Select action based on analysis
                    action = self.bot.select_action(analysis)
                    
                    # Execute the selected action
                    if action:
                        await self._execute_action(action)
                
                # Control loop frequency based on bot's decision frequency
                await asyncio.sleep(self.bot.config['decision_frequency'])
                
        except websockets.exceptions.ConnectionClosedOK:
            self._logger.info("Connection closed normally")
        except Exception as e:
            self._logger.exception("Bot error", e)
        finally:
            self.running = False
            await self._finalize_game()
    
    def stop(self):
        """Stop the bot"""
        self.running = False
    
    def _get_current_game_state(self):
        """
        Extract current game state from the game client.
        This is a simplified mock implementation - in a real scenario,
        this would extract actual game state from the client's received messages.
        """
        # Mock game state for demonstration
        # In a real implementation, this would be populated from actual game messages
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
                    'velocity': (5, 0),
                    'health': 80,
                    'hasLineOfSight': True
                }
            ],
            'projectiles': [],
            'powerUps': [
                {
                    'type': 'health',
                    'position': (150, 200)
                }
            ],
            'boundaries': {
                'left': -400,
                'right': 400,
                'top': -300,
                'bottom': 300
            }
        }
    
    async def _execute_action(self, action):
        """Execute the selected action through the game client"""
        try:
            action_type = action.action_type
            params = action.parameters
            
            if action_type in ['move_left', 'move_right', 'jump', 'crouch']:
                # Handle movement actions
                key = params.get('key')
                pressed = params.get('pressed', True)
                
                if key:
                    await self.client.send_keyboard_input(key, pressed)
                    self._logger.debug(f"Executed {action_type}: {key} {'pressed' if pressed else 'released'}")
                    
                    # Schedule key release if action has duration
                    if pressed and action.duration:
                        asyncio.create_task(self._release_key_after_delay(key, action.duration))
            
            elif action_type == 'evade_threat':
                # Handle evasive movement
                key = params.get('key')
                if key:
                    await self.client.send_keyboard_input(key, True)
                    self._logger.debug(f"Evading threat: {key}")
                    
                    if action.duration:
                        asyncio.create_task(self._release_key_after_delay(key, action.duration))
            
            elif action_type == 'shoot_at_enemy':
                # Handle shooting actions
                x = params.get('x', 0)
                y = params.get('y', 0)
                
                await self.client.send_mouse_input('left', True, x, y)
                self.game_stats['shots_fired'] += 1
                
                self._logger.debug(f"Shooting at ({x:.0f}, {y:.0f})")
                
                # Release mouse after short duration
                release_duration = action.duration or 0.1
                asyncio.create_task(self._release_mouse_after_delay('left', release_duration))
            
            elif action_type == 'move_to_opportunity':
                # Handle strategic movement
                key = params.get('key')
                if key:
                    await self.client.send_keyboard_input(key, True)
                    self._logger.debug(f"Moving to opportunity: {key}")
                    
                    if action.duration:
                        asyncio.create_task(self._release_key_after_delay(key, action.duration))
            
        except Exception as e:
            self._logger.error(f"Error executing action {action_type}: {e}")
    
    async def _release_key_after_delay(self, key: str, delay: float):
        """Release a key after specified delay"""
        await asyncio.sleep(delay)
        try:
            await self.client.send_keyboard_input(key, False)
        except Exception as e:
            self._logger.error(f"Error releasing key {key}: {e}")
    
    async def _release_mouse_after_delay(self, button: str, delay: float):
        """Release mouse button after specified delay"""
        await asyncio.sleep(delay)
        try:
            await self.client.send_mouse_input(button, False)
        except Exception as e:
            self._logger.error(f"Error releasing mouse {button}: {e}")
    
    async def _handle_message(self, message: dict) -> None:
        """Handle incoming game messages"""
        message_type = message.get('type', '')
        
        # Update game state based on messages
        if message_type == 'game_state_update':
            self.current_game_state = message.get('state')
        
        elif message_type == 'player_killed':
            victim = message.get('victim')
            killer = message.get('killer')
            
            if victim == 'rules_bot':
                self.game_stats['deaths'] += 1
                self._logger.info("Bot was killed")
            elif killer == 'rules_bot':
                self.game_stats['kills'] += 1
                self.game_stats['shots_hit'] += 1  # Assume killing shot hit
                self._logger.info("Bot killed an enemy")
        
        elif message_type == 'shot_hit':
            shooter = message.get('shooter')
            if shooter == 'rules_bot':
                self.game_stats['shots_hit'] += 1
        
        elif message_type == 'game_over':
            winner = message.get('winner')
            won = winner == 'rules_bot'
            
            self._logger.info(f"Game over - {'Won' if won else 'Lost'}")
            
            # Update bot performance metrics
            game_result = self._calculate_game_result(won)
            self.bot.update_performance_metrics(game_result)
            
            # Log performance
            self._log_performance()
            
            self.stop()
    
    def _calculate_game_result(self, won: bool) -> dict:
        """Calculate final game result for bot learning"""
        duration = 0
        if self.game_stats['game_start_time']:
            duration = asyncio.get_event_loop().time() - self.game_stats['game_start_time']
        
        return {
            'won': won,
            'kills': self.game_stats['kills'],
            'deaths': self.game_stats['deaths'],
            'shots_fired': self.game_stats['shots_fired'],
            'shots_hit': self.game_stats['shots_hit'],
            'duration': duration
        }
    
    def _log_performance(self):
        """Log current performance metrics"""
        metrics = self.bot.get_performance_metrics()
        adaptation_status = self.bot.get_adaptation_status()
        
        self._logger.info("=== PERFORMANCE SUMMARY ===")
        self._logger.info(f"Difficulty: {self.bot.difficulty.value}")
        self._logger.info(f"Total Games: {metrics['total_games']}")
        self._logger.info(f"Win Rate: {metrics['win_rate']:.1%}")
        self._logger.info(f"K/D Ratio: {metrics['kd_ratio']:.2f}")
        self._logger.info(f"Accuracy: {metrics['accuracy']:.1%}")
        self._logger.info(f"Current Strategy: {adaptation_status['current_strategy']}")
        
        if adaptation_status['config_changes']:
            self._logger.info("Adaptations made:")
            for param, change in adaptation_status['config_changes'].items():
                self._logger.info(f"  {param}: {change['base']:.3f} -> {change['current']:.3f}")
    
    async def _finalize_game(self):
        """Finalize game and clean up"""
        self._logger.info("Finalizing game...")
        
        # Log final adaptation status
        if self.bot.adaptation_enabled:
            status = self.bot.get_adaptation_status()
            self._logger.info(f"Final adaptation status: {status}")
    
    async def wait_for_exit(self):
        """Wait for user to exit"""
        sys.stdout.write("Press Enter to exit...")
        sys.stdout.flush()
        await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        await self.client.exit_game()


async def main(args):
    """Main function"""
    # Parse difficulty level
    difficulty_map = {
        'beginner': DifficultyLevel.BEGINNER,
        'intermediate': DifficultyLevel.INTERMEDIATE,
        'advanced': DifficultyLevel.ADVANCED,
        'expert': DifficultyLevel.EXPERT
    }
    
    difficulty = difficulty_map.get(args.difficulty.lower(), DifficultyLevel.INTERMEDIATE)
    
    # Create a game client
    client = GameClient()
    
    try:
        # Connect to a game room and get the listener task
        listener_task = await client.connect(
            room_code=args.room_code,
            player_name=f"RulesBot-{difficulty.value}",
            room_password=args.password,
        )
        
        # Create and run the rules-based bot
        bot_runner = RulesBasedBotRunner(client, difficulty)
        
        # Enable adaptation if requested
        if args.adaptive:
            bot_runner.bot.set_adaptation_enabled(True)
            print(f"Adaptive behavior enabled for {difficulty.value} bot")
        
        bot_task = asyncio.create_task(bot_runner.run())
        exit_task = asyncio.create_task(bot_runner.wait_for_exit())
        
        # Wait for any task to complete
        await asyncio.wait(
            [listener_task, bot_task, exit_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Rules-based bot for the game")
    parser.add_argument(
        "--player_name", type=str, default="RulesBot", 
        help="Name of the bot player (will be overridden by difficulty)"
    )
    parser.add_argument(
        "--room_code", type=str, required=True, 
        help="Room code to join"
    )
    parser.add_argument(
        "--password", type=str, default=None, 
        help="Optional room password"
    )
    parser.add_argument(
        "--difficulty", type=str, 
        choices=['beginner', 'intermediate', 'advanced', 'expert'],
        default='intermediate',
        help="Bot difficulty level"
    )
    parser.add_argument(
        "--adaptive", action='store_true',
        help="Enable adaptive behavior (bot learns from game outcomes)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting rules-based bot with {args.difficulty} difficulty")
    if args.adaptive:
        print("Adaptive behavior enabled - bot will learn from game outcomes")
    
    asyncio.run(main(args))