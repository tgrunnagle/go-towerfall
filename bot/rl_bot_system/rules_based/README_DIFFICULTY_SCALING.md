# Rules-Based Bot Difficulty Scaling and Adaptive Behavior

This document describes the difficulty scaling and adaptive behavior features implemented for the rules-based bot system.

## Overview

The rules-based bot now supports four difficulty levels with distinct behavioral characteristics and adaptive learning capabilities that allow the bot to adjust its strategy based on game outcomes.

## Difficulty Levels

### Beginner
- **Reaction Time**: 300ms delay
- **Accuracy**: 60% base accuracy with large aim errors (up to 45 pixels)
- **Decision Frequency**: Updates every 200ms
- **Aggression Level**: 30% (very conservative)
- **Strategic Depth**: Level 1 (simple strategies only)
- **Risk Tolerance**: 20% (very risk-averse)
- **Behavioral Traits**:
  - Sometimes holds keys too long or too short
  - 15% chance of suboptimal movement choices
  - 10% chance of complete aim misses
  - Large timing variations in actions

### Intermediate
- **Reaction Time**: 150ms delay
- **Accuracy**: 75% base accuracy with moderate aim errors
- **Decision Frequency**: Updates every 100ms
- **Aggression Level**: 50% (balanced)
- **Strategic Depth**: Level 2 (moderate strategies)
- **Risk Tolerance**: 40% (moderate risk-taking)
- **Behavioral Traits**:
  - Slight timing variations in movement
  - 8% chance of suboptimal choices
  - Moderate aim error with occasional precision

### Advanced
- **Reaction Time**: 80ms delay
- **Accuracy**: 85% base accuracy with small aim errors
- **Decision Frequency**: Updates every 50ms
- **Aggression Level**: 70% (aggressive)
- **Strategic Depth**: Level 3 (complex strategies)
- **Risk Tolerance**: 60% (willing to take risks)
- **Behavioral Traits**:
  - Precise movement timing
  - Small aim errors, mostly accurate
  - Quick decision making

### Expert
- **Reaction Time**: 30ms delay
- **Accuracy**: 95% base accuracy with minimal aim errors
- **Decision Frequency**: Updates every 20ms
- **Aggression Level**: 80% (very aggressive)
- **Strategic Depth**: Level 4 (advanced strategies)
- **Risk Tolerance**: 70% (high risk tolerance)
- **Behavioral Traits**:
  - Near-perfect timing and accuracy
  - Very precise aiming
  - Rapid decision making
  - Advanced tactical awareness

## Adaptive Behavior

The bot can learn and adapt its behavior based on recent game outcomes when adaptive behavior is enabled.

### Adaptation Mechanisms

#### Strategy Adaptation
- **Poor Performance** (win rate < 30%): Switches to more defensive strategies
  - Aggressive → Balanced → Defensive
  - Reduces aggression level by 10%
- **Excellent Performance** (win rate > 70%): Switches to more aggressive strategies
  - Defensive → Balanced → Aggressive
  - Increases aggression level by 10%

#### Accuracy-Based Adaptation
- **Poor Accuracy** (< 30%): Slows down reactions to improve accuracy
  - Increases reaction time by up to 50%
  - Increases decision frequency (slower decisions)
- **High Accuracy** (> 80%): Speeds up reactions
  - Decreases reaction time by up to 20%
  - Decreases decision frequency (faster decisions)

#### Risk Tolerance Adaptation
- **High Death Rate** (K/D < 0.5): Reduces risk tolerance by 10%
- **Low Death Rate** (K/D > 2.0): Increases risk tolerance by 10%

### Adaptation Configuration

```python
# Enable/disable adaptive behavior
bot.set_adaptation_enabled(True)

# Reset adaptations to base configuration
bot.reset_adaptation()

# Get current adaptation status
status = bot.get_adaptation_status()
```

### Adaptation Status

The adaptation system tracks:
- Number of games played
- Recent win rate (last 5 games)
- Current strategy (aggressive/balanced/defensive)
- Configuration changes from base values
- Game history for learning

## Usage Examples

### Basic Usage with Difficulty Levels

```python
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot, DifficultyLevel

# Create bot with specific difficulty
bot = RulesBasedBot(DifficultyLevel.ADVANCED)

# Change difficulty during runtime
bot.set_difficulty_level(DifficultyLevel.EXPERT)

# Get current difficulty
current_difficulty = bot.get_difficulty_level()
```

### Adaptive Behavior Usage

```python
# Create bot with adaptation enabled
bot = RulesBasedBot(DifficultyLevel.INTERMEDIATE)
bot.set_adaptation_enabled(True)

# After each game, update performance metrics
game_result = {
    'won': True,
    'kills': 3,
    'deaths': 1,
    'shots_fired': 15,
    'shots_hit': 12,
    'duration': 120.5
}
bot.update_performance_metrics(game_result)

# Check adaptation status
status = bot.get_adaptation_status()
print(f"Current strategy: {status['current_strategy']}")
print(f"Recent win rate: {status['recent_win_rate']:.1%}")
print(f"Config changes: {status['config_changes']}")
```

### Running the Bot

```bash
# Run with specific difficulty
python rules_based_bot_example.py --room_code ABC123 --difficulty expert

# Run with adaptive behavior enabled
python rules_based_bot_example.py --room_code ABC123 --difficulty intermediate --adaptive

# Run unit tests using the test runner
python run_test.py rl_bot_system/rules_based/tests/test_difficulty_scaling.py

# Run integration tests (requires game client dependencies)
python rl_bot_system/rules_based/tests/test_bot_comparison.py --room_code ABC123
```

## Testing

### Unit Tests

The `tests/test_difficulty_scaling.py` script includes comprehensive tests for:

- **Difficulty Configuration**: Verifies that different difficulty levels have properly scaled parameters
- **Difficulty Switching**: Tests runtime difficulty level changes
- **Shooting Accuracy**: Validates that aim accuracy varies by difficulty level
- **Reaction Time Delays**: Confirms reaction time delays work correctly
- **Adaptive Behavior**: Tests strategy and configuration adaptation
- **Performance Tracking**: Verifies game outcome tracking and metrics calculation

### Integration Tests

The `tests/test_bot_comparison.py` script provides integration testing against other bots:

- **Difficulty Level Testing**: Runs each difficulty level against example bot
- **Adaptive Behavior Testing**: Extended testing to observe adaptation over time
- **Performance Comparison**: Statistical analysis of bot performance

**Note**: Integration tests require game client dependencies and should be run as standalone scripts if dependencies are not available in the test environment.

### Performance Comparison

Example output from difficulty scaling tests:

```
PERFORMANCE COMPARISON
============================================================
    BEGINNER: Reaction: 300ms, Accuracy: 60.0%, Decisions/sec: 12.0
INTERMEDIATE: Reaction: 150ms, Accuracy: 75.0%, Decisions/sec: 30.0
    ADVANCED: Reaction:  80ms, Accuracy: 85.0%, Decisions/sec: 10.0
      EXPERT: Reaction:  30ms, Accuracy: 95.0%, Decisions/sec: 22.0
```

## Implementation Details

### Difficulty Modifiers

The bot applies difficulty-specific modifiers to actions:

- **Shooting Actions**: Aim error calculation based on difficulty level
- **Movement Actions**: Timing variations and suboptimal choice probability
- **Reaction Delays**: Additional delays for lower difficulty levels
- **Confidence Scaling**: Action confidence modified by accuracy parameters

### Adaptive Learning

The adaptation system uses a sliding window approach:
- Tracks last 10 games for adaptation decisions
- Evaluates performance every 3-5 games
- Makes gradual adjustments to avoid oscillation
- Maintains base configuration for reset capability

### Thread Safety

The bot implementation is designed for single-threaded async operation:
- All state modifications happen in the main event loop
- No shared mutable state between concurrent operations
- Async-safe logging and error handling

## Future Enhancements

Potential improvements for the difficulty scaling and adaptive behavior system:

1. **Machine Learning Integration**: Use actual ML models to predict optimal parameter adjustments
2. **Opponent-Specific Adaptation**: Adapt differently based on opponent behavior patterns
3. **Multi-Objective Optimization**: Balance multiple performance metrics simultaneously
4. **Curriculum Learning**: Gradually increase difficulty as bot improves
5. **Behavioral Diversity**: Maintain multiple behavioral profiles for variety

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **Requirement 1.1**: Bot can play games autonomously with varying skill levels
- **Requirement 4.1**: Multiple difficulty levels provide appropriate challenges
- **Requirement 4.4**: Difficulty levels can be adjusted and bot suggests appropriate levels

The adaptive behavior system enables the bot to:
- Learn from game outcomes and adjust strategy
- Modify reaction times and accuracy based on performance
- Maintain performance metrics for analysis
- Reset adaptations when needed