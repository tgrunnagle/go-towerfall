# Replay System

The replay system provides comprehensive episode recording, analysis, and experience replay functionality for the RL bot system.

## Features

### Episode Recording
- **EpisodeRecorder**: Records game episodes during training and evaluation
- **Configurable Storage**: Supports JSON and pickle formats with compression options
- **Session Management**: Organizes episodes into recording sessions with metadata
- **Automatic Cleanup**: Manages storage limits and removes old episodes

### Experience Replay
- **ExperienceBuffer**: Stores state-action-reward transitions for training
- **Batch Sampling**: Provides random and prioritized sampling for RL training
- **Episode Batches**: Samples complete episodes for sequence-based training
- **Memory Management**: Configurable buffer size with automatic cleanup

### Behavior Analysis
- **ReplayAnalyzer**: Detects behavior patterns and strategies in recorded episodes
- **Pattern Detection**: Identifies action sequences, state preferences, and strategic behaviors
- **Generation Comparison**: Compares behavior evolution across model generations
- **Performance Metrics**: Calculates win rates, rewards, and consistency metrics

### Export Functionality
- **Multiple Formats**: Export episodes and analysis results as JSON, CSV, HTML, or pickle
- **External Analysis**: Enables integration with external analysis tools
- **Report Generation**: Creates human-readable HTML reports with visualizations

## Components

### ReplayManager
Central coordinator for all replay functionality:
```python
from bot.rl_bot_system.replay import ReplayManager

manager = ReplayManager(storage_path="bot/data/replays")

# Start recording session
session_id = manager.start_session("training_session")

# Record episodes
manager.record_episode(states, actions, rewards, model_generation=1)

# Get training batches
batch = manager.get_training_batch(batch_size=32)

# Analyze episodes
analysis = manager.analyze_episodes(episodes)

# Export results
manager.export_episodes(episodes, "episodes.json", "json")
```

### EpisodeRecorder
Records episodes with configurable options:
```python
from bot.rl_bot_system.replay import EpisodeRecorder, RecordingConfig

config = RecordingConfig(
    storage_path="bot/data/replays",
    max_episodes_per_file=100,
    compression=True
)

recorder = EpisodeRecorder(config)
recorder.start_recording_session("session_1")
recorder.record_episode(states, actions, rewards, model_generation=1)
```

### ExperienceBuffer
Manages experience replay for training:
```python
from bot.rl_bot_system.replay import ExperienceBuffer, BufferConfig

config = BufferConfig(
    max_size=100000,
    min_size_for_sampling=1000,
    prioritized_replay=True
)

buffer = ExperienceBuffer(config)
buffer.add_episode(episode)
batch = buffer.sample_batch(32)
```

### ReplayAnalyzer
Analyzes behavior patterns:
```python
from bot.rl_bot_system.replay import ReplayAnalyzer, AnalysisConfig

config = AnalysisConfig(
    min_pattern_frequency=3,
    min_confidence_threshold=0.7
)

analyzer = ReplayAnalyzer(config)
analysis = analyzer.analyze_episodes(episodes)
comparison = analyzer.compare_generations(episodes_by_generation)
```

## Configuration Options

### RecordingConfig
- `storage_path`: Base directory for storing episodes
- `max_episodes_per_file`: Episodes per file before creating new file
- `compression`: Enable/disable compression for storage
- `record_states/actions/rewards`: Control what data to record
- `auto_cleanup`: Automatic cleanup of old files
- `max_storage_mb`: Maximum storage size before cleanup

### BufferConfig
- `max_size`: Maximum number of experiences to store
- `min_size_for_sampling`: Minimum experiences before sampling
- `prioritized_replay`: Enable prioritized experience replay
- `alpha/beta`: Prioritization parameters

### AnalysisConfig
- `min_pattern_frequency`: Minimum occurrences for pattern detection
- `min_confidence_threshold`: Minimum confidence for patterns
- `sequence_length`: Length of action sequences to analyze
- `analyze_*`: Enable/disable different analysis types

## Usage Examples

See `examples/example_replay_usage.py` for comprehensive usage examples including:
- Basic episode recording
- Experience replay for training
- Behavior pattern analysis
- Generation comparison
- Export functionality
- Session management

## Testing

Run the test suite:
```bash
python -m pytest bot/rl_bot_system/replay/tests/ -v
```

The test suite includes:
- Episode recording and storage tests
- Experience buffer functionality tests
- Replay analysis and pattern detection tests
- Export functionality tests
- Integration tests for the complete system

## Integration

The replay system integrates with:
- **Training Engine**: Records episodes during training
- **Evaluation Framework**: Analyzes model performance
- **Model Manager**: Tracks behavior evolution across generations
- **Game Environment**: Captures state-action-reward sequences

## File Structure

```
replay/
├── __init__.py                 # Package exports
├── replay_manager.py           # Main coordinator
├── episode_recorder.py         # Episode recording
├── experience_buffer.py        # Experience replay buffer
├── replay_analyzer.py          # Behavior analysis
├── examples/
│   └── example_replay_usage.py # Usage examples
└── tests/
    ├── test_episode_recorder.py
    ├── test_experience_buffer.py
    ├── test_replay_analyzer.py
    └── test_replay_manager.py
```