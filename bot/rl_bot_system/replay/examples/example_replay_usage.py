"""
Example usage of the replay system for episode recording, analysis, and experience replay.
"""

import random
from pathlib import Path

from bot.rl_bot_system.replay.replay_manager import ReplayManager
from bot.rl_bot_system.replay.episode_recorder import RecordingConfig
from bot.rl_bot_system.replay.experience_buffer import BufferConfig
from bot.rl_bot_system.replay.replay_analyzer import AnalysisConfig
from bot.rl_bot_system.evaluation.evaluator import GameEpisode


def create_sample_episode(episode_id: str, model_generation: int, episode_length: int = 10) -> GameEpisode:
    """Create a sample episode for demonstration."""
    
    # Simulate game states (simplified)
    states = []
    actions = []
    rewards = []
    
    for step in range(episode_length):
        # Sample state (position, health, etc.)
        state = {
            "player_x": random.uniform(0, 100),
            "player_y": random.uniform(0, 100),
            "health": max(0, 100 - step * 5),
            "enemies_nearby": random.randint(0, 3),
            "step": step
        }
        states.append(state)
        
        # Sample action (0=move_left, 1=move_right, 2=shoot, 3=jump)
        action = random.randint(0, 3)
        actions.append(action)
        
        # Sample reward based on action and state
        if action == 2 and state["enemies_nearby"] > 0:  # Shooting with enemies nearby
            reward = 10.0
        elif action == 3 and state["health"] < 50:  # Jumping when low health (escape)
            reward = 5.0
        elif state["health"] > 80:  # Staying healthy
            reward = 1.0
        else:
            reward = 0.1
            
        rewards.append(reward)
    
    # Determine game result based on final health
    final_health = states[-1]["health"]
    if final_health > 50:
        game_result = "win"
    elif final_health > 0:
        game_result = "draw"
    else:
        game_result = "loss"
    
    return GameEpisode(
        episode_id=episode_id,
        model_generation=model_generation,
        opponent_generation=max(0, model_generation - 1),
        states=states,
        actions=actions,
        rewards=rewards,
        total_reward=sum(rewards),
        episode_length=len(states),
        game_result=game_result,
        episode_metrics={
            "final_health": final_health,
            "shots_fired": sum(1 for a in actions if a == 2),
            "jumps": sum(1 for a in actions if a == 3)
        }
    )


def demonstrate_basic_recording():
    """Demonstrate basic episode recording functionality."""
    print("=== Basic Episode Recording Demo ===")
    
    # Configure replay system
    recording_config = RecordingConfig(
        storage_path="bot/data/examples/replays",
        max_episodes_per_file=5,
        compression=True
    )
    
    buffer_config = BufferConfig(
        max_size=1000,
        min_size_for_sampling=10
    )
    
    # Create replay manager
    manager = ReplayManager(
        storage_path="bot/data/examples/replays",
        recording_config=recording_config,
        buffer_config=buffer_config
    )
    
    # Start recording session
    session_id = manager.start_session(
        "basic_demo_session",
        {"experiment": "basic_demo", "model_type": "DQN"}
    )
    print(f"Started recording session: {session_id}")
    
    # Record some episodes
    episodes = []
    for i in range(15):
        episode = create_sample_episode(f"demo_ep_{i}", model_generation=1, episode_length=8)
        manager.record_game_episode(episode, add_to_buffer=True)
        episodes.append(episode)
        print(f"Recorded episode {i+1}/15")
    
    # Get system statistics
    stats = manager.get_system_stats()
    print(f"\nSystem Stats:")
    print(f"  Episodes in buffer: {stats['buffer_stats']['total_experiences']}")
    print(f"  Current session: {stats['current_session']}")
    
    # End session
    session_summary = manager.end_session()
    print(f"\nSession Summary:")
    print(f"  Total episodes: {session_summary['total_episodes']}")
    print(f"  Duration: {session_summary['duration_seconds']:.2f} seconds")
    
    return manager, episodes


def demonstrate_experience_replay():
    """Demonstrate experience replay for training."""
    print("\n=== Experience Replay Demo ===")
    
    manager, episodes = demonstrate_basic_recording()
    
    # Add episodes back to buffer for demonstration
    for episode in episodes:
        manager.experience_buffer.add_episode(episode)
    
    print(f"Buffer has {len(manager.experience_buffer)} experiences")
    
    # Sample training batches
    try:
        batch = manager.get_training_batch(batch_size=8)
        print(f"Sampled training batch with {len(batch)} experiences")
        
        # Show sample transition
        sample_transition = batch[0]
        print(f"Sample transition:")
        print(f"  State: {sample_transition.state}")
        print(f"  Action: {sample_transition.action}")
        print(f"  Reward: {sample_transition.reward}")
        print(f"  Done: {sample_transition.done}")
        
        # Sample episode batches
        episode_batch = manager.get_episode_batch(num_episodes=3)
        print(f"\nSampled {len(episode_batch)} complete episodes")
        for i, episode_transitions in enumerate(episode_batch):
            print(f"  Episode {i+1}: {len(episode_transitions)} transitions")
            
    except ValueError as e:
        print(f"Could not sample batch: {e}")
    
    return manager


def demonstrate_replay_analysis():
    """Demonstrate replay analysis and pattern detection."""
    print("\n=== Replay Analysis Demo ===")
    
    manager = demonstrate_experience_replay()
    
    # Create episodes with more obvious patterns for analysis
    pattern_episodes = []
    
    # Generation 1: Simple strategy (mostly shooting)
    for i in range(10):
        episode_length = random.randint(8, 12)
        states = []
        actions = []
        rewards = []
        
        for step in range(episode_length):
            state = {
                "player_x": random.uniform(40, 60),  # Stay in center
                "player_y": random.uniform(40, 60),
                "health": max(20, 100 - step * 3),
                "enemies_nearby": random.randint(1, 2),
                "step": step
            }
            states.append(state)
            
            # Generation 1 prefers shooting (action 2)
            if random.random() < 0.7:
                action = 2  # Shoot
            else:
                action = random.randint(0, 3)
            actions.append(action)
            
            # Reward shooting
            reward = 10.0 if action == 2 else 1.0
            rewards.append(reward)
        
        episode = GameEpisode(
            episode_id=f"gen1_ep_{i}",
            model_generation=1,
            opponent_generation=0,
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=sum(rewards),
            episode_length=len(states),
            game_result="win" if sum(rewards) > 50 else "loss",
            episode_metrics={}
        )
        pattern_episodes.append(episode)
    
    # Generation 2: More balanced strategy
    for i in range(10):
        episode_length = random.randint(10, 15)
        states = []
        actions = []
        rewards = []
        
        for step in range(episode_length):
            state = {
                "player_x": random.uniform(20, 80),  # More movement
                "player_y": random.uniform(20, 80),
                "health": max(30, 100 - step * 2),
                "enemies_nearby": random.randint(0, 3),
                "step": step
            }
            states.append(state)
            
            # Generation 2 uses more varied actions
            if state["health"] < 40:
                action = 3  # Jump (escape)
            elif state["enemies_nearby"] > 1:
                action = 2  # Shoot
            else:
                action = random.randint(0, 1)  # Move
            actions.append(action)
            
            # More complex reward
            if action == 2 and state["enemies_nearby"] > 0:
                reward = 8.0
            elif action == 3 and state["health"] < 50:
                reward = 6.0
            else:
                reward = 2.0
            rewards.append(reward)
        
        episode = GameEpisode(
            episode_id=f"gen2_ep_{i}",
            model_generation=2,
            opponent_generation=1,
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=sum(rewards),
            episode_length=len(states),
            game_result="win" if sum(rewards) > 60 else "loss",
            episode_metrics={}
        )
        pattern_episodes.append(episode)
    
    # Analyze all episodes
    print("Analyzing episodes for behavior patterns...")
    analysis_result = manager.analyze_episodes(pattern_episodes)
    
    print(f"\nAnalysis Results:")
    print(f"  Episodes analyzed: {analysis_result['episode_count']}")
    print(f"  Patterns detected: {len(analysis_result['patterns'])}")
    
    # Show detected patterns
    for i, pattern in enumerate(analysis_result['patterns'][:5]):  # Show first 5 patterns
        print(f"\nPattern {i+1}:")
        print(f"  Type: {pattern['pattern_type']}")
        print(f"  Description: {pattern['description']}")
        print(f"  Frequency: {pattern['frequency']}")
        print(f"  Confidence: {pattern['confidence']:.2f}")
    
    # Performance statistics
    perf_stats = analysis_result['performance_stats']
    print(f"\nPerformance Statistics:")
    print(f"  Win rate: {perf_stats['win_rate']:.2%}")
    print(f"  Average reward: {perf_stats['average_reward']:.2f}")
    print(f"  Average episode length: {perf_stats['average_episode_length']:.1f}")
    
    # Compare generations
    episodes_by_generation = {
        1: [ep for ep in pattern_episodes if ep.model_generation == 1],
        2: [ep for ep in pattern_episodes if ep.model_generation == 2]
    }
    
    print("\nComparing generations...")
    comparison_result = manager.compare_generations(episodes_by_generation)
    
    print(f"Generations analyzed: {comparison_result['generations_analyzed']}")
    
    # Show performance progression
    perf_progression = comparison_result['performance_progression']
    for gen, stats in perf_progression.items():
        print(f"\nGeneration {gen}:")
        print(f"  Win rate: {stats['win_rate']:.2%}")
        print(f"  Average reward: {stats['average_reward']:.2f}")
    
    return manager, analysis_result


def demonstrate_export_functionality():
    """Demonstrate exporting episodes and analysis results."""
    print("\n=== Export Functionality Demo ===")
    
    manager, analysis_result = demonstrate_replay_analysis()
    
    # Create some episodes for export
    export_episodes = []
    for i in range(5):
        episode = create_sample_episode(f"export_ep_{i}", model_generation=1)
        export_episodes.append(episode)
    
    # Export episodes in different formats
    export_dir = Path("bot/data/examples/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export as JSON
    json_path = manager.export_episodes(
        export_episodes, 
        str(export_dir / "episodes_export"), 
        "json"
    )
    print(f"Exported episodes to JSON: {json_path}")
    
    # Export as CSV
    csv_path = manager.export_episodes(
        export_episodes, 
        str(export_dir / "episodes_export"), 
        "csv"
    )
    print(f"Exported episodes to CSV: {csv_path}")
    
    # Export analysis results
    analysis_json_path = manager.export_analysis(
        analysis_result,
        str(export_dir / "analysis_export"),
        "json"
    )
    print(f"Exported analysis to JSON: {analysis_json_path}")
    
    analysis_html_path = manager.export_analysis(
        analysis_result,
        str(export_dir / "analysis_export"),
        "html"
    )
    print(f"Exported analysis to HTML: {analysis_html_path}")
    
    print(f"\nAll exports saved to: {export_dir}")


def demonstrate_session_management():
    """Demonstrate session management and loading."""
    print("\n=== Session Management Demo ===")
    
    manager = ReplayManager(storage_path="bot/data/examples/replays")
    
    # Get available sessions
    available_sessions = manager.get_available_sessions()
    print(f"Available sessions: {len(available_sessions)}")
    
    for session in available_sessions[:3]:  # Show first 3 sessions
        print(f"  Session: {session['session_id']}")
        print(f"    Start time: {session.get('start_time', 'Unknown')}")
        if 'user_metadata' in session:
            print(f"    Metadata: {session['user_metadata']}")
    
    # Load episodes from a session (if any exist)
    if available_sessions:
        session_id = available_sessions[0]['session_id']
        try:
            episodes = manager.load_episodes_from_session(session_id)
            print(f"\nLoaded {len(episodes)} episodes from session {session_id}")
            
            if episodes:
                # Analyze loaded episodes
                analysis = manager.analyze_session(session_id)
                print(f"Session analysis - Episodes: {analysis['episode_count']}, Patterns: {len(analysis['patterns'])}")
                
        except Exception as e:
            print(f"Could not load session {session_id}: {e}")


def main():
    """Run all replay system demonstrations."""
    print("Replay System Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_basic_recording()
        demonstrate_experience_replay()
        demonstrate_replay_analysis()
        demonstrate_export_functionality()
        demonstrate_session_management()
        
        print("\n" + "=" * 50)
        print("Replay system demonstration completed successfully!")
        print("\nKey features demonstrated:")
        print("- Episode recording and storage")
        print("- Experience buffer for training batch retrieval")
        print("- Behavior pattern analysis")
        print("- Generation comparison")
        print("- Export functionality (JSON, CSV, HTML)")
        print("- Session management and loading")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()