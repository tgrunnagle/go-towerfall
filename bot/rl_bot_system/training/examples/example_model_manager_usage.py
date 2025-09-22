"""
Example usage of the ModelManager class with dummy training and inference.

This demonstrates how to use the ModelManager for model lifecycle management,
versioning, knowledge transfer, and promotion logic, including realistic
training and inference workflows.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from typing import Tuple, List, Dict, Any
from bot.rl_bot_system.training.model_manager import ModelManager, RLModel


class ExampleRLModel(nn.Module):
    """Example RL model for demonstration."""
    
    def __init__(self, input_size: int = 64, hidden_size: int = 128, output_size: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class DummyGameEnvironment:
    """Dummy game environment for RL training simulation."""
    
    def __init__(self, state_size: int = 64, action_size: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.step_count = 0
        self.max_steps = 200
        self.state = np.random.randn(self.state_size).astype(np.float32)
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return next state, reward, done, info."""
        self.step_count += 1
        
        # Simulate state transition (random walk with action influence)
        action_effect = np.zeros(self.state_size)
        if action < self.action_size:
            action_effect[action % self.state_size] = 0.5
        
        self.state = 0.9 * self.state + 0.1 * np.random.randn(self.state_size) + 0.1 * action_effect
        self.state = np.clip(self.state, -3, 3).astype(np.float32)
        
        # Simulate reward (higher for certain state patterns)
        reward = float(np.sum(self.state[:4]) * 0.1 + np.random.normal(0, 0.1))
        
        # Episode ends after max steps or if state goes extreme
        done = self.step_count >= self.max_steps or np.abs(self.state).max() > 2.5
        
        info = {
            'step_count': self.step_count,
            'state_magnitude': float(np.linalg.norm(self.state))
        }
        
        return self.state, reward, done, info


class DummyRLTrainer:
    """Dummy RL trainer that simulates training process."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1  # For epsilon-greedy exploration
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return int(q_values.argmax().item())
    
    def train_episode(self, env: DummyGameEnvironment) -> Tuple[float, int, float]:
        """Train for one episode and return total reward, steps, loss."""
        state = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        
        # Collect experience for the episode
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while True:
            action = self.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state.copy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.copy())
            dones.append(done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Train on collected experience (simplified Q-learning update)
        if len(states) > 0:
            batch_loss = self._update_model(states, actions, rewards, next_states, dones)
            total_loss = batch_loss
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return total_reward, steps, total_loss
    
    def _update_model(self, states: List[np.ndarray], actions: List[int], 
                     rewards: List[float], next_states: List[np.ndarray], 
                     dones: List[bool]) -> float:
        """Update model using collected experience."""
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Next Q values (for Q-learning target)
        with torch.no_grad():
            next_q_values = self.model(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + 0.99 * next_q_values * (~dones_tensor)
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    def evaluate(self, env: DummyGameEnvironment, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_rewards = []
        win_count = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            
            while True:
                action = self.select_action(state, training=False)
                state, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            # Consider it a "win" if reward is above a threshold
            if total_reward > 5.0:
                win_count += 1
        
        self.model.train()
        
        return {
            'average_reward': float(np.mean(total_rewards)),
            'win_rate': win_count / num_episodes,
            'episodes_won': win_count,
            'reward_std': float(np.std(total_rewards))
        }


def train_model(model: nn.Module, generation: int, parent_generation: int = None, 
                num_episodes: int = 1000) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a model and return it with performance metrics."""
    print(f"  Training generation {generation} for {num_episodes} episodes...")
    
    env = DummyGameEnvironment()
    trainer = DummyRLTrainer(model, learning_rate=0.001)
    
    # Training loop
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        reward, steps, loss = trainer.train_episode(env)
        episode_rewards.append(reward)
        episode_losses.append(loss)
        
        # Print progress occasionally
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"    Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
    
    # Final evaluation
    print("  Evaluating trained model...")
    eval_metrics = trainer.evaluate(env, num_episodes=50)
    
    # Compile training metadata
    metadata = {
        "algorithm": "DQN",
        "network_architecture": {
            "input_size": 64,
            "hidden_size": model.fc1.out_features,
            "output_size": 4
        },
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 1,  # Online learning in this example
            "gamma": 0.99,
            "epsilon_start": 0.1,
            "epsilon_end": trainer.epsilon
        },
        "training_episodes": num_episodes,
        "performance_metrics": {
            **eval_metrics,
            "final_training_reward": float(np.mean(episode_rewards[-100:])),
            "training_loss": float(np.mean(episode_losses[-100:])) if episode_losses else 0.0
        },
        "parent_generation": parent_generation
    }
    
    print(f"  Training complete! Win rate: {eval_metrics['win_rate']:.1%}, "
          f"Avg reward: {eval_metrics['average_reward']:.2f}")
    
    return model, metadata


def demonstrate_inference(model_manager: ModelManager, generation: int):
    """Demonstrate model inference with the trained model."""
    print(f"\n--- Demonstrating Inference with Generation {generation} ---")
    
    # Load the model
    model, metadata = model_manager.load_model(generation, ExampleRLModel)
    model.eval()
    
    # Create environment for inference
    env = DummyGameEnvironment()
    
    # Run a few inference episodes
    print("Running inference episodes...")
    for episode in range(3):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        actions_taken = []
        
        print(f"\nEpisode {episode + 1}:")
        
        while steps < 20:  # Limit steps for demo
            # Get Q-values from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = int(q_values.argmax().item())
                action_confidence = float(torch.softmax(q_values, dim=1).max().item())
            
            # Take action
            next_state, reward, done, info = env.step(action)
            actions_taken.append(action)
            total_reward += reward
            steps += 1
            
            if steps <= 5:  # Show first few steps
                print(f"  Step {steps}: Action={action} (conf={action_confidence:.2f}), "
                      f"Reward={reward:.2f}, State_norm={info['state_magnitude']:.2f}")
            
            state = next_state
            if done:
                break
        
        print(f"  Episode {episode + 1} complete: {steps} steps, "
              f"Total reward: {total_reward:.2f}")
        print(f"  Actions taken: {actions_taken[:10]}{'...' if len(actions_taken) > 10 else ''}")
    
    print(f"Inference demonstration complete for generation {generation}")


def compare_model_performance(model_manager: ModelManager, gen_a: int, gen_b: int):
    """Compare performance between two model generations through actual gameplay."""
    print(f"\n--- Performance Comparison: Generation {gen_a} vs {gen_b} ---")
    
    # Load both models
    model_a, metadata_a = model_manager.load_model(gen_a, ExampleRLModel)
    model_b, metadata_b = model_manager.load_model(gen_b, ExampleRLModel)
    
    env = DummyGameEnvironment()
    num_games = 20
    
    print(f"Running {num_games} evaluation games for each model...")
    
    # Evaluate model A
    model_a.eval()
    rewards_a = []
    for _ in range(num_games):
        state = env.reset()
        total_reward = 0.0
        
        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model_a(state_tensor)
                action = int(q_values.argmax().item())
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        rewards_a.append(total_reward)
    
    # Evaluate model B
    model_b.eval()
    rewards_b = []
    for _ in range(num_games):
        state = env.reset()
        total_reward = 0.0
        
        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model_b(state_tensor)
                action = int(q_values.argmax().item())
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        rewards_b.append(total_reward)
    
    # Compare results
    avg_reward_a = np.mean(rewards_a)
    avg_reward_b = np.mean(rewards_b)
    wins_a = sum(1 for r in rewards_a if r > 5.0)
    wins_b = sum(1 for r in rewards_b if r > 5.0)
    
    print(f"\nPerformance Comparison Results:")
    print(f"  Generation {gen_a}: Avg reward = {avg_reward_a:.2f}, Wins = {wins_a}/{num_games}")
    print(f"  Generation {gen_b}: Avg reward = {avg_reward_b:.2f}, Wins = {wins_b}/{num_games}")
    
    improvement = ((avg_reward_b - avg_reward_a) / abs(avg_reward_a)) * 100 if avg_reward_a != 0 else 0
    print(f"  Improvement: {improvement:+.1f}% reward, {wins_b - wins_a:+d} more wins")
    
    return {
        'gen_a_avg_reward': avg_reward_a,
        'gen_b_avg_reward': avg_reward_b,
        'gen_a_wins': wins_a,
        'gen_b_wins': wins_b,
        'improvement_pct': improvement
    }


def main():
    """Demonstrate ModelManager usage with actual training and inference."""
    
    # Initialize ModelManager
    print("=== ModelManager Example with Training & Inference ===\n")
    
    # Use a temporary directory for this example
    model_manager = ModelManager(models_dir="data/examples/models")
    print(f"Initialized ModelManager with directory: {model_manager.models_dir}")
    
    # Train and save first generation model
    print("\n1. Training first generation model from scratch...")
    
    model_gen1 = ExampleRLModel(input_size=64, hidden_size=128, output_size=4)
    model_gen1, metadata_gen1 = train_model(model_gen1, generation=1, num_episodes=800)
    
    rl_model_gen1 = model_manager.save_model(model_gen1, 1, metadata_gen1)
    print(f"Saved generation 1: {rl_model_gen1.algorithm} model")
    print(f"Performance: {rl_model_gen1.performance_metrics['win_rate']:.1%} win rate, "
          f"{rl_model_gen1.performance_metrics['average_reward']:.2f} avg reward")
    
    # Promote as current best
    model_manager.promote_model(1)
    print("Promoted generation 1 as current best")
    
    # Demonstrate inference with generation 1
    demonstrate_inference(model_manager, 1)
    
    # Train second generation with knowledge transfer
    print("\n2. Training second generation with knowledge transfer...")
    
    model_gen2 = ExampleRLModel(input_size=64, hidden_size=128, output_size=4)
    
    # Transfer knowledge from generation 1
    model_gen2 = model_manager.transfer_knowledge(1, model_gen2, "weight_copy", ExampleRLModel)
    print("Transferred knowledge from generation 1 to generation 2")
    
    # Train the model (should converge faster due to knowledge transfer)
    model_gen2, metadata_gen2 = train_model(model_gen2, generation=2, parent_generation=1, num_episodes=600)
    
    rl_model_gen2 = model_manager.save_model(model_gen2, 2, metadata_gen2)
    print(f"Saved generation 2: {rl_model_gen2.training_episodes} episodes")
    print(f"Performance: {rl_model_gen2.performance_metrics['win_rate']:.1%} win rate, "
          f"{rl_model_gen2.performance_metrics['average_reward']:.2f} avg reward")
    
    # Compare models using ModelManager's comparison
    print("\n3. Comparing model generations (metadata-based)...")
    
    comparison = model_manager.compare_models(1, 2)
    win_rate_improvement = comparison['improvements']['win_rate']['percentage']
    reward_improvement = comparison['improvements']['average_reward']['percentage']
    
    print(f"Generation 2 vs Generation 1 (from saved metrics):")
    print(f"  Win rate improvement: {win_rate_improvement:.1f}%")
    print(f"  Average reward improvement: {reward_improvement:.1f}%")
    print(f"  Overall better: {comparison['summary']['overall_better']}")
    
    # Compare models through actual gameplay
    gameplay_comparison = compare_model_performance(model_manager, 1, 2)
    
    # Check if should promote
    should_promote, details = model_manager.should_promote_model(2)
    print(f"\nShould promote generation 2? {should_promote}")
    print(f"Reason: {details['promotion_decision']['reason']}")
    
    if should_promote:
        model_manager.promote_model(2)
        print("Promoted generation 2 as new best model")
    
    # Train third generation with improved architecture
    print("\n4. Training third generation with improved architecture...")
    
    model_gen3 = ExampleRLModel(input_size=64, hidden_size=256, output_size=4)  # Larger network
    
    # Transfer knowledge with layer freezing (some layers frozen, others trainable)
    model_gen3 = model_manager.transfer_knowledge(2, model_gen3, "layer_freeze", ExampleRLModel)
    print("Transferred knowledge with layer freezing from generation 2")
    
    # Train the model (frozen layers should help with stability)
    model_gen3, metadata_gen3 = train_model(model_gen3, generation=3, parent_generation=2, num_episodes=1000)
    
    rl_model_gen3 = model_manager.save_model(model_gen3, 3, metadata_gen3)
    print(f"Saved generation 3: {rl_model_gen3.algorithm} model")
    print(f"Performance: {rl_model_gen3.performance_metrics['win_rate']:.1%} win rate, "
          f"{rl_model_gen3.performance_metrics['average_reward']:.2f} avg reward")
    
    # Demonstrate inference with the latest generation
    demonstrate_inference(model_manager, 3)
    
    # Compare all three generations
    print("\n5. Three-way performance comparison...")
    
    print("Generation 1 vs 2:")
    compare_model_performance(model_manager, 1, 2)
    
    print("Generation 2 vs 3:")
    compare_model_performance(model_manager, 2, 3)
    
    print("Generation 1 vs 3:")
    compare_model_performance(model_manager, 1, 3)
    
    # List all models
    print("\n6. Listing all trained models...")
    
    models = model_manager.list_models()
    print(f"Total models: {len(models)}")
    
    for generation, rl_model in models:
        print(f"  Generation {generation}: {rl_model.algorithm}, "
              f"{rl_model.performance_metrics['win_rate']:.1%} win rate, "
              f"{rl_model.performance_metrics['average_reward']:.2f} avg reward, "
              f"{rl_model.training_episodes} episodes")
    
    # Get current best model
    current_best = model_manager.get_best_model()
    if current_best:
        best_gen, best_model = current_best
        print(f"\nCurrent best model: Generation {best_gen}")
        print(f"  Algorithm: {best_model.algorithm}")
        print(f"  Win rate: {best_model.performance_metrics['win_rate']:.1%}")
        print(f"  Average reward: {best_model.performance_metrics['average_reward']:.2f}")
    
    # Get detailed model info
    print("\n7. Getting detailed model information...")
    
    info = model_manager.get_model_info(3)
    print(f"Generation 3 details:")
    print(f"  Created: {info['created_at']}")
    print(f"  Model size: {info['model_size_mb']} MB")
    print(f"  Parent generation: {info['parent_generation']}")
    print(f"  Children: {info['children']}")
    print(f"  Training episodes: {info['training_episodes']}")
    print(f"  Network architecture: {info['network_architecture']}")
    
    # Demonstrate model loading and reuse
    print("\n8. Demonstrating model loading and reuse...")
    
    # Load the best model for continued training or deployment
    best_generation, _ = model_manager.get_best_model()
    loaded_model, loaded_metadata = model_manager.load_model(best_generation, ExampleRLModel)
    
    print(f"Loaded generation {best_generation} for reuse")
    print(f"  Model parameters: {sum(p.numel() for p in loaded_model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in loaded_model.parameters() if p.requires_grad):,}")
    
    # Quick evaluation of loaded model
    env = DummyGameEnvironment()
    trainer = DummyRLTrainer(loaded_model)
    eval_results = trainer.evaluate(env, num_episodes=10)
    print(f"  Quick evaluation: {eval_results['win_rate']:.1%} win rate, "
          f"{eval_results['average_reward']:.2f} avg reward")
    
    print("\n=== Complete Training & Inference Example Finished ===")
    print("\nThis example demonstrated:")
    print("  ✓ Training RL models from scratch")
    print("  ✓ Knowledge transfer between generations")
    print("  ✓ Model inference and action selection")
    print("  ✓ Performance comparison through gameplay")
    print("  ✓ Model management and versioning")
    print("  ✓ Model loading and reuse")


if __name__ == "__main__":
    main()