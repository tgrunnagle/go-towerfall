"""
Example usage of the training metrics server.

This script demonstrates how to set up and use the FastAPI training metrics
server for real-time training data collection and spectator functionality.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from bot.rl_bot_system.server.server import UnifiedServer as TrainingMetricsServer, ServerConfig
from bot.rl_bot_system.server.data_models import (
    TrainingMetricsData,
    BotDecisionData,
    PerformanceGraphData,
    TrainingStatus,
    GraphDataPoint
)
from bot.rl_bot_system.server.integration import TrainingEngineIntegration, MetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_server_setup():
    """Example of basic server setup and configuration."""
    logger.info("Setting up training metrics server...")
    
    # Create server configuration
    config = ServerConfig(
        host="localhost",
        port=8000,
        cors_origins=["http://localhost:3000", "http://localhost:4000"],
        max_connections_per_session=20,
        metrics_history_size=5000,
        game_server_url="http://localhost:4000",
        enable_spectator_integration=True,
        data_storage_path="data/training_metrics",
        enable_data_persistence=True
    )
    
    # Create and configure server
    server = TrainingMetricsServer(config)
    
    logger.info(f"Server configured to run on {config.host}:{config.port}")
    logger.info(f"CORS origins: {config.cors_origins}")
    logger.info(f"Max connections per session: {config.max_connections_per_session}")
    
    return server


async def example_training_session_management(server: TrainingMetricsServer):
    """Example of managing training sessions through the server."""
    logger.info("Creating training session...")
    
    # Create training session
    session_request = {
        "training_session_id": "example_training_001",
        "model_generation": 3,
        "algorithm": "DQN",
        "total_episodes": 1000,
        "room_code": "TRAIN1",
        "enable_spectators": True
    }
    
    # In a real scenario, this would be an HTTP POST request
    # For this example, we'll simulate the session creation
    session_info = await server._create_training_session_internal(session_request)
    
    logger.info(f"Created training session: {session_info['session_id']}")
    logger.info(f"Room code for spectators: {session_info.get('room_code')}")
    
    return session_info


async def example_metrics_updates(server: TrainingMetricsServer, session_id: str):
    """Example of sending training metrics updates."""
    logger.info("Sending training metrics updates...")
    
    # Simulate training progress with metrics updates
    for episode in range(1, 11):
        # Create sample metrics data
        metrics = TrainingMetricsData(
            timestamp=datetime.now(),
            episode=episode,
            total_episodes=1000,
            current_reward=10.0 + episode * 0.5,  # Improving reward
            average_reward=8.0 + episode * 0.3,
            best_reward=15.0 + episode * 0.2,
            episode_length=200 - episode * 2,  # Decreasing episode length
            win_rate=50.0 + episode * 2.0,  # Improving win rate
            loss_value=0.1 - episode * 0.005,  # Decreasing loss
            learning_rate=0.001,
            epsilon=max(0.01, 1.0 - episode * 0.05),  # Decreasing exploration
            model_generation=3,
            algorithm="DQN",
            training_time_elapsed=episode * 60.0,  # 1 minute per episode
            actions_per_second=25.0,
            frames_per_second=60.0,
            memory_usage_mb=256.0 + episode * 5.0
        )
        
        # Send metrics update (in real scenario, this would be HTTP POST)
        await server._update_training_metrics_internal(session_id, metrics)
        
        logger.info(f"Episode {episode}: Reward={metrics.current_reward:.1f}, "
                   f"Win Rate={metrics.win_rate:.1f}%, Loss={metrics.loss_value:.3f}")
        
        # Simulate bot decision data
        if episode % 3 == 0:  # Send decision data every 3rd episode
            decision = BotDecisionData(
                timestamp=datetime.now(),
                action_probabilities={
                    "move_left": 0.2,
                    "move_right": 0.3,
                    "jump": 0.1,
                    "shoot": 0.4
                },
                state_values=5.5 + episode * 0.2,
                q_values=[1.2, 2.3, 0.8, 3.1 + episode * 0.1],
                selected_action="shoot",
                confidence_score=0.7 + episode * 0.02
            )
            
            await server._update_bot_decision_internal(session_id, decision)
            logger.info(f"Episode {episode}: Bot selected '{decision.selected_action}' "
                       f"with confidence {decision.confidence_score:.2f}")
        
        # Small delay to simulate real training
        await asyncio.sleep(0.1)


async def example_performance_graphs(server: TrainingMetricsServer, session_id: str):
    """Example of creating and updating performance graphs."""
    logger.info("Creating performance graphs...")
    
    # Create sample graph data
    timestamps = [datetime.now() for _ in range(10)]
    
    # Reward progress graph
    reward_graph = PerformanceGraphData(
        graph_id="reward_progress",
        title="Training Reward Progress",
        y_label="Reward",
        metrics=["current_reward", "average_reward", "best_reward"],
        data_points={
            "current_reward": [
                GraphDataPoint(timestamp=timestamps[i], value=10.0 + i * 0.5)
                for i in range(10)
            ],
            "average_reward": [
                GraphDataPoint(timestamp=timestamps[i], value=8.0 + i * 0.3)
                for i in range(10)
            ],
            "best_reward": [
                GraphDataPoint(timestamp=timestamps[i], value=15.0 + i * 0.2)
                for i in range(10)
            ]
        },
        max_points=1000
    )
    
    await server._update_performance_graph_internal(session_id, reward_graph)
    logger.info(f"Created reward progress graph with {len(reward_graph.data_points)} metrics")
    
    # Win rate graph
    win_rate_graph = PerformanceGraphData(
        graph_id="win_rate",
        title="Win Rate Over Time",
        y_label="Win Rate (%)",
        metrics=["win_rate"],
        data_points={
            "win_rate": [
                GraphDataPoint(timestamp=timestamps[i], value=50.0 + i * 2.0)
                for i in range(10)
            ]
        },
        max_points=1000
    )
    
    await server._update_performance_graph_internal(session_id, win_rate_graph)
    logger.info(f"Created win rate graph")


async def example_integration_with_training_engine():
    """Example of integrating with a training engine."""
    logger.info("Setting up training engine integration...")
    
    # Create server
    server = await example_basic_server_setup()
    
    # Create integration adapter
    integration = TrainingEngineIntegration(server)
    
    # Create metrics collector
    collector = MetricsCollector(integration)
    
    # Register a training session
    training_id = "integrated_training_001"
    session_id = await integration.register_training_session(
        training_id=training_id,
        model_generation=2,
        algorithm="PPO",
        total_episodes=500,
        room_code="INTEG1"
    )
    
    logger.info(f"Registered training session: {session_id}")
    
    # Simulate training episodes
    for episode in range(1, 21):
        # Record episode start
        await collector.record_episode_start(training_id)
        
        # Simulate episode execution
        episode_reward = 5.0 + episode * 0.3 + (episode % 5) * 0.1  # Some variation
        episode_length = 150 + (episode % 10) * 5
        won = episode_reward > 8.0  # Win condition
        
        # Record episode end with metrics
        await collector.record_episode_end(
            training_id=training_id,
            reward=episode_reward,
            episode_length=episode_length,
            won=won,
            total_episodes=500,
            loss_value=0.05 - episode * 0.001,
            learning_rate=0.0003,
            epsilon=max(0.01, 0.5 - episode * 0.02),
            model_generation=2,
            algorithm="PPO",
            training_time_elapsed=episode * 45.0
        )
        
        # Send bot decision data occasionally
        if episode % 5 == 0:
            await integration.update_bot_decision(
                training_id=training_id,
                action_probabilities={
                    "forward": 0.4,
                    "backward": 0.1,
                    "left": 0.2,
                    "right": 0.2,
                    "action": 0.1
                },
                state_values=3.5 + episode * 0.1,
                selected_action="forward",
                confidence_score=0.8
            )
        
        logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                   f"Length={episode_length}, Won={won}")
        
        await asyncio.sleep(0.05)  # Small delay
    
    # Update training status to completed
    await integration.update_training_status(
        training_id=training_id,
        status=TrainingStatus.COMPLETED,
        end_time=datetime.now()
    )
    
    # Get final statistics
    stats = collector.get_session_stats(training_id)
    logger.info(f"Training completed. Final stats: {stats}")
    
    return integration, collector


async def example_websocket_client_simulation():
    """Example of simulating WebSocket client connections."""
    logger.info("Simulating WebSocket client connections...")
    
    # This would normally be done by frontend clients
    # For demonstration, we'll show what the client-side code would look like
    
    websocket_url = "ws://localhost:8000/ws/example_session_001"
    query_params = "?user_name=TestSpectator&user_id=spectator_123"
    full_url = f"{websocket_url}{query_params}"
    
    logger.info(f"WebSocket clients would connect to: {full_url}")
    
    # Example of messages that would be sent/received:
    example_messages = {
        "connection_status": {
            "type": "connection_status",
            "data": {
                "status": "connected",
                "connection_id": "conn_123",
                "session_id": "example_session_001",
                "subscriptions": ["training_metrics", "bot_decision", "graph_update"]
            }
        },
        "training_metrics": {
            "type": "training_metrics",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "episode": 50,
                "current_reward": 12.5,
                "win_rate": 75.0,
                "algorithm": "DQN"
            }
        },
        "bot_decision": {
            "type": "bot_decision",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "action_probabilities": {"move": 0.6, "shoot": 0.4},
                "selected_action": "move"
            }
        }
    }
    
    for msg_type, message in example_messages.items():
        logger.info(f"Example {msg_type} message: {message}")


async def example_error_handling_and_logging():
    """Example of error handling and logging configuration."""
    logger.info("Demonstrating error handling and logging...")
    
    # Configure detailed logging
    logging.getLogger("bot.rl_bot_system.server").setLevel(logging.DEBUG)
    
    # Create server with error handling
    try:
        config = ServerConfig(
            host="localhost",
            port=8000,
            log_level="DEBUG"
        )
        
        server = TrainingMetricsServer(config)
        
        # Simulate various error conditions and how they're handled
        
        # 1. Invalid session ID
        try:
            await server._get_training_session_internal("nonexistent_session")
        except Exception as e:
            logger.error(f"Expected error for invalid session: {e}")
        
        # 2. Invalid metrics data
        try:
            invalid_metrics = {
                "episode": "not_a_number",  # Invalid type
                "reward": None  # Missing required field
            }
            # This would fail validation in a real scenario
            logger.warning(f"Invalid metrics would be rejected: {invalid_metrics}")
        except Exception as e:
            logger.error(f"Validation error: {e}")
        
        # 3. WebSocket connection errors
        logger.info("WebSocket errors are handled gracefully with automatic cleanup")
        
        # 4. Resource cleanup
        logger.info("Server includes automatic cleanup of expired sessions and connections")
        
    except Exception as e:
        logger.error(f"Server setup error: {e}")


async def main():
    """Main example function demonstrating all features."""
    logger.info("Starting training metrics server examples...")
    
    try:
        # 1. Basic server setup
        server = await example_basic_server_setup()
        
        # 2. Training session management
        session_info = await example_training_session_management(server)
        session_id = session_info.get("session_id", "example_session")
        
        # 3. Metrics updates
        await example_metrics_updates(server, session_id)
        
        # 4. Performance graphs
        await example_performance_graphs(server, session_id)
        
        # 5. Training engine integration
        integration, collector = await example_integration_with_training_engine()
        
        # 6. WebSocket client simulation
        await example_websocket_client_simulation()
        
        # 7. Error handling demonstration
        await example_error_handling_and_logging()
        
        logger.info("All examples completed successfully!")
        
        # In a real scenario, you would start the server with:
        # server.run()
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())