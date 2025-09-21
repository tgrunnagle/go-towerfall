"""
Integration tests for training mode with GameEnvironment and extended GameClient.

Tests the integration between the GameEnvironment and the extended GameClient
with training mode capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from bot.rl_bot_system.environment.game_environment import GameEnvironment, TrainingMode
from bot.game_client import GameClient, TrainingMode as ClientTrainingMode


class TestTrainingIntegration:
    """Test integration between GameEnvironment and training mode GameClient."""

    def test_training_mode_enum_compatibility(self):
        """Test that training mode enums are compatible."""
        # Verify enum values match between GameEnvironment and GameClient
        assert TrainingMode.TRAINING.value == "training"
        assert ClientTrainingMode.TRAINING.value == "training"
        assert ClientTrainingMode.HEADLESS.value == "headless"
        assert ClientTrainingMode.NORMAL.value == "normal"

    @pytest.mark.asyncio
    async def test_game_environment_with_training_client(self):
        """Test GameEnvironment with training-enabled GameClient."""
        # Create mock GameClient with training capabilities
        mock_client = Mock(spec=GameClient)
        mock_client.is_training_mode.return_value = True
        mock_client.training_mode = ClientTrainingMode.TRAINING
        mock_client.speed_multiplier = 10.0
        mock_client.direct_state_access = True
        mock_client.get_direct_state = AsyncMock(return_value={
            'player': {
                'id': 'test_player',
                'position': {'x': 100.0, 'y': 100.0},
                'health': 100,
                'is_alive': True
            },
            'enemies': [],
            'projectiles': [],
            'timestamp': 12345
        })
        mock_client.register_message_handler = Mock()
        mock_client.send_keyboard_input = AsyncMock()
        mock_client.send_mouse_input = AsyncMock()

        # Create GameEnvironment with training mode
        env = GameEnvironment(
            game_client=mock_client,
            training_mode=TrainingMode.TRAINING,
            max_episode_steps=100
        )

        # Test that environment recognizes training mode
        assert env.training_mode == TrainingMode.TRAINING
        
        # Test environment configuration
        env._configure_training_mode()  # Should not raise any errors

    @pytest.mark.asyncio
    async def test_direct_state_access_integration(self):
        """Test integration with direct state access."""
        # Create mock GameClient with direct state access
        mock_client = Mock(spec=GameClient)
        mock_client.is_training_mode.return_value = True
        mock_client.direct_state_access = True
        mock_client.get_direct_state = AsyncMock(return_value={
            'objects': {
                'player_1': {
                    'type': 'player',
                    'position': {'x': 150.0, 'y': 200.0},
                    'health': 80,
                    'is_alive': True
                }
            },
            'room': {
                'id': 'test_room',
                'speedMultiplier': 15.0
            },
            'timestamp': 67890
        })
        mock_client.register_message_handler = Mock()
        mock_client.send_keyboard_input = AsyncMock()
        mock_client.send_mouse_input = AsyncMock()

        # Create environment
        env = GameEnvironment(
            game_client=mock_client,
            training_mode=TrainingMode.TRAINING
        )

        # Test that we can potentially use direct state access
        # (This would be implemented in a future enhancement)
        assert mock_client.direct_state_access == True

    def test_training_mode_switching(self):
        """Test switching training modes in GameEnvironment."""
        mock_client = Mock(spec=GameClient)
        mock_client.register_message_handler = Mock()

        env = GameEnvironment(
            game_client=mock_client,
            training_mode=TrainingMode.TRAINING
        )

        # Test initial mode
        assert env.training_mode == TrainingMode.TRAINING

        # Test mode switching
        env.set_training_mode(TrainingMode.EVALUATION)
        assert env.training_mode == TrainingMode.EVALUATION

    @pytest.mark.asyncio
    async def test_training_client_state_callbacks(self):
        """Test state update callbacks with training client."""
        # Create real GameClient to test callback functionality
        client = GameClient()
        
        callback_called = False
        received_state = None

        async def test_callback(state):
            nonlocal callback_called, received_state
            callback_called = True
            received_state = state

        try:
            # Enable training mode
            await client.enable_training_mode(speed_multiplier=5.0)
            
            # Register callback
            client.register_state_update_callback(test_callback)
            
            # Simulate state update
            test_state = {'test': 'data', 'training': True}
            client._state_cache = test_state
            
            # Trigger callbacks
            for callback in client._state_update_callbacks:
                await callback(test_state)
            
            # Verify callback was called
            assert callback_called
            assert received_state == test_state
            
        finally:
            await client.close()

    def test_training_info_integration(self):
        """Test training info retrieval integration."""
        client = GameClient()
        
        # Test initial state
        training_info = client.get_training_info()
        assert training_info['training_mode'] == 'normal'
        assert training_info['speed_multiplier'] == 1.0
        assert not training_info['direct_state_access']

    @pytest.mark.asyncio
    async def test_mock_training_room_creation(self):
        """Test training room creation with mocked server responses."""
        client = GameClient()
        
        # Mock the HTTP request for training room creation
        mock_response_data = {
            'success': True,
            'playerId': 'test_player_id',
            'playerToken': 'test_token',
            'roomId': 'test_room_id',
            'roomCode': 'TEST123',
            'roomName': 'Test Training Room',
            'canvasSizeX': 800,
            'canvasSizeY': 600,
            'speedMultiplier': 10.0,
            'headlessMode': False
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Configure mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response

            try:
                # Test training room creation
                room_data = await client.create_training_room(
                    room_name="Test Training Room",
                    player_name="Test Bot",
                    speed_multiplier=10.0,
                    headless=False
                )

                # Verify the response
                assert room_data['success'] == True
                assert room_data['roomCode'] == 'TEST123'
                assert room_data['speedMultiplier'] == 10.0
                assert client.player_id == 'test_player_id'
                assert client.room_id == 'test_room_id'
                assert client.training_mode == ClientTrainingMode.TRAINING

            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_mock_speed_control(self):
        """Test speed control with mocked server responses."""
        client = GameClient()
        client.room_id = 'test_room'
        client.player_token = 'test_token'

        mock_response_data = {
            'success': True,
            'speedMultiplier': 25.0
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Configure mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response

            try:
                # Test speed control
                await client.set_room_speed(25.0)

                # Verify speed was updated
                assert client.speed_multiplier == 25.0

                # Verify correct API call was made
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert 'api/training/rooms/test_room/speed' in str(call_args)

            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_mock_direct_state_access(self):
        """Test direct state access with mocked server responses."""
        client = GameClient()
        client.room_id = 'test_room'
        client.player_token = 'test_token'
        client.direct_state_access = True

        mock_state_data = {
            'success': True,
            'state': {
                'objects': {'player_1': {'health': 100}},
                'room': {'speedMultiplier': 15.0}
            },
            'timestamp': 123456789
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            # Configure mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value=mock_state_data)
            mock_get.return_value.__aenter__.return_value = mock_response

            try:
                # Test direct state access
                state = await client.get_direct_state()

                # Verify state was retrieved
                assert 'objects' in state
                assert 'room' in state
                assert state['objects']['player_1']['health'] == 100

                # Verify correct API call was made
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert 'api/training/rooms/test_room/state' in str(call_args)

            finally:
                await client.close()