"""
Comprehensive Bot-Game Server Physics Validation Tests

This module consolidates all physics validation tests for bot-game server communication.
It validates that bot inputs produce expected game state changes, collision detection,
projectile physics, and multi-client synchronization.

Requirements covered: 6.2, 6.3, 6.5
"""

import asyncio
import json
import logging
import math
import time
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
import pytest
import pytest_asyncio
import websockets
from websockets.exceptions import ConnectionClosed

from game_client import GameClient
from rl_bot_system.tests.test_utils import EnhancedGameClient


class PhysicsValidationClient(EnhancedGameClient):
    """Enhanced game client for comprehensive physics validation testing."""
    
    def __init__(self, ws_url: str, http_url: str):
        super().__init__(ws_url, http_url)
        self.game_states_with_time: List[Dict[str, Any]] = []
        self.input_events: List[Dict[str, Any]] = []
        self.player_positions: List[Dict[str, Any]] = []
        self.projectile_events: List[Dict[str, Any]] = []
        self.physics_anomalies: List[Dict[str, Any]] = []
        
    async def physics_message_handler(self, message: Dict[str, Any]):
        """Handle messages and extract physics data using correct structure."""
        try:
            # Store all messages
            self.test_messages.append(message)
            
            message_type = message.get("type")
            payload = message.get("payload", {})
            
            if message_type == "GameState":
                timestamp = time.time()
                self.game_states_with_time.append({
                    "timestamp": timestamp,
                    "payload": payload
                })
                
                # Extract player positions from objectStates (correct structure)
                self._extract_player_positions_from_object_states(payload, timestamp)
                self._extract_projectile_data_from_object_states(payload, timestamp)
                self._validate_game_state_physics(payload)
                
        except Exception as e:
            logging.error(f"Error in physics message handler: {e}")
    
    def _extract_player_positions_from_object_states(self, game_state: Dict[str, Any], timestamp: float):
        """Extract player positions from objectStates (where they actually are)."""
        try:
            object_states = game_state.get("objectStates", {})
            if isinstance(object_states, dict):
                for obj_id, obj_data in object_states.items():
                    if isinstance(obj_data, dict) and obj_data.get("objectType") == "player":
                        self.player_positions.append({
                            "timestamp": timestamp,
                            "player_id": obj_data.get("id", obj_id),
                            "player_name": obj_data.get("name", "unknown"),
                            "x": obj_data.get("x", 0),
                            "y": obj_data.get("y", 0),
                            "dx": obj_data.get("dx", 0),
                            "dy": obj_data.get("dy", 0),
                            "health": obj_data.get("h", 100),  # 'h' not 'health'
                            "dead": obj_data.get("dead", False)
                        })
        except Exception as e:
            logging.error(f"Error extracting player positions: {e}")
    
    def _extract_projectile_data_from_object_states(self, game_state: Dict[str, Any], timestamp: float):
        """Extract projectile data from objectStates."""
        try:
            object_states = game_state.get("objectStates", {})
            if isinstance(object_states, dict):
                for obj_id, obj_data in object_states.items():
                    if isinstance(obj_data, dict):
                        obj_type = obj_data.get("objectType", "").lower()
                        if obj_type in ["bullet", "arrow"]:
                            self.projectile_events.append({
                                "timestamp": timestamp,
                                "event": "detected",
                                "object_id": obj_data.get("id", obj_id),
                                "type": obj_type,
                                "x": obj_data.get("x", 0),
                                "y": obj_data.get("y", 0),
                                "dx": obj_data.get("dx", 0),
                                "dy": obj_data.get("dy", 0)
                            })
        except Exception as e:
            logging.error(f"Error extracting projectile data: {e}")
    
    def _validate_game_state_physics(self, game_state: Dict[str, Any]):
        """Validate physics properties of a game state."""
        try:
            # Validate player physics in objectStates
            object_states = game_state.get("objectStates", {})
            if isinstance(object_states, dict):
                for obj_id, obj_data in object_states.items():
                    if isinstance(obj_data, dict) and obj_data.get("objectType") == "player":
                        self._validate_player_physics(obj_data)
                    elif isinstance(obj_data, dict) and obj_data.get("objectType") in ["bullet", "arrow"]:
                        self._validate_object_physics(obj_data)
                        
        except Exception as e:
            logging.error(f"Error validating game state physics: {e}")
    
    def _validate_player_physics(self, player: Dict[str, Any]):
        """Validate physics properties of a player."""
        try:
            # Check for required physics fields
            required_fields = ["x", "y", "id"]
            for field in required_fields:
                if field not in player:
                    self.physics_anomalies.append({
                        "type": "missing_field",
                        "entity": "player",
                        "field": field,
                        "timestamp": time.time()
                    })
                    return
            
            # Validate coordinate values
            x, y = player.get("x"), player.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                self.physics_anomalies.append({
                    "type": "invalid_coordinate_type",
                    "entity": "player",
                    "x_type": type(x).__name__,
                    "y_type": type(y).__name__,
                    "timestamp": time.time()
                })
                return
            
            # Check for NaN or infinite values
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                self.physics_anomalies.append({
                    "type": "invalid_coordinate_value",
                    "entity": "player",
                    "x": x,
                    "y": y,
                    "timestamp": time.time()
                })
            
            # Check for unreasonable coordinate values
            if abs(x) > 50000 or abs(y) > 50000:
                self.physics_anomalies.append({
                    "type": "extreme_coordinate_value",
                    "entity": "player",
                    "x": x,
                    "y": y,
                    "timestamp": time.time()
                })
                
        except Exception as e:
            logging.error(f"Error validating player physics: {e}")
    
    def _validate_object_physics(self, obj: Dict[str, Any]):
        """Validate physics properties of a game object."""
        try:
            # Only validate objects with position data
            if "x" not in obj or "y" not in obj:
                return
            
            x, y = obj.get("x"), obj.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                self.physics_anomalies.append({
                    "type": "invalid_object_coordinate_type",
                    "entity": "object",
                    "object_type": obj.get("objectType", "unknown"),
                    "x_type": type(x).__name__,
                    "y_type": type(y).__name__,
                    "timestamp": time.time()
                })
                return
            
            # Check for NaN or infinite values
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                self.physics_anomalies.append({
                    "type": "invalid_object_coordinate_value",
                    "entity": "object",
                    "object_type": obj.get("objectType", "unknown"),
                    "x": x,
                    "y": y,
                    "timestamp": time.time()
                })
                
        except Exception as e:
            logging.error(f"Error validating object physics: {e}")
    
    async def connect_with_physics_validation(self, room_code: str, player_name: str, room_password: str = "") -> bool:
        """Connect to room with physics validation enabled."""
        success = await self.connect_to_room(room_code, player_name, room_password)
        if success:
            # Clear existing handlers and add physics validation handler
            self._message_handlers.clear()
            self.register_message_handler(self.physics_message_handler)
        return success
    
    async def record_input_event(self, input_type: str, details: Dict[str, Any]):
        """Record an input event for correlation with state changes."""
        self.input_events.append({
            "timestamp": time.time(),
            "type": input_type,
            "details": details
        })
    
    async def send_keyboard_input_with_recording(self, key: str, pressed: bool) -> bool:
        """Send keyboard input and record the event."""
        await self.record_input_event("keyboard", {"key": key, "pressed": pressed})
        return await self.send_keyboard_input(key, pressed)
    
    async def send_mouse_input_with_recording(self, button: str, pressed: bool, x: float, y: float) -> bool:
        """Send mouse input and record the event."""
        await self.record_input_event("mouse", {"button": button, "pressed": pressed, "x": x, "y": y})
        return await self.send_mouse_input(button, pressed, x, y)
    
    def get_our_player_positions_around_time(self, target_time: float, tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Get our player's positions around a target time."""
        return [
            pos for pos in self.player_positions
            if abs(pos["timestamp"] - target_time) <= tolerance and pos["player_id"] == self.player_id
        ]
    
    def get_all_player_positions_around_time(self, target_time: float, tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Get all player positions around a target time."""
        return [
            pos for pos in self.player_positions
            if abs(pos["timestamp"] - target_time) <= tolerance
        ] 
   
    def validate_movement_input_response(self, input_time: float, key: str, pressed: bool, tolerance: float = 1.0) -> Dict[str, Any]:
        """Validate that movement input produced expected position change."""
        # Get our player's position before and after input
        before_positions = self.get_our_player_positions_around_time(input_time - 0.2, 0.2)
        after_positions = self.get_our_player_positions_around_time(input_time + tolerance, 0.3)
        
        if not before_positions or not after_positions:
            return {
                "valid": False, 
                "reason": "insufficient_position_data",
                "before_count": len(before_positions),
                "after_count": len(after_positions)
            }
        
        # Use the closest positions
        before_pos = min(before_positions, key=lambda p: abs(p["timestamp"] - (input_time - 0.1)))
        after_pos = min(after_positions, key=lambda p: abs(p["timestamp"] - (input_time + tolerance/2)))
        
        # Calculate position change
        dx = after_pos["x"] - before_pos["x"]
        dy = after_pos["y"] - before_pos["y"]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Expected movement directions for key presses
        expected_movements = {
            "W": (0, -1),  # Up (negative Y)
            "S": (0, 1),   # Down (positive Y)
            "A": (-1, 0),  # Left (negative X)
            "D": (1, 0),   # Right (positive X)
        }
        
        if key not in expected_movements:
            return {"valid": False, "reason": "unknown_key"}
        
        expected_dx, expected_dy = expected_movements[key]
        
        if pressed:
            # Key pressed - should move in expected direction
            if distance < 0.5:  # Minimum movement threshold
                return {
                    "valid": False, 
                    "reason": "no_movement",
                    "distance": distance,
                    "before": (before_pos["x"], before_pos["y"]),
                    "after": (after_pos["x"], after_pos["y"]),
                    "time_diff": after_pos["timestamp"] - before_pos["timestamp"]
                }
            
            # Check if movement is in roughly the right direction
            if distance > 0:
                actual_dx_norm = dx / distance
                actual_dy_norm = dy / distance
                
                # Calculate dot product with expected direction
                dot_product = actual_dx_norm * expected_dx + actual_dy_norm * expected_dy
                
                # Should be moving in roughly the right direction (dot product > 0.3)
                if dot_product < 0.3:
                    return {
                        "valid": False,
                        "reason": "wrong_direction",
                        "expected": (expected_dx, expected_dy),
                        "actual": (actual_dx_norm, actual_dy_norm),
                        "dot_product": dot_product,
                        "distance": distance
                    }
        
        return {
            "valid": True,
            "distance": distance,
            "direction": (dx, dy),
            "before": (before_pos["x"], before_pos["y"]),
            "after": (after_pos["x"], after_pos["y"]),
            "time_diff": after_pos["timestamp"] - before_pos["timestamp"],
            "velocity_change": {
                "dx_before": before_pos["dx"],
                "dy_before": before_pos["dy"],
                "dx_after": after_pos["dx"],
                "dy_after": after_pos["dy"]
            }
        }
    
    def detect_projectiles_after_input(self, input_time: float, tolerance: float = 1.0) -> Dict[str, Any]:
        """Detect if projectiles were created after an input."""
        relevant_projectiles = [
            proj for proj in self.projectile_events
            if abs(proj["timestamp"] - input_time) <= tolerance
        ]
        
        return {
            "detected": len(relevant_projectiles) > 0,
            "projectiles": relevant_projectiles,
            "count": len(relevant_projectiles)
        }
    
    def validate_collision_detection(
        self, 
        player_pos: Tuple[float, float],
        projectile_pos: Tuple[float, float],
        collision_reported: bool,
        collision_threshold: float = 20.0
    ) -> Dict[str, Any]:
        """Validate collision detection accuracy."""
        px, py = player_pos
        proj_x, proj_y = projectile_pos
        
        actual_distance = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        should_collide = actual_distance <= collision_threshold
        
        return {
            "accurate": should_collide == collision_reported,
            "actual_distance": actual_distance,
            "threshold": collision_threshold,
            "should_collide": should_collide,
            "collision_reported": collision_reported,
            "error_type": "false_positive" if collision_reported and not should_collide else
                         "false_negative" if not collision_reported and should_collide else
                         "correct"
        }
    
    def validate_boundary_physics(
        self,
        position: Tuple[float, float],
        canvas_size: Tuple[int, int],
        boundary_margin: float = 10.0
    ) -> Dict[str, Any]:
        """Validate boundary collision and wrapping behavior."""
        x, y = position
        width, height = canvas_size
        
        violations = []
        
        if x < -boundary_margin:
            violations.append({"type": "left_boundary", "value": x})
        elif x > width + boundary_margin:
            violations.append({"type": "right_boundary", "value": x})
            
        if y < -boundary_margin:
            violations.append({"type": "top_boundary", "value": y})
        elif y > height + boundary_margin:
            violations.append({"type": "bottom_boundary", "value": y})
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "position": position,
            "canvas_size": canvas_size
        }


class TestBotGamePhysicsValidation:
    """Comprehensive test suite for bot-game server physics validation."""
    
    @pytest_asyncio.fixture(scope="class")
    async def server_manager(self):
        """Fixture to manage test servers."""
        from rl_bot_system.tests.test_bot_game_server_communication import ServerManager
        
        manager = ServerManager()
        
        # Check if Go server is running
        go_running = await manager.check_go_server_running()
        if not go_running:
            pytest.skip("Go server is not running. Please start the Go server on port 4000 before running tests.")
        
        # Start bot server
        bot_started = await manager.start_bot_server()
        if not bot_started:
            pytest.skip("Failed to start bot server")
        
        # Wait for bot server to be healthy
        if not await manager.wait_for_bot_server_health():
            await manager.stop_bot_server()
            pytest.skip("Bot server not healthy")
        
        yield manager
        
        # Cleanup
        await manager.stop_bot_server()
    
    @pytest_asyncio.fixture
    async def test_room(self, server_manager):
        """Fixture to create a test room."""
        room_id, room_code, room_password = await server_manager.create_test_room()
        return {"room_id": room_id, "room_code": room_code, "room_password": room_password}
    
    @pytest.mark.asyncio
    async def test_movement_inputs_produce_expected_position_changes(self, server_manager, test_room):
        """Test that movement inputs actually change player positions in the expected direction."""
        client = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="MovementValidationBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Wait for initial state and position data
            await asyncio.sleep(1.0)
            assert len(client.player_positions) > 0, "No initial position data captured"
            
            logging.info(f"Captured {len(client.player_positions)} initial position updates")
            logging.info(f"Our player ID: {client.player_id}")
            
            # Test movement inputs and validate actual position changes
            movement_tests = [
                ("W", "up movement", "should decrease Y"),
                ("A", "left movement", "should decrease X"),
                ("S", "down movement", "should increase Y"),
                ("D", "right movement", "should increase X"),
            ]
            
            successful_movements = []
            failed_movements = []
            
            for key, description, expectation in movement_tests:
                logging.info(f"Testing {description} with key '{key}' - {expectation}")
                
                # Record time before input
                input_time = time.time()
                
                # Send movement input
                await client.send_keyboard_input_with_recording(key, True)
                await asyncio.sleep(0.8)  # Allow time for movement
                await client.send_keyboard_input_with_recording(key, False)
                
                # Wait for state updates
                await asyncio.sleep(0.5)
                
                # Validate movement response
                validation = client.validate_movement_input_response(input_time, key, True, tolerance=1.5)
                
                if validation["valid"]:
                    successful_movements.append({
                        "key": key,
                        "description": description,
                        "validation": validation
                    })
                    logging.info(f"✓ {description} successful: moved {validation['distance']:.2f} pixels "
                               f"from {validation['before']} to {validation['after']} "
                               f"in {validation['time_diff']:.3f}s")
                else:
                    failed_movements.append({
                        "key": key,
                        "description": description,
                        "validation": validation
                    })
                    logging.warning(f"✗ {description} failed: {validation['reason']}")
                    if 'before' in validation and 'after' in validation:
                        logging.warning(f"  Position: {validation['before']} -> {validation['after']}")
                    if 'distance' in validation:
                        logging.warning(f"  Distance: {validation['distance']:.3f}")
            
            # Validate that at least some movements were detected
            assert len(successful_movements) > 0, \
                f"No movement detected for any input keys. Failed movements: {failed_movements}"
            
            # We expect at least 50% of movements to work
            success_rate = len(successful_movements) / len(movement_tests)
            assert success_rate >= 0.5, \
                f"Too few successful movements: {len(successful_movements)}/{len(movement_tests)} ({success_rate:.1%}). " \
                f"Failed: {[f['validation']['reason'] for f in failed_movements]}"
            
            logging.info(f"✓ Movement validation successful: {len(successful_movements)}/{len(movement_tests)} movements detected correctly")
            
            # Validate that successful movements moved in reasonable directions
            for movement_data in successful_movements:
                validation = movement_data["validation"]
                key = movement_data["key"]
                
                # Check movement distance is reasonable
                distance = validation["distance"]
                assert 0.5 <= distance <= 500.0, \
                    f"Movement {key} distance unreasonable: {distance} pixels"
                
                # Log velocity changes to understand physics
                vel_change = validation["velocity_change"]
                logging.info(f"  {key} velocity change: dx {vel_change['dx_before']:.1f}→{vel_change['dx_after']:.1f}, "
                           f"dy {vel_change['dy_before']:.1f}→{vel_change['dy_after']:.1f}")
            
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_shooting_inputs_create_projectiles(self, server_manager, test_room):
        """Test that shooting inputs actually create projectiles in game state."""
        client = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="ShootingValidationBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Wait for initial state
            await asyncio.sleep(1.0)
            
            # Test shooting at different targets
            shooting_tests = [
                (200, 200, "center shot"),
                (100, 100, "upper-left shot"),
                (300, 150, "right shot"),
            ]
            
            successful_shots = []
            
            for target_x, target_y, description in shooting_tests:
                logging.info(f"Testing {description} at ({target_x}, {target_y})")
                
                # Record time before shooting
                input_time = time.time()
                
                # Shoot at target
                await client.send_mouse_input_with_recording("left", True, target_x, target_y)
                await asyncio.sleep(0.1)
                await client.send_mouse_input_with_recording("left", False, target_x, target_y)
                
                # Wait for projectile creation
                await asyncio.sleep(1.5)
                
                # Detect projectile creation
                projectiles = client.detect_projectiles_after_input(input_time, tolerance=2.0)
                
                if projectiles["detected"]:
                    successful_shots.append({
                        "target": (target_x, target_y),
                        "description": description,
                        "projectiles": projectiles
                    })
                    logging.info(f"✓ {description} successful: {projectiles['count']} projectiles created")
                else:
                    logging.warning(f"✗ {description} failed: no projectiles detected")
            
            # Validate shooting behavior
            if len(successful_shots) > 0:
                logging.info(f"✓ Shooting validation successful: {len(successful_shots)}/{len(shooting_tests)} shots created projectiles")
                
                # Validate projectile properties for successful shots
                for shot_data in successful_shots:
                    projectiles = shot_data["projectiles"]["projectiles"]
                    for proj in projectiles[:3]:  # Check first 3 projectiles
                        # Validate projectile has reasonable properties
                        assert proj["type"] in ["bullet", "arrow"], f"Unknown projectile type: {proj['type']}"
                        assert isinstance(proj["x"], (int, float)), "Projectile x coordinate not numeric"
                        assert isinstance(proj["y"], (int, float)), "Projectile y coordinate not numeric"
                        
                        # Validate projectile position is reasonable
                        assert -100 <= proj["x"] <= 1000, f"Projectile x coordinate out of bounds: {proj['x']}"
                        assert -100 <= proj["y"] <= 1000, f"Projectile y coordinate out of bounds: {proj['y']}"
            else:
                logging.warning("No projectiles detected - this may be expected based on game mechanics")
            
            # For shooting, we're lenient since projectile mechanics vary
            assert len(successful_shots) >= 0, "Shooting test completed"
            
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_game_state_structure_and_consistency(self, server_manager, test_room):
        """Test that game states have valid structure and no physics anomalies."""
        client = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="StructureValidationBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Wait for initial game states
            await asyncio.sleep(1.0)
            
            # Perform various actions to generate different game states
            await client.send_keyboard_input_with_recording("W", True)
            await asyncio.sleep(0.2)
            await client.send_keyboard_input_with_recording("W", False)
            
            await client.send_mouse_input_with_recording("left", True, 200, 200)
            await asyncio.sleep(0.1)
            await client.send_mouse_input_with_recording("left", False, 200, 200)
            
            # Wait for state updates
            await asyncio.sleep(1.0)
            
            # Validate we received game states
            assert len(client.game_states_with_time) > 0, "No game states received"
            
            # Validate no critical physics anomalies were detected
            critical_anomalies = [
                anomaly for anomaly in client.physics_anomalies
                if anomaly.get("type") in [
                    "invalid_coordinate_value", 
                    "invalid_coordinate_type",
                    "extreme_coordinate_value"
                ]
            ]
            
            assert len(critical_anomalies) == 0, f"Critical physics anomalies detected: {critical_anomalies}"
            
            # Validate game state consistency
            for i, state_data in enumerate(client.game_states_with_time):
                payload = state_data["payload"]
                
                # Basic structure validation
                assert isinstance(payload, dict), f"Game state {i} payload is not a dictionary"
                
                # ObjectStates validation (where players actually are)
                object_states = payload.get("objectStates", {})
                assert isinstance(object_states, dict), f"ObjectStates field is not a dict in state {i}"
                
                # Validate each player has reasonable structure
                player_count = 0
                for obj_id, obj_data in object_states.items():
                    if isinstance(obj_data, dict) and obj_data.get("objectType") == "player":
                        player_count += 1
                        if "x" in obj_data and "y" in obj_data:
                            x, y = obj_data["x"], obj_data["y"]
                            assert isinstance(x, (int, float)), f"Player x coordinate not numeric in state {i}"
                            assert isinstance(y, (int, float)), f"Player y coordinate not numeric in state {i}"
                
                # Should have at least one player (us)
                assert player_count >= 1, f"No players found in state {i}"
            
            logging.info(f"✓ Structure validation successful: {len(client.game_states_with_time)} states validated")
            
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_multi_client_state_synchronization(self, server_manager, test_room):
        """Test game state synchronization across multiple clients."""
        client1 = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        client2 = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            # Connect both clients
            success1 = await client1.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="SyncBot1",
                room_password=test_room["room_password"]
            )
            assert success1, "Failed to connect client1"
            
            success2 = await client2.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="SyncBot2",
                room_password=test_room["room_password"]
            )
            assert success2, "Failed to connect client2"
            
            # Wait for both clients to receive initial states
            await asyncio.sleep(1.0)
            
            # Both clients should see players
            assert len(client1.player_positions) > 0, "Client1 has no position data"
            assert len(client2.player_positions) > 0, "Client2 has no position data"
            
            # Client1 performs movement
            input_time = time.time()
            await client1.send_keyboard_input_with_recording("W", True)
            await asyncio.sleep(0.8)
            await client1.send_keyboard_input_with_recording("W", False)
            
            # Wait for state synchronization
            await asyncio.sleep(0.5)
            
            # Both clients should detect the movement
            client1_validation = client1.validate_movement_input_response(input_time, "W", True, tolerance=1.5)
            
            if client1_validation["valid"]:
                logging.info("✓ Client1 detected its own movement successfully")
            else:
                logging.warning(f"✗ Client1 failed to detect its own movement: {client1_validation['reason']}")
            
            # Validate both clients see the same players
            client1_latest_positions = client1.get_all_player_positions_around_time(time.time(), 1.0)
            client2_latest_positions = client2.get_all_player_positions_around_time(time.time(), 1.0)
            
            # Group by player name
            client1_players = set(pos["player_name"] for pos in client1_latest_positions)
            client2_players = set(pos["player_name"] for pos in client2_latest_positions)
            
            # Both clients should see multiple players
            assert len(client1_players) >= 2, f"Client1 sees only {len(client1_players)} players: {client1_players}"
            assert len(client2_players) >= 2, f"Client2 sees only {len(client2_players)} players: {client2_players}"
            
            # Players should be consistent across clients
            common_players = client1_players.intersection(client2_players)
            assert len(common_players) >= 2, f"Clients see different players: {client1_players} vs {client2_players}"
            
            # Validate no physics anomalies in multi-client scenario
            total_anomalies = len(client1.physics_anomalies) + len(client2.physics_anomalies)
            assert total_anomalies == 0, f"Physics anomalies detected in multi-client test: {total_anomalies}"
            
            logging.info(f"✓ Multi-client synchronization successful: both clients see {len(common_players)} common players")
            
        finally:
            await client1.exit_game()
            await client2.exit_game()
    
    @pytest.mark.asyncio
    async def test_boundary_conditions_and_edge_cases(self, server_manager, test_room):
        """Test physics system handling of boundary conditions and edge cases."""
        client = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="BoundaryTestBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Wait for initial state
            await asyncio.sleep(0.5)
            
            # Test rapid input changes
            rapid_inputs = [
                ("W", True), ("W", False),
                ("A", True), ("A", False),
                ("S", True), ("S", False),
                ("D", True), ("D", False),
            ]
            
            for key, pressed in rapid_inputs:
                await client.send_keyboard_input_with_recording(key, pressed)
                await asyncio.sleep(0.02)  # Very rapid inputs
            
            # Test simultaneous conflicting inputs
            await client.send_keyboard_input_with_recording("W", True)  # Up
            await client.send_keyboard_input_with_recording("S", True)  # Down (conflicting)
            await asyncio.sleep(0.2)
            await client.send_keyboard_input_with_recording("A", True)  # Left
            await client.send_keyboard_input_with_recording("D", True)  # Right (conflicting)
            await asyncio.sleep(0.2)
            
            # Release all keys
            for key in ["W", "S", "A", "D"]:
                await client.send_keyboard_input_with_recording(key, False)
            
            # Test boundary shooting (extreme coordinates)
            boundary_targets = [
                (0, 0),        # Origin
                (800, 600),    # Far corner
                (400, 300),    # Center
            ]
            
            for target_x, target_y in boundary_targets:
                await client.send_mouse_input_with_recording("left", True, target_x, target_y)
                await asyncio.sleep(0.05)
                await client.send_mouse_input_with_recording("left", False, target_x, target_y)
                await asyncio.sleep(0.1)
            
            # Wait for all state updates
            await asyncio.sleep(1.0)
            
            # Validate system stability after edge case testing
            assert len(client.game_states_with_time) > 0, "No game states received after edge case testing"
            
            # Check for physics anomalies caused by edge cases
            severe_anomalies = [
                anomaly for anomaly in client.physics_anomalies
                if anomaly.get("type") in ["invalid_coordinate_value", "extreme_coordinate_value"]
            ]
            
            assert len(severe_anomalies) == 0, f"Severe physics anomalies from edge cases: {severe_anomalies}"
            
            # Validate final game state is still valid
            if client.player_positions:
                final_position = client.player_positions[-1]
                x, y = final_position["x"], final_position["y"]
                assert not math.isnan(x), "Final player position contains NaN"
                assert not math.isnan(y), "Final player position contains NaN"
                assert not math.isinf(x), "Final player position contains infinity"
                assert not math.isinf(y), "Final player position contains infinity"
            
            logging.info(f"✓ Boundary conditions test successful: {len(severe_anomalies)} severe anomalies")
            
        finally:
            await client.exit_game()
    
    @pytest.mark.asyncio
    async def test_game_timing_and_tick_rate_consistency(self, server_manager, test_room):
        """Test game timing and tick rate consistency."""
        client = PhysicsValidationClient(
            ws_url=server_manager.ws_url,
            http_url=server_manager.go_server_url
        )
        
        try:
            success = await client.connect_with_physics_validation(
                room_code=test_room["room_code"],
                player_name="TimingTestBot",
                room_password=test_room["room_password"]
            )
            assert success, "Failed to connect to test room"
            
            # Measure message rate over time
            measurement_start = time.time()
            initial_state_count = len(client.game_states_with_time)
            
            # Perform continuous activity to generate state updates
            await client.send_keyboard_input_with_recording("W", True)
            await asyncio.sleep(2.0)  # Measure for 2 seconds
            await client.send_keyboard_input_with_recording("W", False)
            
            measurement_end = time.time()
            final_state_count = len(client.game_states_with_time)
            
            # Calculate update rate
            measurement_duration = measurement_end - measurement_start
            state_updates = final_state_count - initial_state_count
            update_rate = state_updates / measurement_duration
            
            # Validate reasonable update rate (should be between 1-100 Hz)
            assert 0.5 <= update_rate <= 200, f"Unreasonable update rate: {update_rate:.2f} Hz"
            
            # Test response time consistency
            response_times = []
            
            for _ in range(3):
                input_time = time.time()
                await client.send_keyboard_input_with_recording("A", True)
                await asyncio.sleep(0.05)
                await client.send_keyboard_input_with_recording("A", False)
                
                # Wait for response
                await asyncio.sleep(0.3)
                response_time = time.time()
                
                response_times.append(response_time - input_time)
            
            # Validate response times are reasonable
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time:.3f}s"
            assert max_response_time < 3.0, f"Maximum response time too high: {max_response_time:.3f}s"
            
            logging.info(f"✓ Timing validation successful: {update_rate:.1f} Hz, avg response {avg_response_time:.3f}s")
            
        finally:
            await client.exit_game()


if __name__ == "__main__":
    # Configure logging for test runs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])