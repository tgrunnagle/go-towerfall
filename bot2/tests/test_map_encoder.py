"""Tests for map geometry encoding."""

import numpy as np
import pytest

from bot.models.game_objects import BlockState
from bot.observation import MapEncoder, MapEncodingConfig

# =============================================================================
# Helper Functions
# =============================================================================


def create_block_state(
    block_id: str,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> BlockState:
    """Create a BlockState with rectangular bounds.

    Args:
        block_id: Unique block identifier
        min_x: Left edge x coordinate in pixels
        min_y: Top edge y coordinate in pixels
        max_x: Right edge x coordinate in pixels
        max_y: Bottom edge y coordinate in pixels

    Returns:
        BlockState with 4 corner points
    """
    return BlockState.model_validate(
        {
            "id": block_id,
            "objectType": "block",
            "pts": [
                {"x": min_x, "y": min_y},
                {"x": max_x, "y": min_y},
                {"x": max_x, "y": max_y},
                {"x": min_x, "y": max_y},
            ],
        }
    )


def create_block_at_grid(
    block_id: str,
    grid_col: int,
    grid_row: int,
    config: MapEncodingConfig,
) -> BlockState:
    """Create a block at a specific grid cell position.

    Args:
        block_id: Unique block identifier
        grid_col: Grid column (0-indexed)
        grid_row: Grid row (0-indexed)
        config: Map encoding configuration

    Returns:
        BlockState positioned at the specified grid cell
    """
    min_x = grid_col * config.cell_width_px
    min_y = grid_row * config.cell_height_px
    max_x = min_x + config.cell_width_px
    max_y = min_y + config.cell_height_px
    return create_block_state(block_id, min_x, min_y, max_x, max_y)


# =============================================================================
# MapEncodingConfig Tests
# =============================================================================


class TestMapEncodingConfig:
    """Test MapEncodingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MapEncodingConfig()
        assert config.grid_width == 20
        assert config.grid_height == 15
        assert config.room_width_px == 800.0
        assert config.room_height_px == 600.0

    def test_total_size(self):
        """Test total_size property."""
        config = MapEncodingConfig()
        assert config.total_size == 20 * 15  # 300

    def test_custom_config_total_size(self):
        """Test total_size with custom dimensions."""
        config = MapEncodingConfig(grid_width=10, grid_height=8)
        assert config.total_size == 10 * 8  # 80

    def test_cell_dimensions(self):
        """Test cell width and height calculations."""
        config = MapEncodingConfig()
        assert config.cell_width_px == pytest.approx(800.0 / 20)  # 40px
        assert config.cell_height_px == pytest.approx(600.0 / 15)  # 40px

    def test_cell_dimensions_custom(self):
        """Test cell dimensions with custom config."""
        config = MapEncodingConfig(
            grid_width=40, grid_height=30, room_width_px=800.0, room_height_px=600.0
        )
        assert config.cell_width_px == pytest.approx(20.0)
        assert config.cell_height_px == pytest.approx(20.0)

    def test_config_frozen(self):
        """Test that config is immutable."""
        config = MapEncodingConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.grid_width = 10  # type: ignore


# =============================================================================
# BlockState Property Tests
# =============================================================================


class TestBlockStateProperties:
    """Test BlockState center and bounds properties."""

    def test_center_calculation(self):
        """Test center point calculation."""
        block = create_block_state("block1", 100, 200, 300, 400)
        center = block.center
        assert center[0] == pytest.approx(200.0)  # (100 + 300) / 2
        assert center[1] == pytest.approx(300.0)  # (200 + 400) / 2

    def test_center_empty_points(self):
        """Test center with no points returns (0, 0)."""
        block = BlockState.model_validate(
            {"id": "empty", "objectType": "block", "pts": []}
        )
        assert block.center == (0.0, 0.0)

    def test_bounds_calculation(self):
        """Test bounds calculation."""
        block = create_block_state("block1", 100, 200, 300, 400)
        bounds = block.bounds
        assert bounds == (100.0, 200.0, 300.0, 400.0)

    def test_bounds_empty_points(self):
        """Test bounds with no points returns zeros."""
        block = BlockState.model_validate(
            {"id": "empty", "objectType": "block", "pts": []}
        )
        assert block.bounds == (0.0, 0.0, 0.0, 0.0)


# =============================================================================
# MapEncoder Tests
# =============================================================================


class TestMapEncoder:
    """Test MapEncoder class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        encoder = MapEncoder()
        assert encoder.config.grid_width == 20
        assert encoder.config.grid_height == 15

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = MapEncodingConfig(grid_width=10, grid_height=8)
        encoder = MapEncoder(config=config)
        assert encoder.config.grid_width == 10
        assert encoder.config.grid_height == 8

    def test_encode_empty_map(self):
        """Test encoding with no blocks (all cells empty)."""
        encoder = MapEncoder()
        grid = encoder.encode([])

        assert grid.shape == (300,)  # 20 * 15
        assert grid.dtype == np.float32
        assert np.all(grid == -1.0)  # All empty

    def test_encode_single_block(self):
        """Test encoding with a single block."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        # Block at grid cell (5, 5) - each cell is 10x10 pixels
        block = create_block_state("block1", 50, 50, 60, 60)
        grid = encoder.encode([block])

        assert grid.shape == (100,)
        # Cell (5, 5) should be occupied (1.0)
        # Grid is flattened row-major: index = row * width + col = 5 * 10 + 5 = 55
        assert grid[55] == 1.0
        # Other cells should be empty
        assert np.sum(grid == 1.0) == 1

    def test_encode_floor_row(self):
        """Test encoding a floor (row of blocks at bottom)."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        # Create blocks across the bottom row (row 9, y=90-100)
        blocks = []
        for col in range(10):
            block = create_block_at_grid(f"floor_{col}", col, 9, config)
            blocks.append(block)

        grid = encoder.encode(blocks)

        # Bottom row (row 9) should all be 1.0
        for col in range(10):
            idx = 9 * 10 + col
            assert grid[idx] == 1.0

        # Other rows should be -1.0
        assert np.sum(grid == 1.0) == 10

    def test_encode_complex_map(self):
        """Test encoding a multi-level platform map."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        blocks = [
            # Floor (row 9)
            create_block_at_grid("floor_0", 0, 9, config),
            create_block_at_grid("floor_1", 1, 9, config),
            create_block_at_grid("floor_2", 2, 9, config),
            # Platform at row 5, columns 4-6
            create_block_at_grid("plat_4", 4, 5, config),
            create_block_at_grid("plat_5", 5, 5, config),
            create_block_at_grid("plat_6", 6, 5, config),
        ]

        grid = encoder.encode(blocks)

        # Check floor
        assert grid[9 * 10 + 0] == 1.0
        assert grid[9 * 10 + 1] == 1.0
        assert grid[9 * 10 + 2] == 1.0

        # Check platform
        assert grid[5 * 10 + 4] == 1.0
        assert grid[5 * 10 + 5] == 1.0
        assert grid[5 * 10 + 6] == 1.0

        # Total occupied cells
        assert np.sum(grid == 1.0) == 6

    def test_encode_block_spanning_cells(self):
        """Test encoding a block that spans multiple grid cells."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        # Block spanning 2x2 cells (20x20 pixels at cells 3-4, 3-4)
        block = create_block_state("large", 30, 30, 50, 50)
        grid = encoder.encode([block])

        # Should occupy 4 cells: (3,3), (3,4), (4,3), (4,4)
        assert grid[3 * 10 + 3] == 1.0
        assert grid[3 * 10 + 4] == 1.0
        assert grid[4 * 10 + 3] == 1.0
        assert grid[4 * 10 + 4] == 1.0
        assert np.sum(grid == 1.0) == 4

    def test_encode_values_in_range(self):
        """Test that all values are in [-1, 1] range."""
        encoder = MapEncoder()
        blocks = [
            create_block_state(f"block_{i}", i * 40, 0, i * 40 + 40, 40)
            for i in range(5)
        ]
        grid = encoder.encode(blocks)

        assert np.all(grid >= -1.0)
        assert np.all(grid <= 1.0)

    def test_encode_block_at_edges(self):
        """Test encoding blocks at map boundaries."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        blocks = [
            create_block_at_grid("corner_tl", 0, 0, config),  # Top-left
            create_block_at_grid("corner_tr", 9, 0, config),  # Top-right
            create_block_at_grid("corner_bl", 0, 9, config),  # Bottom-left
            create_block_at_grid("corner_br", 9, 9, config),  # Bottom-right
        ]

        grid = encoder.encode(blocks)

        # Check corners
        assert grid[0 * 10 + 0] == 1.0  # Top-left
        assert grid[0 * 10 + 9] == 1.0  # Top-right
        assert grid[9 * 10 + 0] == 1.0  # Bottom-left
        assert grid[9 * 10 + 9] == 1.0  # Bottom-right
        assert np.sum(grid == 1.0) == 4

    def test_encode_block_outside_bounds(self):
        """Test encoding blocks partially outside map bounds."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        # Block extending past right edge
        block = create_block_state("overflow", 90, 50, 110, 60)
        grid = encoder.encode([block])

        # Should only mark cell (5, 9) as occupied
        assert grid[5 * 10 + 9] == 1.0
        assert np.sum(grid == 1.0) == 1


# =============================================================================
# MapEncoder Caching Tests
# =============================================================================


class TestMapEncoderCaching:
    """Test MapEncoder caching behavior."""

    def test_cache_returns_same_result(self):
        """Test that cached result is returned for same blocks."""
        encoder = MapEncoder()
        blocks = [create_block_state("block1", 100, 100, 140, 140)]

        grid1 = encoder.encode(blocks)
        grid2 = encoder.encode(blocks)

        # Should return same array (cached)
        assert grid1 is grid2

    def test_cache_invalidated_on_block_change(self):
        """Test that cache is invalidated when blocks change."""
        encoder = MapEncoder()
        blocks1 = [create_block_state("block1", 100, 100, 140, 140)]
        blocks2 = [create_block_state("block2", 200, 200, 240, 240)]

        grid1 = encoder.encode(blocks1)
        grid2 = encoder.encode(blocks2)

        # Should be different arrays
        assert grid1 is not grid2
        assert not np.array_equal(grid1, grid2)

    def test_cache_invalidated_on_block_add(self):
        """Test that cache is invalidated when block is added."""
        encoder = MapEncoder()
        block1 = create_block_state("block1", 100, 100, 140, 140)
        block2 = create_block_state("block2", 200, 200, 240, 240)

        grid1 = encoder.encode([block1])
        grid2 = encoder.encode([block1, block2])

        assert grid1 is not grid2
        assert np.sum(grid1 == 1.0) < np.sum(grid2 == 1.0)

    def test_clear_cache(self):
        """Test manual cache clearing."""
        encoder = MapEncoder()
        blocks = [create_block_state("block1", 100, 100, 140, 140)]

        grid1 = encoder.encode(blocks)
        encoder.clear_cache()
        grid2 = encoder.encode(blocks)

        # After clearing cache, should get new array
        assert grid1 is not grid2
        # But values should be the same
        assert np.array_equal(grid1, grid2)


# =============================================================================
# MapEncoder Grid Shape Tests
# =============================================================================


class TestMapEncoderShape:
    """Test MapEncoder shape-related methods."""

    def test_get_grid_shape(self):
        """Test get_grid_shape method."""
        config = MapEncodingConfig(grid_width=20, grid_height=15)
        encoder = MapEncoder(config=config)
        assert encoder.get_grid_shape() == (15, 20)

    def test_get_grid_shape_custom(self):
        """Test get_grid_shape with custom config."""
        config = MapEncodingConfig(grid_width=40, grid_height=30)
        encoder = MapEncoder(config=config)
        assert encoder.get_grid_shape() == (30, 40)


# =============================================================================
# Downsampling Tests
# =============================================================================


class TestMapEncoderDownsampling:
    """Test that downsampling works correctly."""

    def test_downsampled_block_representation(self):
        """Test that blocks at full resolution map to downsampled grid correctly."""
        # Full resolution: 40x30 (800px/20px = 40, 600px/20px = 30)
        # Downsampled: 20x15 (2x downsampling)
        config = MapEncodingConfig(
            grid_width=20, grid_height=15, room_width_px=800.0, room_height_px=600.0
        )
        encoder = MapEncoder(config=config)

        # Block at pixel (0-40, 0-40) should map to cell (0, 0)
        block = create_block_state("block1", 0, 0, 40, 40)
        grid = encoder.encode([block])

        assert grid[0] == 1.0  # Cell (0, 0)

    def test_multiple_blocks_same_cell(self):
        """Test that multiple small blocks in same cell work."""
        config = MapEncodingConfig(
            grid_width=10, grid_height=10, room_width_px=100.0, room_height_px=100.0
        )
        encoder = MapEncoder(config=config)

        # Two small blocks in same cell
        blocks = [
            create_block_state("block1", 0, 0, 5, 5),
            create_block_state("block2", 5, 5, 10, 10),
        ]
        grid = encoder.encode(blocks)

        # Both should mark cell (0, 0)
        assert grid[0] == 1.0
        assert np.sum(grid == 1.0) == 1
