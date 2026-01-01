"""Map geometry encoding for observation space.

Converts static map geometry (blocks) into a normalized occupancy grid
for spatial awareness in RL training.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bot.models.game_objects import BlockState


@dataclass(frozen=True)
class MapEncodingConfig:
    """Configuration for map geometry encoding.

    Attributes:
        grid_width: Width of the downsampled occupancy grid
        grid_height: Height of the downsampled occupancy grid
        room_width_px: Room width in pixels
        room_height_px: Room height in pixels
    """

    grid_width: int = 20
    grid_height: int = 15
    room_width_px: float = 800.0
    room_height_px: float = 600.0

    @property
    def total_size(self) -> int:
        """Total number of values in the flattened grid."""
        return self.grid_width * self.grid_height

    @property
    def cell_width_px(self) -> float:
        """Width of each grid cell in pixels."""
        return self.room_width_px / self.grid_width

    @property
    def cell_height_px(self) -> float:
        """Height of each grid cell in pixels."""
        return self.room_height_px / self.grid_height


# Default configuration
DEFAULT_MAP_CONFIG = MapEncodingConfig()


@dataclass
class MapEncoder:
    """Encodes map geometry (blocks) into a normalized occupancy grid.

    The encoder converts block positions into a fixed-size 2D grid where:
    - 1.0 indicates a solid/occupied cell
    - -1.0 indicates an empty/passable cell

    Caching is used since map geometry is static during a game session.
    """

    config: MapEncodingConfig = field(default_factory=MapEncodingConfig)

    # Cache for the encoded grid (since maps are static)
    _cached_grid: NDArray[np.float32] | None = field(
        default=None, init=False, repr=False
    )
    _cached_blocks_hash: int | None = field(default=None, init=False, repr=False)

    def encode(self, blocks: list[BlockState]) -> NDArray[np.float32]:
        """Convert block states to a flattened occupancy grid.

        Args:
            blocks: List of BlockState objects from the game state

        Returns:
            1D array of shape (grid_width * grid_height,) with values in [-1, 1]
            where 1.0 = solid and -1.0 = empty
        """
        # Compute hash of blocks to check cache validity
        blocks_hash = self._compute_blocks_hash(blocks)

        # Return cached grid if blocks haven't changed
        if self._cached_grid is not None and self._cached_blocks_hash == blocks_hash:
            return self._cached_grid

        # Compute new grid
        grid_2d = self._blocks_to_grid(blocks)
        flattened = grid_2d.flatten().astype(np.float32)

        # Cache the result
        object.__setattr__(self, "_cached_grid", flattened)
        object.__setattr__(self, "_cached_blocks_hash", blocks_hash)

        return flattened

    def _compute_blocks_hash(self, blocks: list[BlockState]) -> int:
        """Compute a hash of block IDs for cache invalidation.

        Args:
            blocks: List of BlockState objects

        Returns:
            Hash value representing the current set of blocks
        """
        # Sort IDs for consistent hashing
        block_ids = tuple(sorted(block.id for block in blocks))
        return hash(block_ids)

    def _blocks_to_grid(self, blocks: list[BlockState]) -> NDArray[np.float32]:
        """Convert blocks to a 2D occupancy grid.

        Args:
            blocks: List of BlockState objects

        Returns:
            2D array of shape (grid_height, grid_width) with values in [-1, 1]
        """
        # Initialize grid with empty cells (-1.0)
        grid = np.full(
            (self.config.grid_height, self.config.grid_width),
            -1.0,
            dtype=np.float32,
        )

        # Mark cells occupied by blocks
        for block in blocks:
            self._mark_block_cells(grid, block)

        return grid

    def _mark_block_cells(self, grid: NDArray[np.float32], block: BlockState) -> None:
        """Mark grid cells occupied by a block.

        Args:
            grid: The 2D occupancy grid to modify in place
            block: BlockState with polygon points
        """
        if not block.points:
            return

        # Get block bounding box from points
        xs = [p.x for p in block.points]
        ys = [p.y for p in block.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Convert to grid cell indices
        start_col = max(0, int(min_x / self.config.cell_width_px))
        end_col = min(
            self.config.grid_width, int(np.ceil(max_x / self.config.cell_width_px))
        )
        start_row = max(0, int(min_y / self.config.cell_height_px))
        end_row = min(
            self.config.grid_height, int(np.ceil(max_y / self.config.cell_height_px))
        )

        # Mark cells as occupied (1.0)
        grid[start_row:end_row, start_col:end_col] = 1.0

    def clear_cache(self) -> None:
        """Clear the cached grid (useful when starting a new game)."""
        object.__setattr__(self, "_cached_grid", None)
        object.__setattr__(self, "_cached_blocks_hash", None)

    def get_grid_shape(self) -> tuple[int, int]:
        """Get the shape of the 2D grid (height, width).

        Returns:
            Tuple of (grid_height, grid_width)
        """
        return (self.config.grid_height, self.config.grid_width)
