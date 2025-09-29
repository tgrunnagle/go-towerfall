from dataclasses import dataclass


@dataclass
class PlayerState:
    id: str
    x: float
    y: float
    dx: float
    dy: float
    health: float
    dead: bool
    arrow_count: int
    shooting: bool
    # TODO change from Time to float
    # shooting_start_time: float
    jump_count: int
    radius: float


def player_state_from_dict(data: dict[str, any]) -> PlayerState:
    return PlayerState(
        id=data["id"],
        x=data["x"],
        y=data["y"],
        dx=data["dx"],
        dy=data["dy"],
        health=data["h"],
        dead=data["dead"],
        arrow_count=data["ac"],
        shooting=data["sht"],
        # shooting_start_time = data['shts'],
        jump_count=data["jc"],
        radius=data["rad"],
    )


@dataclass
class ArrowState:
    x: float
    y: float
    dx: float
    dy: float
    grounded: bool
    # TODO owning player


def arrow_state_from_dict(data: dict[str, any]) -> ArrowState:
    return ArrowState(
        x=data["x"], y=data["y"], dx=data["dx"], dy=data["dy"], grounded=data["ag"]
    )


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Block:
    points: list[Point]


def block_from_dict(data: dict[str, any]) -> Block:
    return Block(points=[Point(x=p["X"], y=p["Y"]) for p in data["pts"]])


@dataclass
class GameState:
    """Represents the game state."""

    player: PlayerState | None
    enemies: dict[str, PlayerState]
    blocks: dict[str, Block]
    arrows: dict[str, ArrowState]
