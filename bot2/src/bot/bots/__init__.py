"""Bot implementations for TowerFall game."""

from bot.bots.base_bot import (
    BaseBot,
    BotAction,
    KeyboardAction,
    KeyboardKey,
    MouseAction,
    MouseButton,
)
from bot.bots.rule_based_bot import RuleBasedBot, RuleBasedBotConfig, RuleBasedBotRunner
from bot.bots.shooting_utils import (
    ShootingConfig,
    calculate_aim_point,
    calculate_arrow_speed,
    calculate_max_arrow_speed,
    calculate_optimal_power,
    calculate_optimal_power_with_thresholds,
    compensate_for_gravity,
    should_release_shot,
    should_shoot,
)

__all__ = [
    "BaseBot",
    "BotAction",
    "KeyboardAction",
    "KeyboardKey",
    "MouseAction",
    "MouseButton",
    "RuleBasedBot",
    "RuleBasedBotConfig",
    "RuleBasedBotRunner",
    "ShootingConfig",
    "calculate_aim_point",
    "calculate_arrow_speed",
    "calculate_max_arrow_speed",
    "calculate_optimal_power",
    "calculate_optimal_power_with_thresholds",
    "compensate_for_gravity",
    "should_release_shot",
    "should_shoot",
]
