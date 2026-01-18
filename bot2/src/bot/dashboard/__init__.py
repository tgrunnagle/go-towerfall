"""Dashboard module for visualizing ML model training metrics.

This module provides tools for comparing performance metrics across model
generations in the successive training pipeline.
"""

from bot.dashboard.data_aggregator import DataAggregator
from bot.dashboard.models import GenerationMetrics
from bot.dashboard.visualizer import DashboardVisualizer

__all__ = ["DataAggregator", "DashboardVisualizer", "GenerationMetrics"]
