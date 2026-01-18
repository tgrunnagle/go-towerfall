"""Visualization generation for the training dashboard.

This module provides the DashboardVisualizer class that creates charts and
visualizations for comparing model performance across generations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

# Use Agg backend for non-interactive rendering (no GUI required)
import matplotlib

matplotlib.use("Agg")

from bot.dashboard.data_aggregator import DataAggregator
from bot.dashboard.models import DashboardConfig, GenerationMetrics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class DashboardVisualizer:
    """Generates visualizations for training metrics comparison.

    Creates charts showing model performance progression across generations,
    including K/D ratio trends, win rates, and reward curves.

    Example:
        aggregator = DataAggregator("./model_registry")
        visualizer = DashboardVisualizer(aggregator, output_dir="./reports")
        visualizer.generate_all()
    """

    def __init__(
        self,
        aggregator: DataAggregator,
        output_dir: str | Path = "./reports",
        title: str = "Training Generation Comparison",
    ) -> None:
        """Initialize the visualizer.

        Args:
            aggregator: DataAggregator instance with loaded metrics
            output_dir: Directory to save generated visualizations
            title: Title for the dashboard
        """
        self.aggregator = aggregator
        self.output_dir = Path(output_dir)
        self.title = title

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: DashboardConfig) -> "DashboardVisualizer":
        """Create a DashboardVisualizer from a DashboardConfig.

        Args:
            config: Dashboard configuration

        Returns:
            Configured DashboardVisualizer instance
        """
        aggregator = DataAggregator.from_config(config)
        return cls(
            aggregator=aggregator,
            output_dir=config.output_dir,
            title=config.title,
        )

    def generate_all(
        self,
        output_format: Literal["html", "png", "both"] = "html",
        generation_range: tuple[int, int] | None = None,
    ) -> list[Path]:
        """Generate all dashboard visualizations.

        Args:
            output_format: Output format(s) to generate
            generation_range: Optional (start, end) generation range

        Returns:
            List of paths to generated files
        """
        metrics = self.aggregator.get_all_generation_metrics(generation_range)

        if not metrics:
            logger.warning("No generation metrics available for visualization")
            return []

        generated_files: list[Path] = []

        # Generate K/D ratio progression chart
        kd_files = self._generate_kd_ratio_chart(metrics, output_format)
        generated_files.extend(kd_files)

        # Generate win rate bar chart
        wr_files = self._generate_win_rate_chart(metrics, output_format)
        generated_files.extend(wr_files)

        # Generate reward progression chart
        reward_files = self._generate_reward_chart(metrics, output_format)
        generated_files.extend(reward_files)

        # Generate summary table (HTML only)
        if output_format in ("html", "both"):
            table_path = self._generate_summary_table(metrics)
            generated_files.append(table_path)

            # Generate combined HTML dashboard
            dashboard_path = self._generate_combined_dashboard(metrics)
            generated_files.append(dashboard_path)

        logger.info(
            "Generated %d visualization files in %s",
            len(generated_files),
            self.output_dir,
        )

        return generated_files

    def _generate_kd_ratio_chart(
        self,
        metrics: list[GenerationMetrics],
        output_format: Literal["html", "png", "both"],
    ) -> list[Path]:
        """Generate K/D ratio progression line chart.

        Args:
            metrics: List of generation metrics
            output_format: Output format(s)

        Returns:
            List of generated file paths
        """
        import matplotlib.pyplot as plt

        generations = [m.generation_id for m in metrics]
        kd_ratios = [m.kill_death_ratio for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            generations,
            kd_ratios,
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2563eb",
        )
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Kill/Death Ratio", fontsize=12)
        ax.set_title("K/D Ratio Progression Across Generations", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(generations)

        # Add value labels above points
        for gen, kd in zip(generations, kd_ratios, strict=False):
            ax.annotate(
                f"{kd:.2f}",
                (gen, kd),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()

        return self._save_figure(fig, "kd_ratio_progression", output_format)

    def _generate_win_rate_chart(
        self,
        metrics: list[GenerationMetrics],
        output_format: Literal["html", "png", "both"],
    ) -> list[Path]:
        """Generate win rate bar chart by generation vs opponent.

        Args:
            metrics: List of generation metrics
            output_format: Output format(s)

        Returns:
            List of generated file paths
        """
        import matplotlib.pyplot as plt

        generations = [m.generation_id for m in metrics]
        win_rates = [m.win_rate * 100 for m in metrics]
        opponents = [m.opponent_type for m in metrics]

        # Create color map based on opponent type
        colors = []
        for opp in opponents:
            if opp == "baseline":
                colors.append("#10b981")  # Green for baseline
            else:
                colors.append("#6366f1")  # Indigo for model opponents

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(generations, win_rates, color=colors, edgecolor="white")

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_title("Win Rate by Generation", fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_xticks(generations)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, win_rate in zip(bars, win_rates, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{win_rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#10b981", label="vs Baseline"),
            Patch(facecolor="#6366f1", label="vs Previous Gen"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        return self._save_figure(fig, "win_rate_by_generation", output_format)

    def _generate_reward_chart(
        self,
        metrics: list[GenerationMetrics],
        output_format: Literal["html", "png", "both"],
    ) -> list[Path]:
        """Generate average reward progression chart.

        Args:
            metrics: List of generation metrics
            output_format: Output format(s)

        Returns:
            List of generated file paths
        """
        import matplotlib.pyplot as plt

        generations = [m.generation_id for m in metrics]
        rewards = [m.avg_episode_reward for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            generations,
            rewards,
            marker="s",
            linewidth=2,
            markersize=8,
            color="#f59e0b",
        )
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Average Episode Reward", fontsize=12)
        ax.set_title("Average Reward Progression Across Generations", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(generations)

        # Add value labels
        for gen, reward in zip(generations, rewards, strict=False):
            ax.annotate(
                f"{reward:.1f}",
                (gen, reward),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()

        return self._save_figure(fig, "reward_progression", output_format)

    def _save_figure(
        self,
        fig: Figure,
        name: str,
        output_format: Literal["html", "png", "both"],
    ) -> list[Path]:
        """Save a matplotlib figure in the specified format(s).

        Args:
            fig: Matplotlib figure to save
            name: Base name for the output file
            output_format: Output format(s)

        Returns:
            List of saved file paths
        """
        import matplotlib.pyplot as plt

        saved_paths: list[Path] = []

        if output_format in ("png", "both"):
            png_path = self.output_dir / f"{name}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            saved_paths.append(png_path)
            logger.debug("Saved PNG: %s", png_path)

        if output_format in ("html", "both"):
            # For HTML, embed as base64 in an img tag
            import base64
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")

            html_content = f"""<!DOCTYPE html>
<html>
<head><title>{name}</title></head>
<body style="display: flex; justify-content: center; padding: 20px;">
<img src="data:image/png;base64,{img_base64}" alt="{name}">
</body>
</html>"""

            html_path = self.output_dir / f"{name}.html"
            html_path.write_text(html_content)
            saved_paths.append(html_path)
            logger.debug("Saved HTML: %s", html_path)

        plt.close(fig)
        return saved_paths

    def _generate_summary_table(
        self,
        metrics: list[GenerationMetrics],
    ) -> Path:
        """Generate an HTML summary table of all generations.

        Args:
            metrics: List of generation metrics

        Returns:
            Path to generated HTML file
        """
        table_data = self.aggregator.get_summary_table_data()

        if not table_data:
            # Return empty table
            html_content = """<!DOCTYPE html>
<html>
<head><title>Summary Table</title></head>
<body><p>No data available</p></body>
</html>"""
            path = self.output_dir / "summary_table.html"
            path.write_text(html_content)
            return path

        # Build HTML table
        headers = list(table_data[0].keys())
        header_row = "".join(f"<th>{h}</th>" for h in headers)

        rows = []
        for row_data in table_data:
            cells = "".join(f"<td>{row_data[h]}</td>" for h in headers)
            rows.append(f"<tr>{cells}</tr>")

        table_html = f"""
<table>
<thead><tr>{header_row}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>Generation Summary</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1000px; margin: 0 auto; }}
th, td {{ border: 1px solid #e5e7eb; padding: 12px 16px; text-align: left; }}
th {{ background-color: #f9fafb; font-weight: 600; }}
tr:nth-child(even) {{ background-color: #f9fafb; }}
tr:hover {{ background-color: #f3f4f6; }}
</style>
</head>
<body>
<h2 style="text-align: center;">Generation Summary</h2>
{table_html}
</body>
</html>"""

        path = self.output_dir / "summary_table.html"
        path.write_text(html_content)
        logger.debug("Saved summary table: %s", path)
        return path

    def _generate_combined_dashboard(
        self,
        metrics: list[GenerationMetrics],
    ) -> Path:
        """Generate a combined HTML dashboard with all visualizations.

        Args:
            metrics: List of generation metrics

        Returns:
            Path to generated dashboard HTML file
        """
        import matplotlib.pyplot as plt

        # Generate all charts as base64 images
        chart_data: list[tuple[str, str]] = []

        # K/D ratio chart
        fig = self._create_kd_ratio_figure(metrics)
        chart_data.append(("K/D Ratio Progression", self._fig_to_base64(fig)))
        plt.close(fig)

        # Win rate chart
        fig = self._create_win_rate_figure(metrics)
        chart_data.append(("Win Rate by Generation", self._fig_to_base64(fig)))
        plt.close(fig)

        # Reward chart
        fig = self._create_reward_figure(metrics)
        chart_data.append(("Average Reward Progression", self._fig_to_base64(fig)))
        plt.close(fig)

        # Build chart HTML
        charts_html = ""
        for chart_title, img_base64 in chart_data:
            charts_html += f"""
<div class="chart-container">
<h3>{chart_title}</h3>
<img src="data:image/png;base64,{img_base64}" alt="{chart_title}">
</div>
"""

        # Build summary table
        table_data = self.aggregator.get_summary_table_data()
        table_html = self._build_table_html(table_data)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{self.title}</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f9fafb;
}}
.container {{
    max-width: 1200px;
    margin: 0 auto;
}}
h1 {{
    text-align: center;
    color: #1f2937;
    margin-bottom: 30px;
}}
h2 {{
    color: #374151;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 10px;
}}
h3 {{
    color: #4b5563;
    margin-top: 0;
}}
.chart-container {{
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.chart-container img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}}
.charts-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
}}
@media (min-width: 1000px) {{
    .charts-grid {{
        grid-template-columns: 1fr 1fr;
    }}
    .charts-grid .chart-container:first-child {{
        grid-column: 1 / -1;
    }}
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
th, td {{
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}}
th {{
    background-color: #f3f4f6;
    font-weight: 600;
    color: #374151;
}}
tr:last-child td {{
    border-bottom: none;
}}
tr:hover {{
    background-color: #f9fafb;
}}
.footer {{
    text-align: center;
    color: #6b7280;
    margin-top: 30px;
    font-size: 0.875rem;
}}
</style>
</head>
<body>
<div class="container">
<h1>{self.title}</h1>

<h2>Performance Charts</h2>
<div class="charts-grid">
{charts_html}
</div>

<h2>Generation Summary</h2>
{table_html}

<div class="footer">
Generated by go-towerfall ML Training Dashboard
</div>
</div>
</body>
</html>"""

        path = self.output_dir / "dashboard.html"
        path.write_text(html_content)
        logger.info("Generated combined dashboard: %s", path)
        return path

    def _create_kd_ratio_figure(
        self, metrics: list[GenerationMetrics]
    ) -> Figure:
        """Create K/D ratio figure without saving."""
        import matplotlib.pyplot as plt

        generations = [m.generation_id for m in metrics]
        kd_ratios = [m.kill_death_ratio for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            generations,
            kd_ratios,
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2563eb",
        )
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Kill/Death Ratio", fontsize=12)
        ax.set_title("K/D Ratio Progression Across Generations", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(generations)

        for gen, kd in zip(generations, kd_ratios, strict=False):
            ax.annotate(
                f"{kd:.2f}",
                (gen, kd),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        return fig

    def _create_win_rate_figure(
        self, metrics: list[GenerationMetrics]
    ) -> Figure:
        """Create win rate figure without saving."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        generations = [m.generation_id for m in metrics]
        win_rates = [m.win_rate * 100 for m in metrics]
        opponents = [m.opponent_type for m in metrics]

        colors = []
        for opp in opponents:
            if opp == "baseline":
                colors.append("#10b981")
            else:
                colors.append("#6366f1")

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(generations, win_rates, color=colors, edgecolor="white")

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_title("Win Rate by Generation", fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_xticks(generations)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, win_rate in zip(bars, win_rates, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{win_rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        legend_elements = [
            Patch(facecolor="#10b981", label="vs Baseline"),
            Patch(facecolor="#6366f1", label="vs Previous Gen"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        return fig

    def _create_reward_figure(
        self, metrics: list[GenerationMetrics]
    ) -> Figure:
        """Create reward figure without saving."""
        import matplotlib.pyplot as plt

        generations = [m.generation_id for m in metrics]
        rewards = [m.avg_episode_reward for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            generations,
            rewards,
            marker="s",
            linewidth=2,
            markersize=8,
            color="#f59e0b",
        )
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Average Episode Reward", fontsize=12)
        ax.set_title("Average Reward Progression Across Generations", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(generations)

        for gen, reward in zip(generations, rewards, strict=False):
            ax.annotate(
                f"{reward:.1f}",
                (gen, reward),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        return fig

    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        import base64
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _build_table_html(self, table_data: list[dict]) -> str:
        """Build HTML table from table data."""
        if not table_data:
            return "<p>No data available</p>"

        headers = list(table_data[0].keys())
        header_row = "".join(f"<th>{h}</th>" for h in headers)

        rows = []
        for row_data in table_data:
            cells = "".join(f"<td>{row_data[h]}</td>" for h in headers)
            rows.append(f"<tr>{cells}</tr>")

        return f"""
<table>
<thead><tr>{header_row}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""
