"""
Replay analysis tools for behavior pattern detection and episode analysis.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

from rl_bot_system.evaluation.evaluator import GameEpisode


@dataclass
class BehaviorPattern:
    """Represents a detected behavior pattern."""
    pattern_id: str
    pattern_type: str  # 'action_sequence', 'state_preference', 'strategy'
    description: str
    frequency: int
    confidence: float
    episodes: List[str]  # Episode IDs where pattern was observed
    metadata: Dict[str, Any]


@dataclass
class AnalysisConfig:
    """Configuration for replay analysis."""
    min_pattern_frequency: int = 3
    min_confidence_threshold: float = 0.7
    sequence_length: int = 5
    analyze_action_sequences: bool = True
    analyze_state_preferences: bool = True
    analyze_reward_patterns: bool = True
    analyze_strategic_behavior: bool = True


class ReplayAnalyzer:
    """
    Analyzes recorded game episodes to detect behavior patterns and strategies.
    
    Features:
    - Action sequence pattern detection
    - State preference analysis
    - Strategic behavior identification
    - Performance correlation analysis
    - Comparative analysis between model generations
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the replay analyzer.
        
        Args:
            config: Analysis configuration options
        """
        self.config = config or AnalysisConfig()
        self.detected_patterns = []
        
    def analyze_episodes(self, episodes: List[GameEpisode]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a list of episodes.
        
        Args:
            episodes: List of GameEpisode objects to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including patterns and statistics
        """
        if not episodes:
            return {"error": "No episodes provided for analysis"}
            
        analysis_results = {
            "episode_count": len(episodes),
            "model_generations": self._get_generation_stats(episodes),
            "performance_stats": self._calculate_performance_stats(episodes),
            "patterns": [],
            "behavioral_insights": {}
        }
        
        # Detect various types of patterns
        if self.config.analyze_action_sequences:
            action_patterns = self._detect_action_sequences(episodes)
            analysis_results["patterns"].extend(action_patterns)
            
        if self.config.analyze_state_preferences:
            state_patterns = self._detect_state_preferences(episodes)
            analysis_results["patterns"].extend(state_patterns)
            
        if self.config.analyze_reward_patterns:
            reward_patterns = self._analyze_reward_patterns(episodes)
            analysis_results["patterns"].extend(reward_patterns)
            
        if self.config.analyze_strategic_behavior:
            strategic_patterns = self._detect_strategic_behavior(episodes)
            analysis_results["patterns"].extend(strategic_patterns)
            
        # Generate behavioral insights
        analysis_results["behavioral_insights"] = self._generate_behavioral_insights(
            episodes, analysis_results["patterns"]
        )
        
        # Convert patterns to dictionaries for JSON serialization
        analysis_results["patterns"] = [asdict(p) for p in analysis_results["patterns"]]
        
        # Store detected patterns
        self.detected_patterns = analysis_results["patterns"]
        
        return analysis_results
    
    def compare_generations(
        self, 
        episodes_by_generation: Dict[int, List[GameEpisode]]
    ) -> Dict[str, Any]:
        """
        Compare behavior patterns across different model generations.
        
        Args:
            episodes_by_generation: Episodes grouped by model generation
            
        Returns:
            Dict[str, Any]: Comparative analysis results
        """
        comparison_results = {
            "generations_analyzed": list(episodes_by_generation.keys()),
            "generation_comparisons": {},
            "evolution_trends": {},
            "performance_progression": {}
        }
        
        # Analyze each generation separately
        generation_analyses = {}
        for generation, episodes in episodes_by_generation.items():
            generation_analyses[generation] = self.analyze_episodes(episodes)
            
        # Compare patterns between generations
        for gen in sorted(episodes_by_generation.keys()):
            if gen > 0 and (gen - 1) in generation_analyses:
                comparison = self._compare_generation_patterns(
                    generation_analyses[gen - 1], 
                    generation_analyses[gen]
                )
                comparison_results["generation_comparisons"][f"{gen-1}_to_{gen}"] = comparison
                
        # Identify evolution trends
        comparison_results["evolution_trends"] = self._identify_evolution_trends(
            generation_analyses
        )
        
        # Calculate performance progression
        comparison_results["performance_progression"] = self._calculate_performance_progression(
            episodes_by_generation
        )
        
        return comparison_results
    
    def export_analysis(
        self, 
        analysis_results: Dict[str, Any], 
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export analysis results to file.
        
        Args:
            analysis_results: Results from analyze_episodes or compare_generations
            output_path: Path to save the analysis
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            str: Path to the exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            return self._export_json(analysis_results, output_path)
        elif format.lower() == "csv":
            return self._export_csv(analysis_results, output_path)
        elif format.lower() == "html":
            return self._export_html(analysis_results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _get_generation_stats(self, episodes: List[GameEpisode]) -> Dict[int, int]:
        """Get statistics about model generations in episodes."""
        generation_counts = Counter(episode.model_generation for episode in episodes)
        return dict(generation_counts)
    
    def _calculate_performance_stats(self, episodes: List[GameEpisode]) -> Dict[str, Any]:
        """Calculate performance statistics for episodes."""
        if not episodes:
            return {}
            
        total_rewards = [episode.total_reward for episode in episodes]
        episode_lengths = [episode.episode_length for episode in episodes]
        win_count = sum(1 for episode in episodes if episode.game_result == "win")
        
        return {
            "total_episodes": len(episodes),
            "win_rate": win_count / len(episodes) if episodes else 0,
            "average_reward": np.mean(total_rewards) if total_rewards else 0,
            "reward_std": np.std(total_rewards) if total_rewards else 0,
            "average_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "min_reward": min(total_rewards) if total_rewards else 0,
            "max_reward": max(total_rewards) if total_rewards else 0
        }
    
    def _detect_action_sequences(self, episodes: List[GameEpisode]) -> List[BehaviorPattern]:
        """Detect common action sequences across episodes."""
        patterns = []
        sequence_counts = defaultdict(int)
        sequence_episodes = defaultdict(set)
        
        # Extract action sequences
        for episode in episodes:
            if not episode.actions:
                continue
                
            for i in range(len(episode.actions) - self.config.sequence_length + 1):
                sequence = tuple(episode.actions[i:i + self.config.sequence_length])
                sequence_counts[sequence] += 1
                sequence_episodes[sequence].add(episode.episode_id)
        
        # Identify significant patterns
        for sequence, count in sequence_counts.items():
            if count >= self.config.min_pattern_frequency:
                confidence = count / len(episodes)
                if confidence >= self.config.min_confidence_threshold:
                    pattern = BehaviorPattern(
                        pattern_id=f"action_seq_{hash(sequence)}",
                        pattern_type="action_sequence",
                        description=f"Action sequence: {sequence}",
                        frequency=count,
                        confidence=confidence,
                        episodes=list(sequence_episodes[sequence]),
                        metadata={"sequence": sequence, "length": len(sequence)}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_state_preferences(self, episodes: List[GameEpisode]) -> List[BehaviorPattern]:
        """Detect preferences for certain game states."""
        patterns = []
        
        # Analyze state features (simplified - would need domain-specific implementation)
        state_features = defaultdict(list)
        
        for episode in episodes:
            if not episode.states:
                continue
                
            for state in episode.states:
                # Extract key features from state (this would be game-specific)
                if isinstance(state, dict):
                    for key, value in state.items():
                        if isinstance(value, (int, float)):
                            state_features[key].append(value)
        
        # Identify preferences based on feature distributions
        for feature, values in state_features.items():
            if len(values) > 10:  # Minimum sample size
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Check if there's a clear preference (low variance relative to mean)
                if std_val / (abs(mean_val) + 1e-6) < 0.5:  # Coefficient of variation
                    pattern = BehaviorPattern(
                        pattern_id=f"state_pref_{feature}",
                        pattern_type="state_preference",
                        description=f"Preference for {feature} around {mean_val:.2f}",
                        frequency=len(values),
                        confidence=1.0 - (std_val / (abs(mean_val) + 1e-6)),
                        episodes=[ep.episode_id for ep in episodes if ep.states],
                        metadata={
                            "feature": feature,
                            "preferred_value": mean_val,
                            "variance": std_val
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_reward_patterns(self, episodes: List[GameEpisode]) -> List[BehaviorPattern]:
        """Analyze reward accumulation patterns."""
        patterns = []
        
        # Analyze reward timing and distribution
        reward_timings = []
        for episode in episodes:
            if episode.rewards:
                # Find when significant rewards occur
                significant_rewards = []
                if episode.rewards:
                    reward_std = np.std(episode.rewards)
                    significant_rewards = [
                        (i, reward) for i, reward in enumerate(episode.rewards) 
                        if abs(reward) > reward_std
                    ]
                reward_timings.extend(significant_rewards)
        
        if reward_timings:
            # Analyze timing patterns - need to get episode length for each timing
            timing_positions = []
            for episode in episodes:
                if episode.rewards:
                    reward_std = np.std(episode.rewards)
                    for i, reward in enumerate(episode.rewards):
                        if abs(reward) > reward_std:
                            timing_positions.append(i / len(episode.rewards))
            
            if len(timing_positions) >= 3:
                mean_timing = np.mean(timing_positions)
                
                pattern = BehaviorPattern(
                    pattern_id="reward_timing_pattern",
                    pattern_type="reward_pattern",
                    description=f"Significant rewards typically occur at {mean_timing:.1%} of episode",
                    frequency=len(reward_timings),
                    confidence=0.8,  # Fixed confidence for timing patterns
                    episodes=[ep.episode_id for ep in episodes if ep.rewards],
                    metadata={
                        "average_timing": mean_timing,
                        "timing_variance": np.var(timing_positions)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_strategic_behavior(self, episodes: List[GameEpisode]) -> List[BehaviorPattern]:
        """Detect high-level strategic behavior patterns."""
        patterns = []
        
        # Analyze win/loss patterns
        winning_episodes = [ep for ep in episodes if ep.game_result == "win"]
        losing_episodes = [ep for ep in episodes if ep.game_result == "loss"]
        
        if winning_episodes and losing_episodes:
            # Compare episode lengths
            win_lengths = [ep.episode_length for ep in winning_episodes]
            loss_lengths = [ep.episode_length for ep in losing_episodes]
            
            avg_win_length = np.mean(win_lengths)
            avg_loss_length = np.mean(loss_lengths)
            
            if abs(avg_win_length - avg_loss_length) > 2:  # Significant difference
                strategy_type = "quick_wins" if avg_win_length < avg_loss_length else "endurance_strategy"
                
                pattern = BehaviorPattern(
                    pattern_id=f"strategic_{strategy_type}",
                    pattern_type="strategy",
                    description=f"Tends to {strategy_type.replace('_', ' ')} (win length: {avg_win_length:.1f}, loss length: {avg_loss_length:.1f})",
                    frequency=len(winning_episodes),
                    confidence=0.7,
                    episodes=[ep.episode_id for ep in winning_episodes],
                    metadata={
                        "average_win_length": avg_win_length,
                        "average_loss_length": avg_loss_length,
                        "strategy_type": strategy_type
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _generate_behavioral_insights(
        self, 
        episodes: List[GameEpisode], 
        patterns: List[BehaviorPattern]
    ) -> Dict[str, Any]:
        """Generate high-level behavioral insights from patterns."""
        insights = {
            "dominant_strategies": [],
            "learning_indicators": [],
            "consistency_metrics": {},
            "improvement_areas": []
        }
        
        # Identify dominant strategies
        strategy_patterns = [p for p in patterns if p.pattern_type == "strategy"]
        insights["dominant_strategies"] = [
            {"description": p.description, "confidence": p.confidence}
            for p in sorted(strategy_patterns, key=lambda x: x.confidence, reverse=True)[:3]
        ]
        
        # Calculate consistency metrics
        if episodes:
            rewards = [ep.total_reward for ep in episodes]
            insights["consistency_metrics"] = {
                "reward_consistency": 1.0 - (np.std(rewards) / (np.mean(rewards) + 1e-6)),
                "performance_trend": self._calculate_trend(rewards),
                "episode_length_consistency": 1.0 - (
                    np.std([ep.episode_length for ep in episodes]) / 
                    (np.mean([ep.episode_length for ep in episodes]) + 1e-6)
                )
            }
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "insufficient_data"
            
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _compare_generation_patterns(
        self, 
        prev_analysis: Dict[str, Any], 
        curr_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare patterns between two generations."""
        comparison = {
            "new_patterns": [],
            "lost_patterns": [],
            "evolved_patterns": [],
            "performance_change": {}
        }
        
        # Handle both BehaviorPattern objects and dictionaries
        prev_patterns = {}
        for p in prev_analysis.get("patterns", []):
            if isinstance(p, dict):
                prev_patterns[p["pattern_id"]] = p
            else:
                prev_patterns[p.pattern_id] = asdict(p)
                
        curr_patterns = {}
        for p in curr_analysis.get("patterns", []):
            if isinstance(p, dict):
                curr_patterns[p["pattern_id"]] = p
            else:
                curr_patterns[p.pattern_id] = asdict(p)
        
        # Find new and lost patterns
        comparison["new_patterns"] = [
            p for p_id, p in curr_patterns.items() if p_id not in prev_patterns
        ]
        comparison["lost_patterns"] = [
            p for p_id, p in prev_patterns.items() if p_id not in curr_patterns
        ]
        
        # Find evolved patterns (same ID but different characteristics)
        for p_id in set(prev_patterns.keys()) & set(curr_patterns.keys()):
            prev_p = prev_patterns[p_id]
            curr_p = curr_patterns[p_id]
            
            if abs(prev_p["confidence"] - curr_p["confidence"]) > 0.1:
                comparison["evolved_patterns"].append({
                    "pattern_id": p_id,
                    "previous_confidence": prev_p["confidence"],
                    "current_confidence": curr_p["confidence"],
                    "change": curr_p["confidence"] - prev_p["confidence"]
                })
        
        # Compare performance
        prev_perf = prev_analysis.get("performance_stats", {})
        curr_perf = curr_analysis.get("performance_stats", {})
        
        for metric in ["win_rate", "average_reward", "average_episode_length"]:
            if metric in prev_perf and metric in curr_perf:
                comparison["performance_change"][metric] = {
                    "previous": prev_perf[metric],
                    "current": curr_perf[metric],
                    "change": curr_perf[metric] - prev_perf[metric]
                }
        
        return comparison
    
    def _identify_evolution_trends(self, generation_analyses: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trends across multiple generations."""
        trends = {
            "performance_trajectory": {},
            "pattern_evolution": {},
            "learning_progression": {}
        }
        
        generations = sorted(generation_analyses.keys())
        
        # Track performance metrics over generations
        for metric in ["win_rate", "average_reward", "average_episode_length"]:
            values = []
            for gen in generations:
                perf_stats = generation_analyses[gen].get("performance_stats", {})
                if metric in perf_stats:
                    values.append(perf_stats[metric])
            
            if len(values) > 1:
                trends["performance_trajectory"][metric] = {
                    "values": values,
                    "trend": self._calculate_trend(values),
                    "improvement_rate": (values[-1] - values[0]) / len(values) if values else 0
                }
        
        return trends
    
    def _calculate_performance_progression(
        self, 
        episodes_by_generation: Dict[int, List[GameEpisode]]
    ) -> Dict[str, Any]:
        """Calculate performance progression across generations."""
        progression = {}
        
        for generation, episodes in sorted(episodes_by_generation.items()):
            stats = self._calculate_performance_stats(episodes)
            progression[generation] = stats
        
        return progression
    
    def _export_json(self, analysis_results: Dict[str, Any], output_path: Path) -> str:
        """Export analysis results as JSON."""
        json_path = output_path.with_suffix('.json')
        
        # Convert BehaviorPattern objects to dictionaries
        exportable_results = self._make_json_serializable(analysis_results)
        
        with open(json_path, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        return str(json_path)
    
    def _export_csv(self, analysis_results: Dict[str, Any], output_path: Path) -> str:
        """Export analysis results as CSV."""
        import csv
        
        csv_path = output_path.with_suffix('.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write patterns
            writer.writerow(['Pattern ID', 'Type', 'Description', 'Frequency', 'Confidence'])
            for pattern in analysis_results.get('patterns', []):
                if isinstance(pattern, dict):
                    writer.writerow([
                        pattern.get('pattern_id', ''),
                        pattern.get('pattern_type', ''),
                        pattern.get('description', ''),
                        pattern.get('frequency', 0),
                        pattern.get('confidence', 0.0)
                    ])
        
        return str(csv_path)
    
    def _export_html(self, analysis_results: Dict[str, Any], output_path: Path) -> str:
        """Export analysis results as HTML report."""
        html_path = output_path.with_suffix('.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Replay Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .pattern {{ margin: 10px 0; padding: 10px; border: 1px solid #ccc; }}
                .stats {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Replay Analysis Report</h1>
            
            <div class="stats">
                <h2>Performance Statistics</h2>
                <p>Episodes analyzed: {analysis_results.get('episode_count', 0)}</p>
                <p>Patterns detected: {len(analysis_results.get('patterns', []))}</p>
            </div>
            
            <h2>Detected Patterns</h2>
        """
        
        for pattern in analysis_results.get('patterns', []):
            if isinstance(pattern, dict):
                html_content += f"""
                <div class="pattern">
                    <h3>{pattern.get('pattern_type', 'Unknown').title()}: {pattern.get('pattern_id', '')}</h3>
                    <p><strong>Description:</strong> {pattern.get('description', '')}</p>
                    <p><strong>Frequency:</strong> {pattern.get('frequency', 0)}</p>
                    <p><strong>Confidence:</strong> {pattern.get('confidence', 0.0):.2f}</p>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, BehaviorPattern):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj