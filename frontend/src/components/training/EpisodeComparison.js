import React, { useState, useEffect, useCallback } from 'react';
import EpisodeReplayControls from './EpisodeReplayControls';
import './EpisodeComparison.css';

const EpisodeComparison = ({
  episodes,
  comparisonReplayId,
  replayStatus,
  onCommand,
  onSpeedChange,
  onSeek,
  onClose,
  className = ''
}) => {
  const [selectedEpisodes, setSelectedEpisodes] = useState([]);
  const [comparisonData, setComparisonData] = useState(null);
  const [viewMode, setViewMode] = useState('side-by-side'); // 'side-by-side', 'overlay', 'metrics'

  // Initialize selected episodes
  useEffect(() => {
    if (episodes && episodes.length > 0) {
      setSelectedEpisodes(episodes.slice(0, 4)); // Max 4 episodes
    }
  }, [episodes]);

  // Handle episode selection changes
  const handleEpisodeToggle = useCallback((episode) => {
    setSelectedEpisodes(prev => {
      const isSelected = prev.some(ep => ep.episode_id === episode.episode_id);
      
      if (isSelected) {
        return prev.filter(ep => ep.episode_id !== episode.episode_id);
      } else if (prev.length < 4) {
        return [...prev, episode];
      } else {
        // Replace the first episode if at max capacity
        return [episode, ...prev.slice(1)];
      }
    });
  }, []);

  // Format episode display info
  const formatEpisodeInfo = (episode) => {
    return {
      id: episode.episode_id,
      generation: episode.model_generation,
      opponent: episode.opponent_generation || 'N/A',
      result: episode.game_result,
      reward: episode.total_reward?.toFixed(1) || '0.0',
      length: episode.episode_length || 0
    };
  };

  // Get comparison metrics
  const getComparisonMetrics = () => {
    if (!selectedEpisodes || selectedEpisodes.length < 2) return null;

    const metrics = {
      generations: [...new Set(selectedEpisodes.map(ep => ep.model_generation))].sort(),
      results: {
        wins: selectedEpisodes.filter(ep => ep.game_result === 'win').length,
        losses: selectedEpisodes.filter(ep => ep.game_result === 'loss').length,
        draws: selectedEpisodes.filter(ep => ep.game_result === 'draw').length
      },
      rewards: {
        min: Math.min(...selectedEpisodes.map(ep => ep.total_reward || 0)),
        max: Math.max(...selectedEpisodes.map(ep => ep.total_reward || 0)),
        avg: selectedEpisodes.reduce((sum, ep) => sum + (ep.total_reward || 0), 0) / selectedEpisodes.length
      },
      lengths: {
        min: Math.min(...selectedEpisodes.map(ep => ep.episode_length || 0)),
        max: Math.max(...selectedEpisodes.map(ep => ep.episode_length || 0)),
        avg: selectedEpisodes.reduce((sum, ep) => sum + (ep.episode_length || 0), 0) / selectedEpisodes.length
      }
    };

    return metrics;
  };

  const comparisonMetrics = getComparisonMetrics();

  return (
    <div className={`episode-comparison ${className}`}>
      {/* Header */}
      <div className="comparison-header">
        <h3>Episode Comparison</h3>
        <div className="header-controls">
          <div className="view-mode-selector">
            <button
              className={`view-btn ${viewMode === 'side-by-side' ? 'active' : ''}`}
              onClick={() => setViewMode('side-by-side')}
            >
              Side-by-Side
            </button>
            <button
              className={`view-btn ${viewMode === 'overlay' ? 'active' : ''}`}
              onClick={() => setViewMode('overlay')}
            >
              Overlay
            </button>
            <button
              className={`view-btn ${viewMode === 'metrics' ? 'active' : ''}`}
              onClick={() => setViewMode('metrics')}
            >
              Metrics
            </button>
          </div>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
      </div>

      {/* Episode Selection */}
      <div className="episode-selection">
        <h4>Selected Episodes ({selectedEpisodes.length}/4)</h4>
        <div className="selected-episodes">
          {selectedEpisodes.map((episode, index) => {
            const info = formatEpisodeInfo(episode);
            return (
              <div key={episode.episode_id} className={`selected-episode episode-${index}`}>
                <div className="episode-header">
                  <span className="episode-label">Episode {index + 1}</span>
                  <button
                    className="remove-btn"
                    onClick={() => handleEpisodeToggle(episode)}
                    title="Remove episode"
                  >
                    ×
                  </button>
                </div>
                <div className="episode-info">
                  <div className="info-row">
                    <span className="label">Gen:</span>
                    <span className="value">{info.generation}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Result:</span>
                    <span className={`value result-${info.result}`}>{info.result}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Reward:</span>
                    <span className="value">{info.reward}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Length:</span>
                    <span className="value">{info.length}</span>
                  </div>
                </div>
              </div>
            );
          })}
          
          {/* Add episode slots */}
          {selectedEpisodes.length < 4 && (
            <div className="add-episode-slot">
              <div className="add-episode-prompt">
                <span>Select episodes to compare</span>
                <div className="episode-count">
                  {selectedEpisodes.length}/4 selected
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Metrics */}
      {comparisonMetrics && viewMode === 'metrics' && (
        <div className="comparison-metrics">
          <h4>Comparison Metrics</h4>
          <div className="metrics-grid">
            <div className="metric-group">
              <h5>Generations</h5>
              <div className="metric-value">
                {comparisonMetrics.generations.join(', ')}
              </div>
            </div>
            
            <div className="metric-group">
              <h5>Results</h5>
              <div className="results-breakdown">
                <div className="result-item">
                  <span className="result-label wins">Wins:</span>
                  <span className="result-value">{comparisonMetrics.results.wins}</span>
                </div>
                <div className="result-item">
                  <span className="result-label losses">Losses:</span>
                  <span className="result-value">{comparisonMetrics.results.losses}</span>
                </div>
                <div className="result-item">
                  <span className="result-label draws">Draws:</span>
                  <span className="result-value">{comparisonMetrics.results.draws}</span>
                </div>
              </div>
            </div>
            
            <div className="metric-group">
              <h5>Rewards</h5>
              <div className="metric-stats">
                <div className="stat-item">
                  <span className="stat-label">Min:</span>
                  <span className="stat-value">{comparisonMetrics.rewards.min.toFixed(1)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Max:</span>
                  <span className="stat-value">{comparisonMetrics.rewards.max.toFixed(1)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Avg:</span>
                  <span className="stat-value">{comparisonMetrics.rewards.avg.toFixed(1)}</span>
                </div>
              </div>
            </div>
            
            <div className="metric-group">
              <h5>Episode Length</h5>
              <div className="metric-stats">
                <div className="stat-item">
                  <span className="stat-label">Min:</span>
                  <span className="stat-value">{Math.round(comparisonMetrics.lengths.min)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Max:</span>
                  <span className="stat-value">{Math.round(comparisonMetrics.lengths.max)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Avg:</span>
                  <span className="stat-value">{Math.round(comparisonMetrics.lengths.avg)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Replay Controls */}
      {comparisonReplayId && (
        <div className="comparison-controls">
          <EpisodeReplayControls
            replayId={comparisonReplayId}
            replayStatus={replayStatus}
            onCommand={onCommand}
            onSpeedChange={onSpeedChange}
            onSeek={onSeek}
            className="comparison-replay-controls"
          />
        </div>
      )}

      {/* Comparison View */}
      <div className={`comparison-view view-${viewMode}`}>
        {viewMode === 'side-by-side' && (
          <div className="side-by-side-view">
            {selectedEpisodes.map((episode, index) => (
              <div key={episode.episode_id} className={`episode-panel panel-${index}`}>
                <div className="panel-header">
                  <h5>Episode {index + 1}</h5>
                  <div className="panel-info">
                    Gen {episode.model_generation} • {episode.game_result}
                  </div>
                </div>
                <div className="panel-content">
                  {/* This would contain the actual game visualization */}
                  <div className="game-view-placeholder">
                    <div className="placeholder-text">
                      Game View {index + 1}
                    </div>
                    <div className="placeholder-info">
                      Reward: {episode.total_reward?.toFixed(1) || '0.0'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'overlay' && (
          <div className="overlay-view">
            <div className="overlay-container">
              <div className="overlay-game-view">
                <div className="game-view-placeholder large">
                  <div className="placeholder-text">
                    Overlay Comparison View
                  </div>
                  <div className="placeholder-info">
                    {selectedEpisodes.length} episodes overlaid
                  </div>
                </div>
              </div>
              <div className="overlay-legend">
                {selectedEpisodes.map((episode, index) => (
                  <div key={episode.episode_id} className={`legend-item episode-${index}`}>
                    <div className={`legend-color color-${index}`}></div>
                    <span>Gen {episode.model_generation} ({episode.game_result})</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Available Episodes List (for selection) */}
      {episodes && episodes.length > selectedEpisodes.length && (
        <div className="available-episodes">
          <h4>Available Episodes</h4>
          <div className="episodes-list">
            {episodes
              .filter(ep => !selectedEpisodes.some(sel => sel.episode_id === ep.episode_id))
              .slice(0, 10) // Show max 10 available episodes
              .map(episode => {
                const info = formatEpisodeInfo(episode);
                return (
                  <div
                    key={episode.episode_id}
                    className="available-episode"
                    onClick={() => handleEpisodeToggle(episode)}
                  >
                    <div className="episode-summary">
                      <span className="episode-id">Gen {info.generation}</span>
                      <span className={`episode-result result-${info.result}`}>
                        {info.result}
                      </span>
                      <span className="episode-reward">{info.reward}</span>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
};

export default EpisodeComparison;