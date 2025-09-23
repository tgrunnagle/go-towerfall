import React, { useState, useEffect, useCallback } from 'react';
import EpisodeReplayControls from './EpisodeReplayControls';
import EpisodeComparison from './EpisodeComparison';
import './EpisodeBrowser.css';

const EpisodeBrowser = ({
  sessionId,
  websocketConnection,
  onClose,
  className = ''
}) => {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(sessionId || '');
  const [episodes, setEpisodes] = useState([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Replay state
  const [activeReplayId, setActiveReplayId] = useState(null);
  const [replayStatus, setReplayStatus] = useState(null);
  const [showComparison, setShowComparison] = useState(false);
  
  // Filters and sorting
  const [filters, setFilters] = useState({
    generation: '',
    result: '',
    minReward: '',
    maxReward: ''
  });
  const [sortBy, setSortBy] = useState('episode_id');
  const [sortOrder, setSortOrder] = useState('desc');
  
  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [episodesPerPage] = useState(20);

  // Load available sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Load episodes when session changes
  useEffect(() => {
    if (selectedSession) {
      loadEpisodes(selectedSession);
    }
  }, [selectedSession]);

  // WebSocket message handling for replay updates
  useEffect(() => {
    if (!websocketConnection) return;

    const handleMessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'replay_frame') {
          // Handle replay frame updates
          setReplayStatus(prev => ({
            ...prev,
            current_frame: message.frame_index,
            progress: message.progress
          }));
        } else if (message.type === 'replay_status') {
          // Handle replay status updates
          setReplayStatus(message.status);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocketConnection.addEventListener('message', handleMessage);
    
    return () => {
      websocketConnection.removeEventListener('message', handleMessage);
    };
  }, [websocketConnection]);

  // API calls
  const loadSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/replay/sessions');
      if (!response.ok) throw new Error('Failed to load sessions');
      
      const sessionsData = await response.json();
      setSessions(sessionsData);
      
      if (!selectedSession && sessionsData.length > 0) {
        setSelectedSession(sessionsData[0].session_id);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadEpisodes = async (sessionId) => {
    try {
      setLoading(true);
      const response = await fetch(`/api/replay/sessions/${sessionId}/episodes?limit=1000`);
      if (!response.ok) throw new Error('Failed to load episodes');
      
      const data = await response.json();
      setEpisodes(data.episodes);
      setCurrentPage(1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Episode selection
  const handleEpisodeSelect = useCallback((episode, isMultiSelect = false) => {
    if (isMultiSelect) {
      setSelectedEpisodes(prev => {
        const isSelected = prev.some(ep => ep.episode_id === episode.episode_id);
        if (isSelected) {
          return prev.filter(ep => ep.episode_id !== episode.episode_id);
        } else {
          return [...prev, episode];
        }
      });
    } else {
      setSelectedEpisodes([episode]);
    }
  }, []);

  // Replay controls
  const startReplay = async (episodeId) => {
    try {
      const response = await fetch('/api/replay/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: selectedSession,
          episode_id: episodeId,
          controls: {
            playback_speed: 1.0,
            show_frame_info: true,
            show_decision_overlay: true
          }
        })
      });
      
      if (!response.ok) throw new Error('Failed to start replay');
      
      const data = await response.json();
      setActiveReplayId(data.replay_id);
      
      // Load initial status
      loadReplayStatus(data.replay_id);
    } catch (err) {
      setError(err.message);
    }
  };

  const startComparison = async (episodeIds) => {
    try {
      const response = await fetch('/api/replay/start-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: selectedSession,
          episode_ids: episodeIds,
          controls: {
            playback_speed: 1.0,
            comparison_mode: true
          }
        })
      });
      
      if (!response.ok) throw new Error('Failed to start comparison');
      
      const data = await response.json();
      setActiveReplayId(data.replay_id);
      setShowComparison(true);
      
      // Load initial status
      loadReplayStatus(data.replay_id);
    } catch (err) {
      setError(err.message);
    }
  };

  const loadReplayStatus = async (replayId) => {
    try {
      const response = await fetch(`/api/replay/status/${replayId}`);
      if (!response.ok) throw new Error('Failed to load replay status');
      
      const data = await response.json();
      setReplayStatus(data.status);
    } catch (err) {
      console.error('Error loading replay status:', err);
    }
  };

  const sendReplayCommand = async (command, parameters = {}) => {
    if (!activeReplayId) return;
    
    try {
      const response = await fetch(`/api/replay/control/${activeReplayId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, parameters })
      });
      
      if (!response.ok) throw new Error('Failed to send replay command');
      
      const data = await response.json();
      setReplayStatus(data.replay_status);
    } catch (err) {
      setError(err.message);
    }
  };

  const stopReplay = async () => {
    if (!activeReplayId) return;
    
    try {
      await fetch(`/api/replay/stop/${activeReplayId}`, { method: 'DELETE' });
      setActiveReplayId(null);
      setReplayStatus(null);
      setShowComparison(false);
    } catch (err) {
      setError(err.message);
    }
  };

  // Filtering and sorting
  const getFilteredEpisodes = () => {
    let filtered = [...episodes];
    
    // Apply filters
    if (filters.generation) {
      filtered = filtered.filter(ep => ep.model_generation.toString() === filters.generation);
    }
    if (filters.result) {
      filtered = filtered.filter(ep => ep.game_result === filters.result);
    }
    if (filters.minReward) {
      filtered = filtered.filter(ep => ep.total_reward >= parseFloat(filters.minReward));
    }
    if (filters.maxReward) {
      filtered = filtered.filter(ep => ep.total_reward <= parseFloat(filters.maxReward));
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];
      
      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }
      
      if (sortOrder === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
    
    return filtered;
  };

  const filteredEpisodes = getFilteredEpisodes();
  const totalPages = Math.ceil(filteredEpisodes.length / episodesPerPage);
  const paginatedEpisodes = filteredEpisodes.slice(
    (currentPage - 1) * episodesPerPage,
    currentPage * episodesPerPage
  );

  // Get unique values for filter options
  const getUniqueGenerations = () => {
    return [...new Set(episodes.map(ep => ep.model_generation))].sort((a, b) => a - b);
  };

  const getUniqueResults = () => {
    return [...new Set(episodes.map(ep => ep.game_result))];
  };

  return (
    <div className={`episode-browser ${className}`}>
      {/* Header */}
      <div className="browser-header">
        <h3>Episode Browser</h3>
        <div className="header-actions">
          {selectedEpisodes.length > 1 && (
            <button
              className="compare-btn"
              onClick={() => startComparison(selectedEpisodes.map(ep => ep.episode_id))}
            >
              Compare Selected ({selectedEpisodes.length})
            </button>
          )}
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
      </div>

      {/* Session Selector */}
      <div className="session-selector">
        <label htmlFor="session-select">Session:</label>
        <select
          id="session-select"
          value={selectedSession}
          onChange={(e) => setSelectedSession(e.target.value)}
          disabled={loading}
        >
          <option value="">Select a session...</option>
          {sessions.map(session => (
            <option key={session.session_id} value={session.session_id}>
              {session.session_id} ({new Date(session.start_time).toLocaleDateString()})
            </option>
          ))}
        </select>
      </div>

      {/* Filters */}
      <div className="episode-filters">
        <div className="filter-group">
          <label>Generation:</label>
          <select
            value={filters.generation}
            onChange={(e) => setFilters(prev => ({ ...prev, generation: e.target.value }))}
          >
            <option value="">All</option>
            {getUniqueGenerations().map(gen => (
              <option key={gen} value={gen}>{gen}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Result:</label>
          <select
            value={filters.result}
            onChange={(e) => setFilters(prev => ({ ...prev, result: e.target.value }))}
          >
            <option value="">All</option>
            {getUniqueResults().map(result => (
              <option key={result} value={result}>{result}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label>Min Reward:</label>
          <input
            type="number"
            value={filters.minReward}
            onChange={(e) => setFilters(prev => ({ ...prev, minReward: e.target.value }))}
            placeholder="Min"
          />
        </div>

        <div className="filter-group">
          <label>Max Reward:</label>
          <input
            type="number"
            value={filters.maxReward}
            onChange={(e) => setFilters(prev => ({ ...prev, maxReward: e.target.value }))}
            placeholder="Max"
          />
        </div>

        <div className="filter-group">
          <label>Sort by:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          >
            <option value="episode_id">Episode ID</option>
            <option value="model_generation">Generation</option>
            <option value="total_reward">Reward</option>
            <option value="episode_length">Length</option>
            <option value="game_result">Result</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Order:</label>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
          >
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          Error: {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="loading-message">
          Loading episodes...
        </div>
      )}

      {/* Episodes List */}
      {!loading && paginatedEpisodes.length > 0 && (
        <div className="episodes-section">
          <div className="episodes-header">
            <span>
              Showing {paginatedEpisodes.length} of {filteredEpisodes.length} episodes
            </span>
            {selectedEpisodes.length > 0 && (
              <span className="selection-info">
                {selectedEpisodes.length} selected
              </span>
            )}
          </div>

          <div className="episodes-list">
            {paginatedEpisodes.map(episode => {
              const isSelected = selectedEpisodes.some(ep => ep.episode_id === episode.episode_id);
              
              return (
                <div
                  key={episode.episode_id}
                  className={`episode-item ${isSelected ? 'selected' : ''}`}
                  onClick={(e) => handleEpisodeSelect(episode, e.ctrlKey || e.metaKey)}
                >
                  <div className="episode-info">
                    <div className="episode-id">
                      {episode.episode_id}
                    </div>
                    <div className="episode-details">
                      <span className="generation">Gen {episode.model_generation}</span>
                      <span className={`result result-${episode.game_result}`}>
                        {episode.game_result}
                      </span>
                      <span className="reward">
                        {episode.total_reward?.toFixed(1) || '0.0'}
                      </span>
                      <span className="length">
                        {episode.episode_length} frames
                      </span>
                    </div>
                  </div>
                  <div className="episode-actions">
                    <button
                      className="replay-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        startReplay(episode.episode_id);
                      }}
                    >
                      ▶ Replay
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="pagination">
              <button
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
              >
                Previous
              </button>
              
              <span className="page-info">
                Page {currentPage} of {totalPages}
              </span>
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}

      {/* No Episodes Message */}
      {!loading && paginatedEpisodes.length === 0 && episodes.length > 0 && (
        <div className="no-episodes-message">
          No episodes match the current filters.
        </div>
      )}

      {/* Replay Controls */}
      {activeReplayId && !showComparison && (
        <div className="replay-section">
          <div className="replay-header">
            <h4>Episode Replay</h4>
            <button className="stop-replay-btn" onClick={stopReplay}>
              Stop Replay
            </button>
          </div>
          <EpisodeReplayControls
            replayId={activeReplayId}
            replayStatus={replayStatus}
            onCommand={sendReplayCommand}
            onSpeedChange={(speed) => sendReplayCommand('speed', { speed })}
            onSeek={(frame) => sendReplayCommand('seek', { frame })}
          />
        </div>
      )}

      {/* Comparison View */}
      {showComparison && (
        <EpisodeComparison
          episodes={selectedEpisodes}
          comparisonReplayId={activeReplayId}
          replayStatus={replayStatus}
          onCommand={sendReplayCommand}
          onSpeedChange={(speed) => sendReplayCommand('speed', { speed })}
          onSeek={(frame) => sendReplayCommand('seek', { frame })}
          onClose={() => {
            setShowComparison(false);
            stopReplay();
          }}
        />
      )}
    </div>
  );
};

export default EpisodeBrowser;