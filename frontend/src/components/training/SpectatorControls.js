import React, { useState } from 'react';
import EpisodeBrowser from './EpisodeBrowser';

const SpectatorControls = ({ 
  settings, 
  onSettingsChange, 
  onExportData, 
  roomId,
  websocketConnection,
  sessionId
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showEpisodeBrowser, setShowEpisodeBrowser] = useState(false);

  const handleToggleSetting = (settingName) => {
    onSettingsChange({
      [settingName]: !settings[settingName]
    });
  };

  const handleTimeWindowChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0) {
      onSettingsChange({
        graphTimeWindow: value
      });
    }
  };

  const handleUpdateFrequencyChange = (event) => {
    const value = parseInt(event.target.value, 10);
    if (!isNaN(value) && value > 0) {
      onSettingsChange({
        updateFrequency: value
      });
    }
  };

  const handleResetView = () => {
    onSettingsChange({
      showMetrics: true,
      showGraphs: true,
      showDecisions: true,
      graphTimeWindow: 100,
      updateFrequency: 1000
    });
  };

  return (
    <div className="spectator-controls">
      <div className="controls-header">
        <button 
          className="expand-button"
          onClick={() => setIsExpanded(!isExpanded)}
          title={isExpanded ? "Collapse Controls" : "Expand Controls"}
        >
          {isExpanded ? '▼' : '▶'} Controls
        </button>
      </div>

      {isExpanded && (
        <div className="controls-content">
          <div className="control-section">
            <h5>Display Options</h5>
            <div className="toggle-controls">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={settings.showMetrics}
                  onChange={() => handleToggleSetting('showMetrics')}
                />
                <span className="toggle-text">Show Metrics</span>
              </label>
              
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={settings.showGraphs}
                  onChange={() => handleToggleSetting('showGraphs')}
                />
                <span className="toggle-text">Show Performance Graphs</span>
              </label>
              
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={settings.showDecisions}
                  onChange={() => handleToggleSetting('showDecisions')}
                />
                <span className="toggle-text">Show Bot Decisions</span>
              </label>
            </div>
          </div>

          <div className="control-section">
            <h5>Graph Settings</h5>
            <div className="input-controls">
              <label className="input-label">
                <span>Time Window (episodes):</span>
                <input
                  type="number"
                  min="10"
                  max="1000"
                  value={settings.graphTimeWindow}
                  onChange={handleTimeWindowChange}
                  className="number-input"
                />
              </label>
              
              <label className="input-label">
                <span>Update Frequency (ms):</span>
                <input
                  type="number"
                  min="100"
                  max="10000"
                  step="100"
                  value={settings.updateFrequency}
                  onChange={handleUpdateFrequencyChange}
                  className="number-input"
                />
              </label>
            </div>
          </div>

          <div className="control-section">
            <h5>Actions</h5>
            <div className="action-controls">
              <button 
                className="control-button export-button"
                onClick={onExportData}
                title="Export training data as JSON"
              >
                📊 Export Data
              </button>
              
              <button 
                className="control-button replay-button"
                onClick={() => setShowEpisodeBrowser(true)}
                title="Browse and replay episodes"
              >
                🎬 Episode Replay
              </button>
              
              <button 
                className="control-button reset-button"
                onClick={handleResetView}
                title="Reset all settings to default"
              >
                🔄 Reset View
              </button>
            </div>
          </div>

          <div className="control-section">
            <h5>Room Info</h5>
            <div className="room-info">
              <div className="info-row">
                <span className="info-label">Room ID:</span>
                <span className="info-value">{roomId}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Mode:</span>
                <span className="info-value">Training Spectator</span>
              </div>
            </div>
          </div>

          <div className="control-section">
            <h5>Keyboard Shortcuts</h5>
            <div className="shortcuts-info">
              <div className="shortcut-row">
                <span className="shortcut-key">M</span>
                <span className="shortcut-desc">Toggle Metrics</span>
              </div>
              <div className="shortcut-row">
                <span className="shortcut-key">G</span>
                <span className="shortcut-desc">Toggle Graphs</span>
              </div>
              <div className="shortcut-row">
                <span className="shortcut-key">D</span>
                <span className="shortcut-desc">Toggle Decisions</span>
              </div>
              <div className="shortcut-row">
                <span className="shortcut-key">H</span>
                <span className="shortcut-desc">Hide/Show Overlay</span>
              </div>
              <div className="shortcut-row">
                <span className="shortcut-key">R</span>
                <span className="shortcut-desc">Open Episode Replay</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Episode Browser Modal */}
      {showEpisodeBrowser && (
        <div className="episode-browser-modal">
          <div className="modal-backdrop" onClick={() => setShowEpisodeBrowser(false)} />
          <div className="modal-content">
            <EpisodeBrowser
              sessionId={sessionId}
              websocketConnection={websocketConnection}
              onClose={() => setShowEpisodeBrowser(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default SpectatorControls;