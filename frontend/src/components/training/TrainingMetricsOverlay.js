import React, { useState, useEffect, useRef } from 'react';
import PerformanceGraphs from './PerformanceGraphs';
import BotDecisionVisualization from './BotDecisionVisualization';
import SpectatorControls from './SpectatorControls';
import './TrainingMetricsOverlay.css';

const TrainingMetricsOverlay = ({ 
  roomId, 
  isVisible = true, 
  onToggleVisibility,
  websocketConnection 
}) => {
  const [trainingMetrics, setTrainingMetrics] = useState({
    currentEpisode: 0,
    totalEpisodes: 0,
    currentReward: 0,
    averageReward: 0,
    winRate: 0,
    modelGeneration: 1,
    algorithm: 'DQN',
    trainingTime: 0,
    episodesPerSecond: 0
  });

  const [performanceData, setPerformanceData] = useState({
    rewards: [],
    winRates: [],
    episodes: [],
    timestamps: []
  });

  const [botDecisionData, setBotDecisionData] = useState({
    actionProbabilities: {},
    qValues: {},
    selectedAction: null,
    stateValue: 0,
    confidence: 0
  });

  const [overlaySettings, setOverlaySettings] = useState({
    showMetrics: true,
    showGraphs: true,
    showDecisions: true,
    graphTimeWindow: 100, // Show last 100 episodes
    updateFrequency: 1000 // Update every 1 second
  });

  const wsRef = useRef(null);

  // WebSocket message handling
  useEffect(() => {
    if (!websocketConnection) return;

    wsRef.current = websocketConnection;

    const handleTrainingMetrics = (data) => {
      setTrainingMetrics(prevMetrics => ({
        ...prevMetrics,
        ...data,
        trainingTime: data.trainingTime || prevMetrics.trainingTime + 1
      }));
    };

    const handleBotDecision = (data) => {
      setBotDecisionData({
        actionProbabilities: data.actionProbabilities || {},
        qValues: data.qValues || {},
        selectedAction: data.selectedAction,
        stateValue: data.stateValue || 0,
        confidence: data.confidence || 0
      });
    };

    const handleGraphUpdate = (data) => {
      setPerformanceData(prevData => {
        const maxDataPoints = overlaySettings.graphTimeWindow;
        
        const newRewards = [...prevData.rewards, data.reward].slice(-maxDataPoints);
        const newWinRates = [...prevData.winRates, data.winRate].slice(-maxDataPoints);
        const newEpisodes = [...prevData.episodes, data.episode].slice(-maxDataPoints);
        const newTimestamps = [...prevData.timestamps, Date.now()].slice(-maxDataPoints);

        return {
          rewards: newRewards,
          winRates: newWinRates,
          episodes: newEpisodes,
          timestamps: newTimestamps
        };
      });
    };

    const handleMessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
          case 'training_metrics':
            handleTrainingMetrics(message.data);
            break;
          case 'bot_decision':
            handleBotDecision(message.data);
            break;
          case 'graph_update':
            handleGraphUpdate(message.data);
            break;
          default:
            // Ignore other message types
            break;
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocketConnection.addEventListener('message', handleMessage);

    return () => {
      if (websocketConnection) {
        websocketConnection.removeEventListener('message', handleMessage);
      }
    };
  }, [websocketConnection, overlaySettings.graphTimeWindow]);

  const handleSettingsChange = (newSettings) => {
    setOverlaySettings(prevSettings => ({
      ...prevSettings,
      ...newSettings
    }));
  };

  const handleExportData = () => {
    const exportData = {
      trainingMetrics,
      performanceData,
      timestamp: new Date().toISOString(),
      roomId
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `training-metrics-${roomId}-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!isVisible) {
    return (
      <div className="training-overlay-toggle">
        <button 
          className="toggle-button"
          onClick={onToggleVisibility}
          title="Show Training Metrics"
        >
          📊
        </button>
      </div>
    );
  }

  return (
    <div className="training-metrics-overlay">
      <div className="overlay-header">
        <h3>Training Metrics - Room {roomId}</h3>
        <button 
          className="close-button"
          onClick={onToggleVisibility}
          title="Hide Training Metrics"
        >
          ✕
        </button>
      </div>

      <div className="overlay-content">
        {overlaySettings.showMetrics && (
          <div className="metrics-section">
            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Episode:</span>
                <span className="metric-value">
                  {trainingMetrics.currentEpisode} / {trainingMetrics.totalEpisodes}
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Current Reward:</span>
                <span className="metric-value">{trainingMetrics.currentReward.toFixed(2)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Avg Reward:</span>
                <span className="metric-value">{trainingMetrics.averageReward.toFixed(2)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Win Rate:</span>
                <span className="metric-value">{(trainingMetrics.winRate * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Generation:</span>
                <span className="metric-value">{trainingMetrics.modelGeneration}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Algorithm:</span>
                <span className="metric-value">{trainingMetrics.algorithm}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Episodes/sec:</span>
                <span className="metric-value">{trainingMetrics.episodesPerSecond.toFixed(1)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Training Time:</span>
                <span className="metric-value">{Math.floor(trainingMetrics.trainingTime / 60)}m {trainingMetrics.trainingTime % 60}s</span>
              </div>
            </div>
          </div>
        )}

        {overlaySettings.showGraphs && (
          <PerformanceGraphs 
            data={performanceData}
            timeWindow={overlaySettings.graphTimeWindow}
          />
        )}

        {overlaySettings.showDecisions && (
          <BotDecisionVisualization 
            data={botDecisionData}
          />
        )}

        <SpectatorControls
          settings={overlaySettings}
          onSettingsChange={handleSettingsChange}
          onExportData={handleExportData}
          roomId={roomId}
        />
      </div>
    </div>
  );
};

export default TrainingMetricsOverlay;