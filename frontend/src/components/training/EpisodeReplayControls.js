import React, { useState, useEffect, useCallback } from 'react';
import './EpisodeReplayControls.css';

const EpisodeReplayControls = ({
  replayId,
  replayStatus,
  onCommand,
  onSpeedChange,
  onSeek,
  className = ''
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [progress, setProgress] = useState(0);

  // Update state from replay status
  useEffect(() => {
    if (replayStatus) {
      setIsPlaying(replayStatus.state === 'playing');
      setCurrentFrame(replayStatus.current_frame || 0);
      setTotalFrames(replayStatus.total_frames || 0);
      setPlaybackSpeed(replayStatus.playback_speed || 1.0);
      setProgress(replayStatus.progress || 0);
    }
  }, [replayStatus]);

  // Control handlers
  const handlePlay = useCallback(() => {
    onCommand('play');
  }, [onCommand]);

  const handlePause = useCallback(() => {
    onCommand('pause');
  }, [onCommand]);

  const handleStop = useCallback(() => {
    onCommand('stop');
  }, [onCommand]);

  const handleStepForward = useCallback(() => {
    onCommand('step', { direction: 1 });
  }, [onCommand]);

  const handleStepBackward = useCallback(() => {
    onCommand('step', { direction: -1 });
  }, [onCommand]);

  const handleSeekChange = useCallback((event) => {
    const newFrame = parseInt(event.target.value, 10);
    setCurrentFrame(newFrame);
    onSeek(newFrame);
  }, [onSeek]);

  const handleSpeedChange = useCallback((event) => {
    const newSpeed = parseFloat(event.target.value);
    setPlaybackSpeed(newSpeed);
    onSpeedChange(newSpeed);
  }, [onSpeedChange]);

  // Format time display
  const formatTime = (frame, fps = 60) => {
    const seconds = frame / fps;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Speed options
  const speedOptions = [
    { value: 0.25, label: '0.25x' },
    { value: 0.5, label: '0.5x' },
    { value: 1.0, label: '1x' },
    { value: 1.5, label: '1.5x' },
    { value: 2.0, label: '2x' },
    { value: 4.0, label: '4x' }
  ];

  return (
    <div className={`episode-replay-controls ${className}`}>
      {/* Main Controls */}
      <div className="replay-main-controls">
        <button
          className="replay-btn step-backward"
          onClick={handleStepBackward}
          disabled={currentFrame <= 0}
          title="Step Backward"
        >
          ⏮
        </button>

        <button
          className={`replay-btn play-pause ${isPlaying ? 'playing' : 'paused'}`}
          onClick={isPlaying ? handlePause : handlePlay}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        <button
          className="replay-btn step-forward"
          onClick={handleStepForward}
          disabled={currentFrame >= totalFrames - 1}
          title="Step Forward"
        >
          ⏭
        </button>

        <button
          className="replay-btn stop"
          onClick={handleStop}
          title="Stop"
        >
          ⏹
        </button>
      </div>

      {/* Progress Bar */}
      <div className="replay-progress-section">
        <div className="replay-time-display">
          <span className="current-time">
            {formatTime(currentFrame)}
          </span>
          <span className="separator">/</span>
          <span className="total-time">
            {formatTime(totalFrames)}
          </span>
        </div>

        <div className="replay-progress-container">
          <input
            type="range"
            className="replay-progress-bar"
            min="0"
            max={totalFrames - 1}
            value={currentFrame}
            onChange={handleSeekChange}
            disabled={totalFrames === 0}
          />
          <div 
            className="replay-progress-fill"
            style={{ width: `${progress * 100}%` }}
          />
        </div>

        <div className="replay-frame-info">
          <span>Frame: {currentFrame} / {totalFrames}</span>
        </div>
      </div>

      {/* Speed Control */}
      <div className="replay-speed-control">
        <label htmlFor="speed-select">Speed:</label>
        <select
          id="speed-select"
          value={playbackSpeed}
          onChange={handleSpeedChange}
          className="speed-select"
        >
          {speedOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {/* Episode Info */}
      {replayStatus?.episode_info && (
        <div className="replay-episode-info">
          <div className="episode-detail">
            <span className="label">Generation:</span>
            <span className="value">{replayStatus.episode_info.model_generation}</span>
          </div>
          <div className="episode-detail">
            <span className="label">Result:</span>
            <span className={`value result-${replayStatus.episode_info.game_result}`}>
              {replayStatus.episode_info.game_result}
            </span>
          </div>
          <div className="episode-detail">
            <span className="label">Reward:</span>
            <span className="value">{replayStatus.episode_info.total_reward?.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default EpisodeReplayControls;