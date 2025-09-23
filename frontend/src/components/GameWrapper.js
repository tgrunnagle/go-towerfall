import React, { useRef, useEffect, useState } from 'react';
import TrainingMetricsOverlay from './training/TrainingMetricsOverlay';

const GameWrapper = ({
  roomId,
  playerId,
  playerToken,
  canvasSizeX,
  canvasSizeY,
  isSpectator,
  isTrainingRoom,
  setPlayerName,
  setRoomName,
  setRoomCode,
  setRoomPassword,
  onExitGame,
}) => {
  const canvasRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [showTrainingOverlay, setShowTrainingOverlay] = useState(false);
  const [websocketConnection, setWebsocketConnection] = useState(null);

  // Initialize the game
  useEffect(() => {
    if (!canvasRef.current) return;

    if (!roomId || !playerId || !playerToken || !canvasSizeX || !canvasSizeY) {
      setError('Missing required parameters');
      return;
    }

    if (!window.gameInstance) {
      setError('Game instance not found');
      return;
    }

    window.gameInstance.initGame(
      canvasRef.current,
      roomId,
      playerId,
      playerToken,
      canvasSizeX,
      canvasSizeY,
      {
        onConnectionChange: (isConnected) => {
          setConnected(isConnected);
        },
        onGameInfoChange: (info) => {
          setRoomName && setRoomName(info.roomName);
          setRoomCode && setRoomCode(info.roomCode);
          setRoomPassword && setRoomPassword(info.roomPassword);
          setPlayerName && setPlayerName(info.playerName);
          
          // Check if this is a training room and enable overlay
          if (info.isTrainingRoom || isTrainingRoom) {
            setShowTrainingOverlay(true);
          }
        },
        onError: (errorMessage) => {
          setError(errorMessage);
        },
        onWebSocketReady: (ws) => {
          setWebsocketConnection(ws);
        }
      }
    );

    // Clean up on unmount
    return () => { };
  }, [
    canvasSizeX,
    canvasSizeY,
    roomId,
    playerId,
    playerToken,
    canvasRef,
    setConnected,
    setPlayerName,
    setRoomName,
    setRoomCode,
    setRoomPassword,
    setError,
    isTrainingRoom
  ]);

  // Keyboard shortcuts for training overlay
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Only handle shortcuts if we're in a training room or spectating
      if (!isSpectator && !isTrainingRoom && !showTrainingOverlay) return;
      
      switch (event.key.toLowerCase()) {
        case 'h':
          setShowTrainingOverlay(prev => !prev);
          break;
        case 'm':
          // Toggle metrics - this would be handled by the overlay component
          break;
        case 'g':
          // Toggle graphs - this would be handled by the overlay component
          break;
        case 'd':
          // Toggle decisions - this would be handled by the overlay component
          break;
        default:
          break;
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [isSpectator, isTrainingRoom, showTrainingOverlay]);

  // Handle exit game
  const handleExitGame = () => {
    if (window.gameInstance) {
      window.gameInstance.exitGame();
    }

    setRoomName && setRoomName(null);
    setRoomCode && setRoomCode(null);
    setRoomPassword && setRoomPassword(null);
    setPlayerName && setPlayerName(null);
    if (onExitGame) {
      onExitGame();
    }
  };

  return (
    <div className="game-wrapper">
      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {!connected && (
        <div className="connection-status">
          Connecting to server...
        </div>
      )}

      <canvas
        ref={canvasRef}
        className="game-canvas"
        width={canvasSizeX}
        height={canvasSizeY}
      />

      <div className="game-controls">
        <button onClick={handleExitGame} className="exit-button">
          Exit Game
        </button>
      </div>

      {/* Training Metrics Overlay for spectators and training rooms */}
      {(isSpectator || isTrainingRoom || showTrainingOverlay) && (
        <TrainingMetricsOverlay
          roomId={roomId}
          isVisible={showTrainingOverlay}
          onToggleVisibility={() => setShowTrainingOverlay(prev => !prev)}
          websocketConnection={websocketConnection}
        />
      )}
    </div>
  );
};

export default GameWrapper;
