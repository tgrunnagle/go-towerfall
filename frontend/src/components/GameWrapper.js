import React, { useRef, useEffect, useState } from 'react';

const GameWrapper = ({
  roomId,
  playerId,
  playerToken,
  canvasSizeX,
  canvasSizeY,
  setPlayerName,
  setRoomName,
  setRoomCode,
  setRoomPassword,
  onExitGame,
}) => {
  const canvasRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

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
        },
        onError: (errorMessage) => {
          setError(errorMessage);
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
    setError
  ]);

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
    </div>
  );
};

export default GameWrapper;
