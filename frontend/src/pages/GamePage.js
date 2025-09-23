import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import GameWrapper from '@/components/GameWrapper';

const GamePage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [playerName, setPlayerName] = useState('');
  const [roomName, setRoomName] = useState('');
  const [roomCode, setRoomCode] = useState('');
  const [roomPassword, setRoomPassword] = useState('');
  
  // Extract game parameters from URL query parameters
  const queryParams = new URLSearchParams(location.search);
  const queryRoomId = queryParams.get('roomId');
  const queryPlayerId = queryParams.get('playerId');
  const queryPlayerToken = queryParams.get('playerToken');
  const queryIsSpectator = queryParams.get('isSpectator') === 'true';
  const queryIsTrainingRoom = queryParams.get('isTrainingRoom') === 'true';
  const queryCanvasSizeX = parseInt(queryParams.get('canvasSizeX'), 10);
  const queryCanvasSizeY = parseInt(queryParams.get('canvasSizeY'), 10);
  
  // Redirect to home if missing required parameters
  useEffect(() => {
    if (!queryRoomId || !queryPlayerId || !queryPlayerToken || !queryCanvasSizeX || !queryCanvasSizeY) {
      navigate('/');
    }
  }, [queryRoomId, queryPlayerId, queryPlayerToken, queryCanvasSizeX, queryCanvasSizeY, queryIsSpectator, navigate]);
  
  // Handle exit game
  const handleExitGame = useCallback(() => {
    setRoomName && setRoomName(null);
    setRoomCode && setRoomCode(null);
    setRoomPassword && setRoomPassword(null);
    setPlayerName && setPlayerName(null);
    navigate('/');
  }, [navigate]);

  
  if (!queryRoomId || !queryPlayerId || !queryPlayerToken) {
    return null; // Don't render anything if redirecting
  }
  
  return (
    <div className="game-container container">
      <div className="game-info">
        <div>
          <h2>Game Room: {roomName ? roomName : 'Loading...'}</h2>
          {playerName && (
            <p>Player Name: <strong>{playerName}</strong></p>
          )}
          {roomCode && (
            <p>Room Code: <strong>{roomCode}</strong></p>
          )}
          {roomPassword && (
            <p>Password: <strong>{roomPassword}</strong></p>
          )}
        </div>
      </div>
      
      <GameWrapper
        roomId={queryRoomId}
        playerId={queryPlayerId}
        playerToken={queryPlayerToken}
        isSpectator={queryIsSpectator}
        isTrainingRoom={queryIsTrainingRoom}
        canvasSizeX={queryCanvasSizeX}
        canvasSizeY={queryCanvasSizeY}
        setPlayerName={setPlayerName}
        setRoomName={setRoomName}
        setRoomCode={setRoomCode}
        setRoomPassword={setRoomPassword}
        onExitGame={handleExitGame}
      />
      
      <div className="controls">
        {!queryIsSpectator && (
          <div className="instructions">
            <p>Use <strong>W, A, S, D</strong> keys to move your player</p>
            <p>Click, hold, and release left mouse button to shoot arrows, right mouse button to cancel</p>
            <p>Don't forget to pick up grounded arrows if you run out</p>
          </div>
        )}
        {queryIsSpectator && (
          <div className="spectator-instructions">
            <p>You are spectating</p>
            {queryIsTrainingRoom && (
              <p><strong>Training Room:</strong> Press <kbd>H</kbd> to toggle training metrics overlay</p>
            )}
          </div>
        )}
        {queryIsTrainingRoom && !queryIsSpectator && (
          <div className="training-instructions">
            <p><strong>Training Mode:</strong> Press <kbd>H</kbd> to show/hide training metrics</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default GamePage;
