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
  const queryCanvasSizeX = parseInt(queryParams.get('canvasSizeX'), 10);
  const queryCanvasSizeY = parseInt(queryParams.get('canvasSizeY'), 10);
  // Training mode parameters
  const queryTrainingMode = queryParams.get('trainingMode') === 'true';
  const queryTickMultiplier = parseFloat(queryParams.get('tickMultiplier')) || 1.0;
  const queryMaxGameDurationSec = parseInt(queryParams.get('maxGameDurationSec'), 10) || 0;
  const queryMaxKills = parseInt(queryParams.get('maxKills'), 10) || 0;
  
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
        canvasSizeX={queryCanvasSizeX}
        canvasSizeY={queryCanvasSizeY}
        trainingMode={queryTrainingMode}
        tickMultiplier={queryTickMultiplier}
        maxGameDurationSec={queryMaxGameDurationSec}
        maxKills={queryMaxKills}
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
            {queryTrainingMode && (
              <div className="training-mode-info">
                <p><strong>Training Mode</strong> - {queryTickMultiplier}x Speed</p>
                {queryMaxGameDurationSec > 0 && <p>Max Duration: {queryMaxGameDurationSec}s</p>}
                {queryMaxKills > 0 && <p>Max Kills: {queryMaxKills}</p>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default GamePage;
