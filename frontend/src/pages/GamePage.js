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
  
  // Redirect to home if missing required parameters
  useEffect(() => {
    if (!queryRoomId || !queryPlayerId || !queryPlayerToken) {
      navigate('/');
    }
  }, [queryRoomId, queryPlayerId, queryPlayerToken, navigate]);
  
  // Handle exit game
  const handleExitGame = useCallback(() => {
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
        setPlayerName={setPlayerName}
        setRoomName={setRoomName}
        setRoomCode={setRoomCode}
        setRoomPassword={setRoomPassword}
        onExitGame={handleExitGame}
      />
      
      <div className="controls">
        <div className="instructions">
          <p>Use <strong>W, A, S, D</strong> keys to move your player</p>
        </div>
      </div>
    </div>
  );
};

export default GamePage;
