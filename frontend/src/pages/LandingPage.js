import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();
  
  const [activeOption, setActiveOption] = useState(null);
  const [roomName, setRoomName] = useState('');
  const [roomCode, setRoomCode] = useState('');
  const [playerName, setPlayerName] = useState('');
  const [roomPassword, setRoomPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleCreateGame = useCallback(async (e) => {
    e.preventDefault();
    if (!roomName || !playerName) {
      setError('Please fill in all fields');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:4000/api/createGame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          playerName,
          roomName
        })
      });
      
      const data = await response.json();
      
      if (!data.success) {
        setError(data.error || 'Failed to create game');
        setIsLoading(false);
        return;
      }
      
      // Navigate to game page with game info as query parameters
      navigate(`/game?roomId=${data.roomId}&playerId=${data.playerId}&playerToken=${data.playerToken}&roomCode=${data.roomCode}`);
    } catch (error) {
      console.error('Error creating game:', error);
      setError('Failed to connect to server');
      setIsLoading(false);
    }
  }, [navigate, playerName, roomName]);

  const handleJoinGame = useCallback(async (e) => {
    e.preventDefault();
    if (!roomCode || !playerName || !roomPassword) {
      setError('Please fill in all fields');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:4000/api/joinGame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          playerName,
          roomCode,
          roomPassword
        })
      });
      
      const data = await response.json();
      
      if (!data.success) {
        setError(data.error || 'Failed to join game');
        setIsLoading(false);
        return;
      }
      
      // Navigate to game page with game info as query parameters
      navigate(`/game?roomId=${data.roomId}&playerId=${data.playerId}&playerToken=${data.playerToken}&roomCode=${data.roomCode}`);
    } catch (error) {
      console.error('Error joining game:', error);
      setError('Failed to connect to server');
      setIsLoading(false);
    }
  }, [navigate, playerName, roomCode, roomPassword]);

  return (
    <div className="landing-page">
      <div className="landing-header">
        <h1>Game Room</h1>
        <p>Create a new game room or join an existing one</p>
      </div>

      <div className="landing-content">
        <div className="options-container">
          <div className="option-card">
            <h2>Create Game</h2>
            <p>Start a new game room that others can join</p>
            {activeOption !== 'create' && (
              <button 
                className="btn" 
                onClick={() => setActiveOption('create')}
              >
                Create New Game
              </button>
            )}
            
            {activeOption === 'create' && (
              <form className="form" onSubmit={handleCreateGame}>
                <div className="form-group">
                  <label htmlFor="create-room-name">Room Name</label>
                  <input
                    id="create-room-name"
                    type="text"
                    value={roomName}
                    onChange={(e) => setRoomName(e.target.value)}
                    placeholder="Enter room name"
                    required
                    disabled={isLoading}
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="create-player-name">Your Name</label>
                  <input
                    id="create-player-name"
                    type="text"
                    value={playerName}
                    onChange={(e) => setPlayerName(e.target.value)}
                    placeholder="Enter your name"
                    required
                    disabled={isLoading}
                  />
                </div>
                
                {error && <div className="error">{error}</div>}
                
                <button type="submit" className="btn" disabled={isLoading}>
                  {isLoading ? 'Creating...' : 'Start Game'}
                </button>
              </form>
            )}
          </div>
          
          <div className="option-card">
            <h2>Join Game</h2>
            <p>Join an existing game room with a password</p>
            {activeOption !== 'join' && (
              <button 
                className="btn" 
                onClick={() => setActiveOption('join')}
              >
                Join Existing Game
              </button>
            )}
            
            {activeOption === 'join' && (
              <form className="form" onSubmit={handleJoinGame}>
                <div className="form-group">
                  <label htmlFor="join-room-code">Room Code</label>
                  <input
                    id="join-room-code"
                    type="text"
                    value={roomCode}
                    onChange={(e) => setRoomCode(e.target.value)}
                    placeholder="Enter room code"
                    required
                    disabled={isLoading}
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="room-password">Room Password</label>
                  <input
                    id="room-password"
                    type="text"
                    value={roomPassword}
                    onChange={(e) => setRoomPassword(e.target.value)}
                    placeholder="Enter room password"
                    required
                    disabled={isLoading}
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="join-player-name">Your Name</label>
                  <input
                    id="join-player-name"
                    type="text"
                    value={playerName}
                    onChange={(e) => setPlayerName(e.target.value)}
                    placeholder="Enter your name"
                    required
                    disabled={isLoading}
                  />
                </div>
                
                {error && <div className="error">{error}</div>}
                
                <button type="submit" className="btn" disabled={isLoading}>
                  {isLoading ? 'Joining...' : 'Join Game'}
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
