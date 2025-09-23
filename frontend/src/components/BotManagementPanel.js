import React, { useState, useEffect, useCallback } from 'react';
import { 
  getAvailableBots, 
  getRoomBots, 
  addBotToRoom, 
  removeBotFromRoom, 
  configureBotDifficulty 
} from '../Api';
import './BotManagementPanel.css';

const BotManagementPanel = ({ roomId, playerToken, isVisible, onClose }) => {
  const [availableBots, setAvailableBots] = useState([]);
  const [roomBots, setRoomBots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedBotType, setSelectedBotType] = useState('');
  const [selectedDifficulty, setSelectedDifficulty] = useState('intermediate');
  const [botName, setBotName] = useState('');
  const [selectedGeneration, setSelectedGeneration] = useState(null);

  // Load available bot types and current room bots
  const loadData = useCallback(async () => {
    if (!isVisible) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const [availableResponse, roomBotsResponse] = await Promise.all([
        getAvailableBots(),
        getRoomBots(roomId, playerToken)
      ]);
      
      if (availableResponse.success) {
        setAvailableBots(availableResponse.botTypes || []);
        if (availableResponse.botTypes.length > 0) {
          setSelectedBotType(availableResponse.botTypes[0].type);
        }
      } else {
        setError(availableResponse.error || 'Failed to load available bots');
      }
      
      if (roomBotsResponse.success) {
        setRoomBots(roomBotsResponse.bots || []);
      } else {
        setError(roomBotsResponse.error || 'Failed to load room bots');
      }
    } catch (err) {
      setError('Failed to load bot data: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, [roomId, playerToken, isVisible]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle adding a bot to the room
  const handleAddBot = async () => {
    if (!selectedBotType || !botName.trim()) {
      setError('Please select a bot type and enter a name');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const botConfig = {
        botType: selectedBotType,
        difficulty: selectedDifficulty,
        botName: botName.trim(),
        generation: selectedGeneration,
        trainingMode: false
      };

      const response = await addBotToRoom(roomId, playerToken, botConfig);
      
      if (response.success) {
        setBotName(''); // Clear the name field
        await loadData(); // Refresh the bot list
      } else {
        setError(response.error || 'Failed to add bot');
      }
    } catch (err) {
      setError('Failed to add bot: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle removing a bot from the room
  const handleRemoveBot = async (botId) => {
    setLoading(true);
    setError(null);

    try {
      const response = await removeBotFromRoom(roomId, botId, playerToken);
      
      if (response.success) {
        await loadData(); // Refresh the bot list
      } else {
        setError(response.error || 'Failed to remove bot');
      }
    } catch (err) {
      setError('Failed to remove bot: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle changing bot difficulty
  const handleChangeDifficulty = async (botId, newDifficulty) => {
    setLoading(true);
    setError(null);

    try {
      const response = await configureBotDifficulty(roomId, botId, newDifficulty, playerToken);
      
      if (response.success) {
        await loadData(); // Refresh the bot list
      } else {
        setError(response.error || 'Failed to change bot difficulty');
      }
    } catch (err) {
      setError('Failed to change bot difficulty: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Get the selected bot type details
  const selectedBotTypeDetails = availableBots.find(bot => bot.type === selectedBotType);

  if (!isVisible) return null;

  return (
    <div className="bot-management-overlay">
      <div className="bot-management-panel">
        <div className="panel-header">
          <h3>Bot Management</h3>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="panel-content">
          {/* Add Bot Section */}
          <div className="add-bot-section">
            <h4>Add Bot</h4>
            
            <div className="form-group">
              <label>Bot Type:</label>
              <select 
                value={selectedBotType} 
                onChange={(e) => setSelectedBotType(e.target.value)}
                disabled={loading}
              >
                {availableBots.map(bot => (
                  <option key={bot.type} value={bot.type}>
                    {bot.name}
                  </option>
                ))}
              </select>
              {selectedBotTypeDetails && (
                <div className="bot-description">
                  {selectedBotTypeDetails.description}
                </div>
              )}
            </div>

            <div className="form-group">
              <label>Difficulty:</label>
              <select 
                value={selectedDifficulty} 
                onChange={(e) => setSelectedDifficulty(e.target.value)}
                disabled={loading}
              >
                {selectedBotTypeDetails?.difficulties?.map(difficulty => (
                  <option key={difficulty} value={difficulty}>
                    {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {selectedBotTypeDetails?.availableGenerations && (
              <div className="form-group">
                <label>Generation:</label>
                <select 
                  value={selectedGeneration || ''} 
                  onChange={(e) => setSelectedGeneration(e.target.value ? parseInt(e.target.value) : null)}
                  disabled={loading}
                >
                  <option value="">Latest</option>
                  {selectedBotTypeDetails.availableGenerations.map(gen => (
                    <option key={gen} value={gen}>
                      Generation {gen}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="form-group">
              <label>Bot Name:</label>
              <input 
                type="text" 
                value={botName} 
                onChange={(e) => setBotName(e.target.value)}
                placeholder="Enter bot name"
                disabled={loading}
                maxLength={20}
              />
            </div>

            <button 
              className="add-bot-button" 
              onClick={handleAddBot}
              disabled={loading || !selectedBotType || !botName.trim()}
            >
              {loading ? 'Adding...' : 'Add Bot'}
            </button>
          </div>

          {/* Current Bots Section */}
          <div className="current-bots-section">
            <h4>Current Bots ({roomBots.length})</h4>
            
            {roomBots.length === 0 ? (
              <div className="no-bots-message">
                No bots in this room
              </div>
            ) : (
              <div className="bots-list">
                {roomBots.map(bot => (
                  <div key={bot.botId} className="bot-item">
                    <div className="bot-info">
                      <div className="bot-name">{bot.name}</div>
                      <div className="bot-details">
                        {bot.botType.replace('_', ' ')} • {bot.difficulty}
                        {bot.generation && ` • Gen ${bot.generation}`}
                      </div>
                      <div className="bot-status">
                        Status: <span className={`status-${bot.status}`}>{bot.status}</span>
                      </div>
                      {bot.performance && (
                        <div className="bot-performance">
                          Games: {bot.performance.games_played || 0} • 
                          Wins: {bot.performance.wins || 0} • 
                          Losses: {bot.performance.losses || 0}
                        </div>
                      )}
                    </div>
                    
                    <div className="bot-controls">
                      <select 
                        value={bot.difficulty} 
                        onChange={(e) => handleChangeDifficulty(bot.botId, e.target.value)}
                        disabled={loading}
                        className="difficulty-select"
                      >
                        {selectedBotTypeDetails?.difficulties?.map(difficulty => (
                          <option key={difficulty} value={difficulty}>
                            {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
                          </option>
                        ))}
                      </select>
                      
                      <button 
                        className="remove-bot-button" 
                        onClick={() => handleRemoveBot(bot.botId)}
                        disabled={loading}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BotManagementPanel;