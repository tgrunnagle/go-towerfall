import axios from 'axios';

// Main game server API (Go backend)
export const api = axios.create({
  baseURL: window.APP_CONFIG?.BACKEND_API_URL || 'http://localhost:4000',
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Python server API for bot management and training metrics
export const pythonApi = axios.create({
  baseURL: window.APP_CONFIG?.PYTHON_API_URL || 'http://localhost:4002',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const getMaps = async () => {
  const response = await api.get('/api/maps');
  return response.data.maps;
};

export const createGame = async ({ playerName, roomName, mapType }) => {
  const response = await api.post('/api/createGame', {
    playerName,
    roomName,
    mapType
  });
  return response.data;
};

export const joinGame = async ({ playerName, roomCode, roomPassword, isSpectator }) => {
  const response = await api.post('/api/joinGame', {
    playerName,
    roomCode,
    roomPassword,
    isSpectator: isSpectator || false
  });
  return response.data;
};

// Bot management API functions (using Python server)

export const getAvailableBots = async () => {
  try {
    const response = await pythonApi.get('/api/bots/available');
    return response.data;
  } catch (error) {
    console.error('Error getting available bots:', error);
    throw error;
  }
};

export const getBotServerStatus = async () => {
  try {
    const response = await pythonApi.get('/api/bots/status');
    return response.data;
  } catch (error) {
    console.error('Error getting bot server status:', error);
    throw error;
  }
};

export const getRoomBots = async (roomId, playerToken) => {
  try {
    const response = await pythonApi.get(`/api/bots/rooms/${roomId}/bots`);
    return response.data;
  } catch (error) {
    console.error('Error getting room bots:', error);
    throw error;
  }
};

export const addBotToRoom = async (roomId, playerToken, botConfig) => {
  try {
    // Get room details from the game server to get room code
    const roomResponse = await api.get(`/api/rooms/${roomId}/details`, {
      headers: {
        'Authorization': `Bearer ${playerToken}`
      }
    });
    
    if (!roomResponse.data.success) {
      throw new Error('Failed to get room details');
    }
    
    const room = roomResponse.data;
    
    // Spawn bot via Python server
    const spawnRequest = {
      bot_type: botConfig.botType,
      difficulty: botConfig.difficulty,
      bot_name: botConfig.botName,
      room_code: room.roomCode,
      room_password: room.roomPassword || '',
      generation: botConfig.generation,
      training_mode: botConfig.trainingMode || false
    };
    
    const response = await pythonApi.post('/api/bots/spawn', spawnRequest);
    return response.data;
  } catch (error) {
    console.error('Error adding bot to room:', error);
    throw error;
  }
};

export const removeBotFromRoom = async (roomId, botId, playerToken) => {
  try {
    const response = await pythonApi.post(`/api/bots/${botId}/terminate`);
    return response.data;
  } catch (error) {
    console.error('Error removing bot from room:', error);
    throw error;
  }
};

export const configureBotDifficulty = async (roomId, botId, difficulty, playerToken) => {
  try {
    const response = await pythonApi.put(`/api/bots/${botId}/configure`, {
      difficulty
    });
    return response.data;
  } catch (error) {
    console.error('Error configuring bot difficulty:', error);
    throw error;
  }
};