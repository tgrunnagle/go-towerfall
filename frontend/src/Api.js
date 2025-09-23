import axios from 'axios';

export const api = axios.create({
  baseURL: window.APP_CONFIG?.BACKEND_API_URL || 'http://localhost:4000',
  timeout: 5000,
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

// Bot management API functions

export const getAvailableBots = async () => {
  const response = await api.get('/api/bots/available');
  return response.data;
};

export const getBotServerStatus = async () => {
  const response = await api.get('/api/bots/status');
  return response.data;
};

export const getRoomBots = async (roomId, playerToken) => {
  const response = await api.get(`/api/rooms/${roomId}/bots`, {
    headers: {
      'Authorization': `Bearer ${playerToken}`
    }
  });
  return response.data;
};

export const addBotToRoom = async (roomId, playerToken, botConfig) => {
  const response = await api.post(`/api/rooms/${roomId}/bots`, botConfig, {
    headers: {
      'Authorization': `Bearer ${playerToken}`
    }
  });
  return response.data;
};

export const removeBotFromRoom = async (roomId, botId, playerToken) => {
  const response = await api.delete(`/api/rooms/${roomId}/bots/${botId}`, {
    headers: {
      'Authorization': `Bearer ${playerToken}`
    }
  });
  return response.data;
};

export const configureBotDifficulty = async (roomId, botId, difficulty, playerToken) => {
  const response = await api.put(`/api/rooms/${roomId}/bots/${botId}/difficulty`, {
    difficulty
  }, {
    headers: {
      'Authorization': `Bearer ${playerToken}`
    }
  });
  return response.data;
};