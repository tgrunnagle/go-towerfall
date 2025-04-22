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

export const joinGame = async ({ playerName, roomCode, roomPassword }) => {
  const response = await api.post('/api/joinGame', {
    playerName,
    roomCode,
    roomPassword
  });
  return response.data;
};