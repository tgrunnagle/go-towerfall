import axios from 'axios';

export const api = axios.create({
  baseURL: window.APP_CONFIG?.BACKEND_API_URL || 'http://localhost:4000',
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json'
  }
});