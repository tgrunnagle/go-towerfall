# Frontend Spectator UI Implementation Guide

## Overview
This document provides detailed implementation guidance for task 7.1.1 - creating the frontend training metrics overlay UI for the spectator system.

The backend spectator system is complete and functional, but users need frontend components to visualize the training metrics. This guide provides the technical details for implementing the required React components.

## 1. Core Components to Create

### TrainingMetricsOverlay.js
```javascript
// Main overlay component that displays training metrics
// - Episode progress, rewards, win rate
// - Algorithm info, model generation
// - Real-time updates via WebSocket
```

### PerformanceGraphs.js  
```javascript
// Real-time charts using Chart.js or similar
// - Reward progress over time
// - Win rate trends
// - Episode length tracking
// - Learning metrics (loss, epsilon)
```

### BotDecisionVisualization.js
```javascript
// Displays bot decision-making process
// - Action probability bars/pie charts
// - State values
// - Q-values visualization
// - Highlighted selected action
```

### SpectatorControls.js
```javascript
// Spectator-specific controls
// - Toggle metrics overlay on/off
// - Switch between different graph views
// - Export training data
// - Room management (if creator)
```

## 2. WebSocket Integration

### Enhanced GameWrapper.js
```javascript
// Add WebSocket message handlers for:
useEffect(() => {
  if (isSpectator && websocket) {
    websocket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'training_metrics':
          setTrainingMetrics(message.data);
          break;
        case 'bot_decision':
          setBotDecision(message);
          break;
        case 'graph_update':
          updateGraphs(message.graphs);
          break;
        case 'graph_config':
          setGraphConfig(message.graphs);
          break;
      }
    });
  }
}, [isSpectator, websocket]);
```

## 3. UI Layout Changes

### Enhanced GamePage.js
```javascript
// Add spectator-specific UI elements
{queryIsSpectator && (
  <div className="spectator-interface">
    <SpectatorControls 
      onToggleOverlay={setShowOverlay}
      onToggleGraphs={setShowGraphs}
    />
    
    {showOverlay && (
      <TrainingMetricsOverlay 
        metrics={trainingMetrics}
        className="metrics-overlay"
      />
    )}
    
    {showGraphs && (
      <PerformanceGraphs 
        graphData={graphData}
        graphConfig={graphConfig}
        className="performance-graphs"
      />
    )}
    
    <BotDecisionVisualization 
      decision={botDecision}
      className="decision-viz"
    />
  </div>
)}
```

## 4. CSS Styling

### spectator.css
```css
/* Overlay positioning and styling */
.metrics-overlay {
  position: absolute;
  top: 20px;
  right: 20px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 15px;
  border-radius: 8px;
  min-width: 300px;
}

.performance-graphs {
  position: absolute;
  bottom: 20px;
  left: 20px;
  background: rgba(255, 255, 255, 0.95);
  padding: 15px;
  border-radius: 8px;
  width: 400px;
  height: 300px;
}

.decision-viz {
  position: absolute;
  top: 20px;
  left: 20px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 15px;
  border-radius: 8px;
}
```

## 5. Dependencies to Add

### package.json additions
```json
{
  "dependencies": {
    "chart.js": "^4.0.0",
    "react-chartjs-2": "^5.0.0",
    "recharts": "^2.8.0"
  }
}
```

## 6. API Extensions

### Enhanced Api.js
```javascript
// Add spectator room management endpoints
export const createSpectatorRoom = async (trainingSessionId, options) => {
  const response = await api.post('/api/spectator/createRoom', {
    trainingSessionId,
    ...options
  });
  return response.data;
};

export const joinSpectatorRoom = async (roomCode, spectatorName, password) => {
  const response = await api.post('/api/spectator/joinRoom', {
    roomCode,
    spectatorName,
    password
  });
  return response.data;
};
```

## 7. Integration Flow

1. **User joins as spectator** → Frontend detects `isSpectator=true`
2. **WebSocket connects** → Backend sends `graph_config` message
3. **Training starts** → Backend sends `training_metrics` updates
4. **Bot makes decisions** → Backend sends `bot_decision` data
5. **Graphs update** → Frontend renders real-time charts
6. **User controls** → Toggle overlays, export data, etc.

## 8. Example Message Handling

```javascript
// In GameWrapper.js or new SpectatorManager.js
const handleSpectatorMessage = (message) => {
  switch (message.type) {
    case 'training_metrics':
      setMetrics({
        episode: message.data.episode,
        reward: message.data.current_reward,
        winRate: message.data.win_rate,
        algorithm: message.data.algorithm,
        // ... other metrics
      });
      break;
      
    case 'bot_decision':
      setDecision({
        actionProbs: message.action_probabilities,
        selectedAction: message.selected_action,
        stateValue: message.state_values,
        qValues: message.q_values
      });
      break;
      
    case 'graph_update':
      message.graphs.forEach(graph => {
        updateGraph(graph.graph_id, graph.data_points);
      });
      break;
  }
};
```

## 9. Backend Message Types Reference

The backend spectator system sends these WebSocket message types:

### training_metrics
```json
{
  "type": "training_metrics",
  "data": {
    "episode": 150,
    "total_episodes": 1000,
    "current_reward": 85.5,
    "average_reward": 72.3,
    "best_reward": 120.0,
    "win_rate": 0.68,
    "algorithm": "DQN",
    "model_generation": 3,
    "training_time_elapsed": 1800.0
  }
}
```

### bot_decision
```json
{
  "type": "bot_decision",
  "action_probabilities": {
    "move_left": 0.25,
    "move_right": 0.35,
    "shoot": 0.4
  },
  "selected_action": "shoot",
  "state_values": 0.82,
  "q_values": [0.1, 0.8, 0.6]
}
```

### graph_update
```json
{
  "type": "graph_update",
  "graphs": [
    {
      "graph_id": "reward_progress",
      "timestamp": "2024-01-15T10:30:00Z",
      "data_points": {
        "current_reward": 85.5,
        "average_reward": 72.3,
        "best_reward": 120.0
      }
    }
  ]
}
```

### graph_config
```json
{
  "type": "graph_config",
  "graphs": [
    {
      "graph_id": "reward_progress",
      "title": "Reward Progress",
      "y_label": "Reward",
      "metrics": ["current_reward", "average_reward", "best_reward"],
      "max_points": 1000
    }
  ]
}
```

## Summary

The backend spectator system is complete and functional, but users need these frontend components to actually see and interact with the training metrics. The main work involves:

1. **Creating React components** for metrics display and visualization
2. **Adding WebSocket message handling** for real-time updates  
3. **Integrating charts/graphs** for performance visualization
4. **Enhancing the spectator UI** with controls and overlays
5. **Adding CSS styling** for proper positioning and appearance

This document provides the technical roadmap for implementing task 7.1.1.