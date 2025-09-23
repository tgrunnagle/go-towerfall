# Training Metrics Overlay Components

This directory contains React components for displaying real-time training metrics and bot decision analysis during RL bot training sessions.

## Components

### TrainingMetricsOverlay
Main overlay component that displays training metrics, performance graphs, and bot decision analysis.

**Props:**
- `roomId` (string): The training room ID
- `isVisible` (boolean): Whether the overlay is visible
- `onToggleVisibility` (function): Callback to toggle overlay visibility
- `websocketConnection` (WebSocket): WebSocket connection for receiving training data

### PerformanceGraphs
Displays real-time charts of training performance using Chart.js.

**Features:**
- Episode reward tracking with moving average
- Win rate visualization over time
- Configurable time window for data display

### BotDecisionVisualization
Shows bot decision-making process including action probabilities and Q-values.

**Features:**
- Action probability bar charts
- Q-value visualization
- Selected action highlighting
- State value and confidence display

### SpectatorControls
Provides controls for customizing the overlay display and exporting data.

**Features:**
- Toggle display sections on/off
- Adjust graph time window and update frequency
- Export training data as JSON
- Keyboard shortcuts reference

## Usage

### Basic Integration

```jsx
import TrainingMetricsOverlay from './components/training/TrainingMetricsOverlay';

function GameComponent({ roomId, websocketConnection }) {
  const [showOverlay, setShowOverlay] = useState(true);
  
  return (
    <div>
      {/* Your game canvas */}
      <canvas />
      
      {/* Training metrics overlay */}
      <TrainingMetricsOverlay
        roomId={roomId}
        isVisible={showOverlay}
        onToggleVisibility={() => setShowOverlay(!showOverlay)}
        websocketConnection={websocketConnection}
      />
    </div>
  );
}
```

### WebSocket Message Format

The overlay expects WebSocket messages in the following formats:

#### Training Metrics Message
```json
{
  "type": "training_metrics",
  "data": {
    "currentEpisode": 150,
    "totalEpisodes": 1000,
    "currentReward": 45.2,
    "averageReward": 38.7,
    "winRate": 0.65,
    "modelGeneration": 3,
    "algorithm": "PPO",
    "trainingTime": 3600,
    "episodesPerSecond": 2.5
  }
}
```

#### Bot Decision Message
```json
{
  "type": "bot_decision",
  "data": {
    "actionProbabilities": {
      "0": 0.1,
      "1": 0.3,
      "2": 0.4,
      "3": 0.2
    },
    "qValues": {
      "0": -0.5,
      "1": 1.2,
      "2": 2.1,
      "3": 0.8
    },
    "selectedAction": 2,
    "stateValue": 1.8,
    "confidence": 0.85
  }
}
```

#### Graph Update Message
```json
{
  "type": "graph_update",
  "data": {
    "episode": 150,
    "reward": 45.2,
    "winRate": 0.65
  }
}
```

## Keyboard Shortcuts

When the overlay is active, the following keyboard shortcuts are available:

- `H` - Toggle overlay visibility
- `M` - Toggle metrics section (handled by overlay)
- `G` - Toggle performance graphs (handled by overlay)
- `D` - Toggle bot decision visualization (handled by overlay)

## Styling

The overlay uses CSS custom properties and is designed to be responsive. It includes:

- Dark theme with transparency and blur effects
- Responsive grid layouts for different screen sizes
- Smooth animations and transitions
- Chart.js integration with custom styling

## Testing

Run the component tests with:

```bash
npm test -- --testPathPattern=TrainingMetricsOverlay.test.js
```

The tests cover:
- Component rendering in different states
- User interactions (toggle visibility, expand controls)
- Default data display
- WebSocket message handling (mocked)

## Dependencies

- React 18+
- Chart.js and react-chartjs-2 for performance graphs
- CSS Grid and Flexbox for responsive layouts