import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import TrainingMetricsOverlay from './TrainingMetricsOverlay';

// Mock Chart.js components
jest.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="mock-line-chart">Line Chart</div>,
  Bar: () => <div data-testid="mock-bar-chart">Bar Chart</div>
}));

describe('TrainingMetricsOverlay', () => {
  const mockProps = {
    roomId: 'test-room-123',
    isVisible: true,
    onToggleVisibility: jest.fn(),
    websocketConnection: null
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders overlay when visible', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    expect(screen.getByText('Training Metrics - Room test-room-123')).toBeInTheDocument();
    expect(screen.getByText('Episode:')).toBeInTheDocument();
    expect(screen.getByText('Current Reward:')).toBeInTheDocument();
  });

  test('renders toggle button when not visible', () => {
    render(<TrainingMetricsOverlay {...mockProps} isVisible={false} />);
    
    expect(screen.getByTitle('Show Training Metrics')).toBeInTheDocument();
    expect(screen.queryByText('Training Metrics - Room test-room-123')).not.toBeInTheDocument();
  });

  test('calls onToggleVisibility when close button is clicked', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    const closeButton = screen.getByTitle('Hide Training Metrics');
    fireEvent.click(closeButton);
    
    expect(mockProps.onToggleVisibility).toHaveBeenCalledTimes(1);
  });

  test('displays default training metrics', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    expect(screen.getByText('0 / 0')).toBeInTheDocument(); // Episode count
    expect(screen.getAllByText('0.00')).toHaveLength(2); // Current and average reward
    expect(screen.getAllByText('0.0%')).toHaveLength(2); // Win rate appears in metrics and decision sections
    expect(screen.getByText('1')).toBeInTheDocument(); // Generation
    expect(screen.getByText('DQN')).toBeInTheDocument(); // Algorithm
  });

  test('renders performance graphs', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    expect(screen.getByText('Episode Rewards')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getAllByTestId('mock-line-chart')).toHaveLength(2);
  });

  test('renders bot decision visualization', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    expect(screen.getByText('Bot Decision Analysis')).toBeInTheDocument();
    expect(screen.getByText('Selected Action:')).toBeInTheDocument();
    expect(screen.getByText('State Value:')).toBeInTheDocument();
    expect(screen.getByText('Confidence:')).toBeInTheDocument();
  });

  test('renders spectator controls', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    expect(screen.getByText('▶ Controls')).toBeInTheDocument();
  });

  test('expands controls when clicked', () => {
    render(<TrainingMetricsOverlay {...mockProps} />);
    
    const expandButton = screen.getByText('▶ Controls');
    fireEvent.click(expandButton);
    
    expect(screen.getByText('▼ Controls')).toBeInTheDocument();
    expect(screen.getByText('Display Options')).toBeInTheDocument();
    expect(screen.getByText('Show Metrics')).toBeInTheDocument();
  });
});