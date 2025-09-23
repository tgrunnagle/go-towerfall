import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const BotDecisionVisualization = ({ data }) => {
  const actionNames = useMemo(() => ({
    0: 'No Action',
    1: 'Move Left',
    2: 'Move Right', 
    3: 'Jump',
    4: 'Shoot',
    5: 'Move Left + Shoot',
    6: 'Move Right + Shoot',
    7: 'Jump + Shoot'
  }), []);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#333333',
        borderWidth: 1,
        callbacks: {
          label: function(context) {
            return `${context.label}: ${(context.parsed.y * 100).toFixed(1)}%`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        ticks: {
          color: '#ffffff',
          maxRotation: 45,
          minRotation: 0
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      y: {
        display: true,
        min: 0,
        max: 1,
        ticks: {
          color: '#ffffff',
          callback: function(value) {
            return (value * 100).toFixed(0) + '%';
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        title: {
          display: true,
          text: 'Probability',
          color: '#ffffff'
        }
      }
    },
    animation: {
      duration: 200
    }
  };

  const actionProbabilityData = useMemo(() => {
    const actions = Object.keys(data.actionProbabilities || {});
    const probabilities = Object.values(data.actionProbabilities || {});
    
    const labels = actions.map(action => actionNames[action] || `Action ${action}`);
    
    // Color the selected action differently
    const backgroundColors = actions.map(action => 
      action === data.selectedAction ? 'rgba(255, 99, 132, 0.8)' : 'rgba(75, 192, 192, 0.6)'
    );
    
    const borderColors = actions.map(action => 
      action === data.selectedAction ? 'rgba(255, 99, 132, 1)' : 'rgba(75, 192, 192, 1)'
    );

    return {
      labels,
      datasets: [
        {
          label: 'Action Probability',
          data: probabilities,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 2
        }
      ]
    };
  }, [data.actionProbabilities, data.selectedAction, actionNames]);

  const qValueData = useMemo(() => {
    const actions = Object.keys(data.qValues || {});
    const qValues = Object.values(data.qValues || {});
    
    const labels = actions.map(action => actionNames[action] || `Action ${action}`);
    
    // Color the selected action differently
    const backgroundColors = actions.map(action => 
      action === data.selectedAction ? 'rgba(255, 206, 86, 0.8)' : 'rgba(153, 102, 255, 0.6)'
    );
    
    const borderColors = actions.map(action => 
      action === data.selectedAction ? 'rgba(255, 206, 86, 1)' : 'rgba(153, 102, 255, 1)'
    );

    return {
      labels,
      datasets: [
        {
          label: 'Q-Value',
          data: qValues,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 2
        }
      ]
    };
  }, [data.qValues, data.selectedAction, actionNames]);

  const qValueOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        min: undefined,
        max: undefined,
        title: {
          display: true,
          text: 'Q-Value',
          color: '#ffffff'
        },
        ticks: {
          color: '#ffffff',
          callback: function(value) {
            return value.toFixed(2);
          }
        }
      }
    }
  };

  const hasActionData = Object.keys(data.actionProbabilities || {}).length > 0;
  const hasQValueData = Object.keys(data.qValues || {}).length > 0;

  return (
    <div className="bot-decision-visualization">
      <h4>Bot Decision Analysis</h4>
      
      <div className="decision-info">
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Selected Action:</span>
            <span className="info-value selected-action">
              {data.selectedAction !== null ? 
                (actionNames[data.selectedAction] || `Action ${data.selectedAction}`) : 
                'None'
              }
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">State Value:</span>
            <span className="info-value">{data.stateValue.toFixed(3)}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Confidence:</span>
            <span className="info-value">{(data.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      <div className="charts-container">
        {hasActionData && (
          <div className="chart-section">
            <h5>Action Probabilities</h5>
            <div className="chart-wrapper">
              <Bar data={actionProbabilityData} options={chartOptions} />
            </div>
          </div>
        )}

        {hasQValueData && (
          <div className="chart-section">
            <h5>Q-Values</h5>
            <div className="chart-wrapper">
              <Bar data={qValueData} options={qValueOptions} />
            </div>
          </div>
        )}

        {!hasActionData && !hasQValueData && (
          <div className="no-data-message">
            <p>Waiting for bot decision data...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default BotDecisionVisualization;