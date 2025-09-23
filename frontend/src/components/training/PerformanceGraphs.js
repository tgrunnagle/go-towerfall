import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PerformanceGraphs = ({ data, timeWindow = 100 }) => {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
          font: {
            size: 12
          }
        }
      },
      title: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#333333',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Episode',
          color: '#ffffff'
        },
        ticks: {
          color: '#ffffff',
          maxTicksLimit: 10
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      y: {
        display: true,
        ticks: {
          color: '#ffffff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    animation: {
      duration: 300
    }
  };

  const rewardChartData = useMemo(() => {
    const episodes = data.episodes.slice(-timeWindow);
    const rewards = data.rewards.slice(-timeWindow);
    
    // Calculate moving average for smoother visualization
    const movingAverage = [];
    const windowSize = Math.min(10, episodes.length);
    
    for (let i = 0; i < rewards.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = rewards.slice(start, i + 1);
      const avg = window.reduce((sum, val) => sum + val, 0) / window.length;
      movingAverage.push(avg);
    }

    return {
      labels: episodes,
      datasets: [
        {
          label: 'Episode Reward',
          data: rewards,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 1,
          pointRadius: 1,
          pointHoverRadius: 4,
          tension: 0.1
        },
        {
          label: 'Moving Average',
          data: movingAverage,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3
        }
      ]
    };
  }, [data.episodes, data.rewards, timeWindow]);

  const winRateChartData = useMemo(() => {
    const episodes = data.episodes.slice(-timeWindow);
    const winRates = data.winRates.slice(-timeWindow).map(rate => rate * 100); // Convert to percentage

    return {
      labels: episodes,
      datasets: [
        {
          label: 'Win Rate (%)',
          data: winRates,
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 2,
          pointRadius: 1,
          pointHoverRadius: 4,
          tension: 0.2,
          fill: true
        }
      ]
    };
  }, [data.episodes, data.winRates, timeWindow]);

  const winRateOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        min: 0,
        max: 100,
        title: {
          display: true,
          text: 'Win Rate (%)',
          color: '#ffffff'
        }
      }
    }
  };

  const rewardOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        title: {
          display: true,
          text: 'Reward',
          color: '#ffffff'
        }
      }
    }
  };

  return (
    <div className="performance-graphs">
      <div className="graph-container">
        <h4>Episode Rewards</h4>
        <div className="chart-wrapper">
          <Line data={rewardChartData} options={rewardOptions} />
        </div>
      </div>
      
      <div className="graph-container">
        <h4>Win Rate</h4>
        <div className="chart-wrapper">
          <Line data={winRateChartData} options={winRateOptions} />
        </div>
      </div>

      <div className="graph-stats">
        <div className="stat-item">
          <span className="stat-label">Latest Reward:</span>
          <span className="stat-value">
            {data.rewards.length > 0 ? data.rewards[data.rewards.length - 1].toFixed(2) : 'N/A'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Latest Win Rate:</span>
          <span className="stat-value">
            {data.winRates.length > 0 ? (data.winRates[data.winRates.length - 1] * 100).toFixed(1) + '%' : 'N/A'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Data Points:</span>
          <span className="stat-value">{data.episodes.length}</span>
        </div>
      </div>
    </div>
  );
};

export default PerformanceGraphs;