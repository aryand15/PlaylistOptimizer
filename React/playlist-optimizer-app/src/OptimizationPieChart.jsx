import { Pie } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(ArcElement, Tooltip, Legend, ChartDataLabels);

const OptimizationPieChart = ({ optimizationParams, percentageData, colors }) => {

  const pieChartData = {
    labels: optimizationParams,
    datasets: [
      {
        data: percentageData,
        backgroundColor: colors,
        borderWidth: 1
      }
    ]
  };

  const options = {
    plugins: {
      legend: {
        display: false // Hide the legend labels are displayed on the chart
      },
      tooltip: {
        enabled: true // Keep tooltips for additional information on hover
      },
      datalabels: {
        display: true,
        color: '#fff',
        font: {
          weight: 'bold',
          size: 12
        },
        formatter: (value, context) => {
          const label = context.chart.data.labels[context.dataIndex];
          return value > 0 ? `${label.split(" ")[0]}\n${value}%` : '';
        },
        anchor: 'center',
        align: 'center',
        offset: 0,
        clamp: true
      }
    },
    layout: {
      padding: 20
    }
  };

  return <Pie data={pieChartData} options={options} />;
};

export default OptimizationPieChart;
