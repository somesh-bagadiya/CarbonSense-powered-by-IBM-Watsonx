
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface WeeklyTrendsProps {
  data: number[];
}

const WeeklyTrends = ({ data }: WeeklyTrendsProps) => {
  const getDayNames = () => {
    const days = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      days.push(d.toLocaleDateString('en-US', { weekday: 'short' }));
    }
    return days;
  };

  const chartData = data.map((value, index) => ({
    name: getDayNames()[index],
    value
  }));

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <h3 className="text-lg font-medium text-gray-700 mb-3">Weekly Trends</h3>
      
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis 
              hide={true}
            />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#6366f1" 
              strokeWidth={2}
              dot={{ stroke: '#6366f1', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: '#4f46e5', strokeWidth: 2, fill: '#c7d2fe' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff',
                border: '1px solid #e2e8f0',
                borderRadius: '0.375rem',
                fontSize: '0.75rem'
              }}
              formatter={(value: number) => [`${value} kg CO₂`, 'Carbon']}
              labelFormatter={(name) => `${name}`}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default WeeklyTrends;
