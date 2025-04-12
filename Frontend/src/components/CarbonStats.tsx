
import { CircularProgress } from "@/components/CircularProgress";

interface CarbonStatsProps {
  title: string;
  value: number;
  unit: string;
  subtitle: string;
  progress: number;
  color: string;
}

const CarbonStats = ({ title, value, unit, subtitle, progress, color }: CarbonStatsProps) => {
  const getColorClasses = () => {
    const colors = {
      teal: "text-teal-600 stroke-teal-500",
      emerald: "text-emerald-600 stroke-emerald-500",
      cyan: "text-cyan-600 stroke-cyan-500",
      blue: "text-blue-600 stroke-blue-500",
    };
    return colors[color as keyof typeof colors] || colors.teal;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4 flex flex-col items-center">
      <h3 className="text-lg font-medium text-gray-700 mb-2 text-center">{title}</h3>
      <div className="flex items-center justify-center relative w-32 h-32">
        <CircularProgress 
          progress={progress} 
          size={120} 
          strokeWidth={12} 
          circleOneStroke="#f3f4f6" 
          circleTwoStroke={`currentColor`} 
          className={getColorClasses()}
        />
        <div className="absolute flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-gray-800">{value}</span>
          <span className="text-sm text-gray-500">{unit}</span>
        </div>
      </div>
      <p className="text-sm text-gray-500 mt-2">{subtitle}</p>
    </div>
  );
};

export default CarbonStats;
