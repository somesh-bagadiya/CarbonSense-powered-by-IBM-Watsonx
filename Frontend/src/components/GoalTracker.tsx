
import { Progress } from "@/components/ui/progress";

interface GoalTrackerProps {
  value: number;
  target: number;
}

const GoalTracker = ({ value, target }: GoalTrackerProps) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <h3 className="text-lg font-medium text-gray-700 mb-3">Your Goal</h3>
      <p className="text-gray-800 font-medium mb-2">
        Reduce CO<sub>2</sub> by{" "}
        <span className="font-bold">{target}%</span> in 1 month
      </p>
      <Progress value={value} className="h-2 mb-1" />
      <div className="flex justify-between text-xs text-gray-500">
        <span>0%</span>
        <span>{value}% complete</span>
        <span>100%</span>
      </div>
    </div>
  );
};

export default GoalTracker;
