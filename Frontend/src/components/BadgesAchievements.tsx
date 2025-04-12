
import { Award, Leaf, Zap } from "lucide-react";

const BadgesAchievements = () => {
  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <h3 className="text-lg font-medium text-gray-700 mb-3">Badges & Achievements</h3>
      
      <div className="flex justify-between">
        <div className="flex flex-col items-center">
          <div className="bg-green-100 p-2 rounded-full mb-1">
            <Leaf className="text-green-600" size={20} />
          </div>
          <span className="text-xs text-gray-600">Eco Starter</span>
        </div>
        
        <div className="flex flex-col items-center opacity-40">
          <div className="bg-amber-100 p-2 rounded-full mb-1">
            <Award className="text-amber-600" size={20} />
          </div>
          <span className="text-xs text-gray-600">Committed</span>
        </div>
        
        <div className="flex flex-col items-center opacity-40">
          <div className="bg-blue-100 p-2 rounded-full mb-1">
            <Zap className="text-blue-600" size={20} />
          </div>
          <span className="text-xs text-gray-600">Carbon Pro</span>
        </div>
      </div>
    </div>
  );
};

export default BadgesAchievements;
