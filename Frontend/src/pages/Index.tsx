
import { useState, useCallback } from "react";
import ChatInterface from "@/components/ChatInterface";
import CarbonStats from "@/components/CarbonStats";
import GoalTracker from "@/components/GoalTracker";
import BadgesAchievements from "@/components/BadgesAchievements";
import WeeklyTrends from "@/components/WeeklyTrends";
import { CarbonData } from "@/types/carbon";
import { toast } from "sonner";

const Index = () => {
  const [carbonData, setCarbonData] = useState<CarbonData>({
    total: 8.2,
    food: 2.5,
    household: 3.1,
    transportation: 2.6,
    goal: { target: 20, progress: 30 },
    weeklyData: [4.2, 5.1, 6.3, 5.8, 7.2, 6.9, 8.2],
    history: []
  });

  const [messages, setMessages] = useState<Array<{text: string, sender: 'user' | 'bot'}>>([]);

  const processLLMResponse = useCallback((message: string, response: string) => {
    // Extract carbon data from LLM response
    let category: 'food' | 'household' | 'transportation' | null = null;
    let extractedValue = 0;
    
    // Check for carbon data in the response
    const carbonMatch = response.match(/(\d+(?:\.\d+)?)\s*kg\s*(?:of)?\s*CO[₂2]/i);
    
    if (carbonMatch) {
      extractedValue = parseFloat(carbonMatch[1]);
      
      // Determine category based on message content
      if (message.toLowerCase().includes('food') || 
          message.toLowerCase().includes('eat') || 
          message.toLowerCase().includes('meal') ||
          message.toLowerCase().includes('beef') ||
          message.toLowerCase().includes('meat')) {
        category = 'food';
      } else if (message.toLowerCase().includes('electricity') || 
                message.toLowerCase().includes('power') || 
                message.toLowerCase().includes('energy') ||
                message.toLowerCase().includes('gas') ||
                message.toLowerCase().includes('home')) {
        category = 'household';
      } else if (message.toLowerCase().includes('car') || 
                message.toLowerCase().includes('drove') || 
                message.toLowerCase().includes('drive') ||
                message.toLowerCase().includes('bus') ||
                message.toLowerCase().includes('miles') ||
                message.toLowerCase().includes('travel') ||
                message.toLowerCase().includes('transport')) {
        category = 'transportation';
      }
      
      if (category && extractedValue > 0) {
        updateCarbonData(category, extractedValue, message);
        toast.success(`Added ${extractedValue}kg CO₂ to your ${category} footprint`);
      }
    }
    
    return response;
  }, []);
  
  const updateCarbonData = useCallback((
    category: 'food' | 'household' | 'transportation', 
    value: number,
    description: string
  ) => {
    setCarbonData(prev => {
      // Update the specific category
      const updatedCategory = {
        ...prev,
        [category]: parseFloat((prev[category] + value).toFixed(1)),
        total: parseFloat((prev.total + value).toFixed(1)),
        history: [
          ...prev.history,
          {
            timestamp: new Date().toISOString(),
            category,
            value,
            description
          }
        ]
      };
      
      // Update weekly data (assuming last day is today)
      const newWeeklyData = [...prev.weeklyData];
      newWeeklyData[newWeeklyData.length - 1] = parseFloat(
        (newWeeklyData[newWeeklyData.length - 1] + value).toFixed(1)
      );
      updatedCategory.weeklyData = newWeeklyData;
      
      // Update goal progress
      const totalReduction = prev.history.reduce((sum, item) => sum + item.value, 0);
      const newProgress = Math.min(
        Math.round((totalReduction / prev.goal.target) * 100), 
        100
      );
      updatedCategory.goal.progress = newProgress;
      
      return updatedCategory;
    });
  }, []);

  const handleChatSubmit = async (message: string) => {
    // Add user message to chat
    setMessages(prev => [...prev, {text: message, sender: 'user'}]);
    
    // Process the message to determine what kind of carbon calculation to make
    let response = "";
    
    // Simple pattern matching for demo purposes
    if (message.toLowerCase().includes("mile") || message.toLowerCase().includes("drove")) {
      const milesMatch = message.match(/(\d+)\s*miles/);
      if (milesMatch) {
        const miles = parseInt(milesMatch[1]);
        const emissions = (miles * 0.36).toFixed(1); // Simple calculation for demo
        response = `Driving ${miles} miles emits about ${emissions} kg of CO₂.`;
        
        // This is where we would normally wait for LLM response
        // Instead we'll process our simulated response
        processLLMResponse(message, response);
      }
    } else if (message.toLowerCase().includes("beef") || message.toLowerCase().includes("meat")) {
      const servingMatch = message.match(/(\d+)\s*(?:serving|portion|oz|ounce|gram|g)/i);
      const servings = servingMatch ? parseInt(servingMatch[1]) : 1;
      const emissions = (servings * 7).toFixed(1);
      
      response = `A serving of beef produces around ${emissions} kg of CO₂.`;
      
      // Process this simulated response
      if (message.toLowerCase().includes("eat") || message.toLowerCase().includes("ate") || message.toLowerCase().includes("had")) {
        processLLMResponse(message, response);
      }
    } else if (message.toLowerCase().includes("energy") || message.toLowerCase().includes("electricity") || message.toLowerCase().includes("power")) {
      const kwhMatch = message.match(/(\d+)\s*(?:kwh|kilowatt|kw)/i);
      const kwh = kwhMatch ? parseInt(kwhMatch[1]) : 30;
      const emissions = (kwh * 0.5).toFixed(1);
      
      response = `Using ${kwh} kWh produces approximately ${emissions} kg of CO₂.`;
      processLLMResponse(message, response);
    } else {
      response = "I can help you track your carbon footprint. Try telling me about your activities like driving or food choices.";
    }
    
    // Add bot response with a slight delay to simulate processing
    setTimeout(() => {
      setMessages(prev => [...prev, {text: response, sender: 'bot'}]);
    }, 500);
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-6">
          <h1 className="text-4xl font-bold text-gray-800">CarbonSense</h1>
        </header>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <CarbonStats 
            title="Total Carbon Footprint" 
            value={carbonData.total} 
            unit="kg"
            subtitle="Today" 
            progress={75} 
            color="teal"
          />
          <CarbonStats 
            title="Food & Beverages" 
            value={carbonData.food} 
            unit="kg"
            subtitle="Today's intake" 
            progress={60} 
            color="emerald"
          />
          <CarbonStats 
            title="Household Energy Use" 
            value={carbonData.household} 
            unit="kg"
            subtitle="Elec, Gas" 
            progress={40} 
            color="cyan"
          />
          <CarbonStats 
            title="Transportation" 
            value={carbonData.transportation} 
            unit="kg"
            subtitle="Today" 
            progress={30} 
            color="blue"
          />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2 bg-white rounded-lg shadow-md p-4">
            <ChatInterface messages={messages} onSendMessage={handleChatSubmit} />
          </div>
          
          <div className="space-y-4">
            <GoalTracker value={carbonData.goal.progress} target={carbonData.goal.target} />
            <BadgesAchievements />
            <WeeklyTrends data={carbonData.weeklyData} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
