
export interface CarbonData {
  total: number;
  food: number;
  household: number;
  transportation: number;
  goal: {
    target: number;
    progress: number;
  };
  weeklyData: number[];
  history: {
    timestamp: string;
    category: 'food' | 'household' | 'transportation';
    value: number;
    description: string;
  }[];
}
