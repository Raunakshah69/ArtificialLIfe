import { createContext, useContext } from 'react';

import type { AvailableExperiment, RuntimeBacktest, RuntimeDailyEvent, RuntimeGeneration, SimulationState } from '../types/domain';

export interface SimulationContextValue {
  state: SimulationState | null;
  loading: boolean;
  busy: boolean;
  error: string | null;
  experiments: AvailableExperiment[];
  selectedGeneration: number;
  selectExperiment: (modelName: AvailableExperiment['model_name']) => Promise<void>;
  setSelectedGeneration: (generation: number) => void;
  nextDay: () => Promise<void>;
  runGeneration: () => Promise<void>;
  runGenerations: (count: number) => Promise<void>;
  reset: () => Promise<void>;
  refresh: () => Promise<void>;
  getGeneration: (generation: number) => Promise<RuntimeGeneration>;
  getReplay: (generation: number) => Promise<RuntimeDailyEvent[]>;
  getBacktest: (generation: number, agentId?: string) => Promise<RuntimeBacktest>;
}

export const SimulationContext = createContext<SimulationContextValue | undefined>(undefined);

export function useSimulation(): SimulationContextValue {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
}