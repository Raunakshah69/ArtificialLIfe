import { useEffect, useState, type ReactNode } from 'react';

import {
  advanceSimulationDay,
  loadBacktest,
  loadExperiments,
  loadGeneration,
  loadReplayEvents,
  loadSimulation,
  resetSimulation,
  runSimulationGeneration,
  runSimulationGenerations,
  selectSimulationExperiment,
} from '../data/simulationApi';
import type { AvailableExperiment, SimulationState } from '../types/domain';
import { SimulationContext, type SimulationContextValue } from './simulation-context';

export function SimulationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SimulationState | null>(null);
  const [experiments, setExperiments] = useState<AvailableExperiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedGeneration, setSelectedGeneration] = useState(0);

  async function refresh() {
    try {
      const [nextState, available] = await Promise.all([loadSimulation(), loadExperiments()]);
      setState(nextState);
      setExperiments(available);
      setSelectedGeneration((current) => nextState.generations.some((generation) => generation.generation === current) ? current : nextState.generation);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not load interactive simulation state.');
    } finally {
      setLoading(false);
    }
  }

  async function runControl(operation: () => Promise<SimulationState>) {
    setBusy(true);
    try {
      const nextState = await operation();
      setState(nextState);
      setSelectedGeneration(nextState.generation);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Simulation action failed.');
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    let active = true;
    void Promise.all([loadSimulation(), loadExperiments()])
      .then(([nextState, available]) => {
        if (active) {
          setState(nextState);
          setExperiments(available);
          setSelectedGeneration(nextState.generation);
          setError(null);
        }
      })
      .catch((requestError: unknown) => {
        if (active) setError(requestError instanceof Error ? requestError.message : 'Could not load interactive simulation state.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, []);

  const value: SimulationContextValue = {
    state,
    loading,
    busy,
    error,
    experiments,
    selectedGeneration,
    setSelectedGeneration,
    selectExperiment: (modelName) => runControl(() => selectSimulationExperiment(modelName)),
    nextDay: () => runControl(advanceSimulationDay),
    runGeneration: () => runControl(runSimulationGeneration),
    runGenerations: (count) => runControl(() => runSimulationGenerations(count)),
    reset: () => runControl(resetSimulation),
    refresh,
    getGeneration: loadGeneration,
    getReplay: async (generation) => (await loadReplayEvents(generation)).events,
    getBacktest: loadBacktest,
  };

  return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>;
}