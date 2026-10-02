import type { AvailableExperiment, RuntimeBacktest, RuntimeDailyEvent, RuntimeGeneration, SimulationState } from '../types/domain';

const apiBase = import.meta.env.VITE_API_BASE_URL ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `Simulation API returned ${response.status}.`);
  }
  return (await response.json()) as T;
}

export const loadSimulation = () => request<SimulationState>('/simulation');
export const loadExperiments = async () => (await request<{ experiments: AvailableExperiment[] }>('/experiments')).experiments;
export const selectSimulationExperiment = (modelName: AvailableExperiment['model_name']) =>
  request<SimulationState>('/simulation/select-experiment', { method: 'POST', body: JSON.stringify({ model_name: modelName }) });
export const advanceSimulationDay = () => request<SimulationState>('/simulation/next-day', { method: 'POST' });
export const runSimulationGeneration = () => request<SimulationState>('/simulation/run-generation', { method: 'POST' });
export const runSimulationGenerations = (count: number) =>
  request<SimulationState>('/simulation/run-generations', { method: 'POST', body: JSON.stringify({ count }) });
export const resetSimulation = () => request<SimulationState>('/simulation/reset', { method: 'POST' });
export const loadGeneration = (generation: number) => request<RuntimeGeneration>(`/generations/${generation}`);
export const loadReplayEvents = (generation: number) =>
  request<{ generation: number; events: RuntimeDailyEvent[] }>(`/replay/${generation}`);
export const loadBacktest = (generation: number, agentId?: string) => {
  const query = agentId ? `?agent_id=${encodeURIComponent(agentId)}` : '';
  return request<RuntimeBacktest>(`/backtest/${generation}${query}`);
};