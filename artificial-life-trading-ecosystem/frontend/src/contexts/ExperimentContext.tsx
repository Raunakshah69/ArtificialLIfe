import { createContext, useContext, useMemo, type ReactNode } from 'react';

import { getExperimentMetadata } from '../data/repositories';
import type { ExperimentMetadata } from '../types/domain';

interface ExperimentContextValue {
  experiment: ExperimentMetadata;
}

const ExperimentContext = createContext<ExperimentContextValue | undefined>(undefined);

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const value = useMemo(() => ({ experiment: getExperimentMetadata() }), []);

  return <ExperimentContext.Provider value={value}>{children}</ExperimentContext.Provider>;
}

export function useExperiment() {
  const context = useContext(ExperimentContext);

  if (!context) {
    throw new Error('useExperiment must be used within an ExperimentProvider');
  }

  return context.experiment;
}
