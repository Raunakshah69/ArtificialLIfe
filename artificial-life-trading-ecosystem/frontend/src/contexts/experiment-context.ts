import { createContext, useContext } from 'react';

import type { ExperimentMetadata } from '../types/domain';

interface ExperimentContextValue {
  experiment: ExperimentMetadata;
}

export const ExperimentContext = createContext<ExperimentContextValue | undefined>(undefined);

export function useExperiment() {
  const context = useContext(ExperimentContext);

  if (!context) {
    throw new Error('useExperiment must be used within an ExperimentProvider');
  }

  return context.experiment;
}