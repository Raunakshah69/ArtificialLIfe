import { useMemo, type ReactNode } from 'react';

import { ExperimentContext } from './experiment-context';
import { getExperimentMetadata } from '../data/repositories';

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const value = useMemo(() => ({ experiment: getExperimentMetadata() }), []);

  return <ExperimentContext.Provider value={value}>{children}</ExperimentContext.Provider>;
}
