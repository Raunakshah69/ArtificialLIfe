import { NavLink, Outlet } from 'react-router-dom';

import { useExperiment } from '../contexts/ExperimentContext';

const navigation = [
  { label: 'Overview', to: '/' },
  { label: 'Forecasting', to: '/forecasting' },
  { label: 'Population', to: '/population' },
  { label: 'Lineage', to: '/lineage' },
  { label: 'Agents', to: '/agents' },
  { label: 'Backtest', to: '/backtest' },
  { label: 'Replay', to: '/replay' },
  { label: 'Experiment', to: '/experiment' },
];

export function AppLayout() {
  const experiment = useExperiment();

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Artificial-Life Trading Ecosystem</p>
          <h1>{experiment.experiment_name}</h1>
        </div>
        <div className="topbar-meta" aria-label="Selected experiment metadata">
          <span>{experiment.experiment_id}</span>
          <span>{experiment.ticker}</span>
        </div>
      </header>

      <nav className="main-nav" aria-label="Main navigation">
        {navigation.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <main className="page-shell">
        <Outlet />
      </main>
    </div>
  );
}
