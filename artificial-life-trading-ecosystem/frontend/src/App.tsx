import { BrowserRouter, Route, Routes } from 'react-router-dom';

import './App.css';
import { AppLayout } from './components/Layout';
import { ExperimentProvider } from './contexts/ExperimentContext';
import { AgentsPage } from './pages/AgentsPage';
import { BacktestPage } from './pages/BacktestPage';
import { ExperimentPage } from './pages/ExperimentPage';
import { ForecastingPage } from './pages/ForecastingPage';
import { LineagePage } from './pages/LineagePage';
import { OverviewPage } from './pages/OverviewPage';
import { PopulationPage } from './pages/PopulationPage';
import { ReplayPage } from './pages/ReplayPage';

export default function App() {
  return (
    <BrowserRouter>
      <ExperimentProvider>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/forecasting" element={<ForecastingPage />} />
            <Route path="/population" element={<PopulationPage />} />
            <Route path="/lineage" element={<LineagePage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="/replay" element={<ReplayPage />} />
            <Route path="/experiment" element={<ExperimentPage />} />
          </Route>
        </Routes>
      </ExperimentProvider>
    </BrowserRouter>
  );
}
