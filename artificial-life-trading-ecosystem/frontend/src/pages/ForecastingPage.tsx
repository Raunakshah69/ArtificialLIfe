import { useState } from 'react';

import { formatCount, formatNumber } from '../components/format';
import { loadForecastingResults } from '../data/repositories';

export function ForecastingPage() {
  const results = loadForecastingResults();
  const [selectedName, setSelectedName] = useState(results[0]?.name ?? '');
  const selected = results.find((result) => result.name === selectedName) ?? results[0];

  return (
    <>
      <section className="lead-section forecasting-lead">
        <p className="section-kicker">Shared forecasting backbone</p>
        <div className="section-heading-row"><h2>Forecast model comparison</h2><span className="status-label status-evaluated">FROZEN DURING EVOLUTION</span></div>
        <p>The population shares one trained forecaster. Agents evolve their own decision head; the recurrent backbone is not changed by evolution.</p>
        {selected && <div className="metric-strip forecasting-selected">
          <div><span className="metric-label">Selected model</span><strong>{selected.name}</strong></div>
          <div><span className="metric-label">Parameters</span><strong>{formatCount(selected.parameter_count, 'parameter')}</strong></div>
          <div><span className="metric-label">Validation MAE</span><strong>{formatNumber(selected.validation_mae, 6)}</strong></div>
          <div><span className="metric-label">Validation RMSE</span><strong>{formatNumber(selected.validation_rmse, 6)}</strong></div>
        </div>}
      </section>

      <section>
        <p className="section-kicker">Validation measurements</p>
        {results.length === 0 ? <div className="empty-state">No recorded forecasting comparison is available.</div> : (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Model</th><th>Parameters</th><th>Training time · sec</th><th>Best validation loss</th><th>Validation MAE</th><th>Validation RMSE</th><th>Epochs</th></tr></thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.name} aria-selected={result.name === selected?.name} onClick={() => setSelectedName(result.name)}>
                    <td><button className="table-select" type="button" onClick={(event) => { event.stopPropagation(); setSelectedName(result.name); }}>{result.name}</button></td>
                    <td className="table-numeric">{formatCount(result.parameter_count, 'parameter')}</td>
                    <td className="table-numeric">{formatNumber(result.training_time_seconds, 2)}</td>
                    <td className="table-numeric">{formatNumber(result.best_validation_loss, 6)}</td>
                    <td className="table-numeric">{formatNumber(result.validation_mae, 6)}</td>
                    <td className="table-numeric">{formatNumber(result.validation_rmse, 6)}</td>
                    <td className="table-numeric">{formatCount(result.epochs_trained, 'epoch')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
