import { loadForecastingResults } from '../data/repositories';

export function ForecastingPage() {
  const results = loadForecastingResults();

  return (
    <>
      <section>
        <h2>Forecasting model comparison</h2>
        <p>
          The forecasting model is shared by the trading agents. During population evolution, the forecasting backbone remains frozen while the agents evolve their decision behavior.
        </p>
      </section>

      <section>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameter count</th>
              <th>Training time (s)</th>
              <th>Best validation loss</th>
              <th>Validation MAE</th>
              <th>Validation RMSE</th>
              <th>Epochs trained</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result) => (
              <tr key={result.name}>
                <td>{result.name}</td>
                <td>{result.parameter_count}</td>
                <td>{result.training_time_seconds.toFixed(2)}</td>
                <td>{result.best_validation_loss.toFixed(6)}</td>
                <td>{result.validation_mae.toFixed(6)}</td>
                <td>{result.validation_rmse.toFixed(6)}</td>
                <td>{result.epochs_trained}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
