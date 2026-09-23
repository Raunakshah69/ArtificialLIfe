import { useExperiment } from '../contexts/ExperimentContext';
import { loadGenerationMetrics, loadPopulationSummary, loadStrategyComparison } from '../data/repositories';

export function OverviewPage() {
  const experiment = useExperiment();
  const generationMetrics = loadGenerationMetrics();
  const populationSummary = loadPopulationSummary();
  const strategyComparison = loadStrategyComparison();
  const latestGeneration = generationMetrics[generationMetrics.length - 1] ?? null;

  return (
    <>
      <section>
        <h2>Experiment identity</h2>
        <dl className="summary-grid">
          <div>
            <dt>Experiment</dt>
            <dd>{experiment.experiment_name}</dd>
          </div>
          <div>
            <dt>Experiment ID</dt>
            <dd>{experiment.experiment_id}</dd>
          </div>
          <div>
            <dt>Ticker</dt>
            <dd>{experiment.ticker}</dd>
          </div>
          <div>
            <dt>Date range</dt>
            <dd>
              {experiment.start_date} — {experiment.end_date}
            </dd>
          </div>
          <div>
            <dt>Forecasting backbone</dt>
            <dd>{experiment.forecasting_backbone}</dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd>{experiment.target_mode}</dd>
          </div>
          <div>
            <dt>Development / test status</dt>
            <dd>{experiment.experiment_status}</dd>
          </div>
          <div>
            <dt>Final held-out test</dt>
            <dd>{experiment.final_test_status}</dd>
          </div>
        </dl>
      </section>

      <section>
        <h2>Current experiment summary</h2>
        <div className="stat-grid">
          <div className="stat-card">
            <span>Generation count</span>
            <strong>{generationMetrics.length}</strong>
          </div>
          <div className="stat-card">
            <span>Population size</span>
            <strong>{experiment.population_size}</strong>
          </div>
          <div className="stat-card">
            <span>Starting capital</span>
            <strong>{experiment.starting_capital.toLocaleString()}</strong>
          </div>
          <div className="stat-card">
            <span>Final-generation alive agents</span>
            <strong>{latestGeneration?.alive_count ?? 0}</strong>
          </div>
          <div className="stat-card">
            <span>Final-generation dead agents</span>
            <strong>{latestGeneration?.dead_count ?? 0}</strong>
          </div>
          <div className="stat-card">
            <span>Final-generation mean capital</span>
            <strong>{latestGeneration ? latestGeneration.mean_capital.toFixed(2) : '0.00'}</strong>
          </div>
          <div className="stat-card">
            <span>Final-generation best capital</span>
            <strong>{latestGeneration ? latestGeneration.best_capital.toFixed(2) : '0.00'}</strong>
          </div>
          <div className="stat-card">
            <span>Survival rate</span>
            <strong>{latestGeneration ? (latestGeneration.survival_rate * 100).toFixed(1) + '%' : '0.0%'}</strong>
          </div>
        </div>
      </section>

      <section>
        <h2>Evolution summary</h2>
        <ul className="plain-list">
          {generationMetrics.map((generation) => (
            <li key={generation.generation}>
              <span>Generation {generation.generation}</span>
              <span>mean capital: {generation.mean_capital.toFixed(2)}</span>
              <span>survival: {(generation.survival_rate * 100).toFixed(1)}%</span>
              <span>diversity: {generation.genetic_diversity.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>Best lineage summary</h2>
        {populationSummary.agents.length === 0 ? (
          <p>No lineage data is available for this experiment.</p>
        ) : (
          <dl className="summary-grid">
            <div>
              <dt>Best final agent</dt>
              <dd>{populationSummary.agents[0]?.agent_id ?? 'Not available'}</dd>
            </div>
            <div>
              <dt>Generation</dt>
              <dd>{populationSummary.agents[0]?.generation ?? 0}</dd>
            </div>
            <div>
              <dt>Ending capital</dt>
              <dd>{populationSummary.agents[0]?.ending_capital.toFixed(2) ?? '0.00'}</dd>
            </div>
            <div>
              <dt>Lineage depth</dt>
              <dd>0</dd>
            </div>
            <div>
              <dt>Root ancestor</dt>
              <dd>{populationSummary.agents[0]?.parent_a ?? 'Not recorded'}</dd>
            </div>
            <div>
              <dt>Descendant count</dt>
              <dd>0</dd>
            </div>
          </dl>
        )}
      </section>

      <section>
        <h2>Evaluation summary</h2>
        <div className="split-columns">
          <div>
            <h3>Development results</h3>
            {strategyComparison[0] ? (
              <ul className="plain-list">
                <li>Strategy: {strategyComparison[0].strategy}</li>
                <li>Ending capital: {strategyComparison[0].ending_capital.toFixed(2)}</li>
                <li>Cumulative return: {(strategyComparison[0].cumulative_return * 100).toFixed(2)}%</li>
              </ul>
            ) : (
              <p>This experiment is missing development metrics.</p>
            )}
          </div>
          <div>
            <h3>Final held-out test results</h3>
            <p>Final held-out evaluation has not been run.</p>
          </div>
        </div>
      </section>
    </>
  );
}
