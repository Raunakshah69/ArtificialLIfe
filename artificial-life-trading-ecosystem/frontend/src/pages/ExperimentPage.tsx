import { useExperiment } from '../contexts/ExperimentContext';
import { getArtifactManifest } from '../data/repositories';

export function ExperimentPage() {
  const experiment = useExperiment();
  const manifest = getArtifactManifest();

  return (
    <>
      <section>
        <h2>Experiment metadata</h2>
        <dl className="summary-grid">
          <div><dt>Experiment ID</dt><dd>{experiment.experiment_id}</dd></div>
          <div><dt>Experiment name</dt><dd>{experiment.experiment_name}</dd></div>
          <div><dt>Timestamp</dt><dd>{new Date().toISOString()}</dd></div>
          <div><dt>Ticker</dt><dd>{experiment.ticker}</dd></div>
          <div><dt>Source</dt><dd>Local result artifact snapshot</dd></div>
          <div><dt>Date range</dt><dd>{experiment.start_date} — {experiment.end_date}</dd></div>
          <div><dt>Interval</dt><dd>{experiment.interval}</dd></div>
          <div><dt>Dataset hash</dt><dd>{experiment.dataset_hash ?? 'Not recorded'}</dd></div>
          <div><dt>Target mode</dt><dd>{experiment.target_mode}</dd></div>
          <div><dt>Features</dt><dd>{experiment.feature_set.join(', ')}</dd></div>
          <div><dt>Sequence length</dt><dd>{experiment.sequence_length}</dd></div>
          <div><dt>Forecaster</dt><dd>{experiment.forecasting_backbone}</dd></div>
          <div><dt>Random seed</dt><dd>{experiment.seed}</dd></div>
          <div><dt>Population size</dt><dd>{experiment.population_size}</dd></div>
          <div><dt>Generations</dt><dd>{experiment.generations}</dd></div>
          <div><dt>Starting capital</dt><dd>{experiment.starting_capital}</dd></div>
          <div><dt>Survival threshold</dt><dd>{experiment.survival_threshold}</dd></div>
          <div><dt>Mutation rate</dt><dd>{experiment.mutation_rate}</dd></div>
          <div><dt>Mutation sigma</dt><dd>{experiment.mutation_sigma}</dd></div>
          <div><dt>Immigrant fraction</dt><dd>{experiment.immigrant_fraction}</dd></div>
          <div><dt>Transaction cost</dt><dd>{experiment.transaction_cost}</dd></div>
        </dl>
      </section>

      <section>
        <h2>Experiment status</h2>
        <ul className="plain-list">
          <li>{manifest.forecasting_metrics ? '✓ forecasting metrics' : '✗ forecasting metrics'}</li>
          <li>{manifest.generation_metrics ? '✓ generation metrics' : '✗ generation metrics'}</li>
          <li>{manifest.lineage_data ? '✓ lineage data' : '✗ lineage data'}</li>
          <li>{manifest.strategy_comparison ? '✓ strategy comparison' : '✗ strategy comparison'}</li>
          <li>{manifest.final_test ? '✓ final test' : '✗ final test'}</li>
        </ul>
      </section>

      <section>
        <h2>Development status</h2>
        <p>{experiment.development_status}</p>
      </section>

      <section>
        <h2>Final test status</h2>
        <p>{experiment.final_test_status}</p>
      </section>
    </>
  );
}
