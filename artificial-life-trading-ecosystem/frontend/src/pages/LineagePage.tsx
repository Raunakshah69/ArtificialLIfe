import { loadLineageMetrics, loadPopulationSummary } from '../data/repositories';

export function LineagePage() {
  const lineageMetrics = loadLineageMetrics();
  const { agents } = loadPopulationSummary();

  return (
    <>
      <section>
        <h2>Best lineage</h2>
        <dl className="summary-grid">
          <div><dt>Root ancestor</dt><dd>{agents[0]?.parent_a ?? 'Unknown'}</dd></div>
          <div><dt>Generation path</dt><dd>{agents[0]?.generation ?? 0}</dd></div>
          <div><dt>Final agent</dt><dd>{agents[0]?.agent_id ?? 'Not available'}</dd></div>
          <div><dt>Lineage depth</dt><dd>{lineageMetrics.maximum_lineage_depth}</dd></div>
          <div><dt>Descendants</dt><dd>{lineageMetrics.descendants_per_successful_ancestor}</dd></div>
          <div><dt>Survival history</dt><dd>{lineageMetrics.surviving_lineages} surviving</dd></div>
        </dl>
      </section>

      <section>
        <h3>Lineage metrics</h3>
        <dl className="summary-grid">
          <div><dt>Total lineage records</dt><dd>{lineageMetrics.total_lineage_records}</dd></div>
          <div><dt>Unique root lineages</dt><dd>{lineageMetrics.unique_root_lineages}</dd></div>
          <div><dt>Extinct lineages</dt><dd>{lineageMetrics.extinct_lineages}</dd></div>
          <div><dt>Extinction rate</dt><dd>{(lineageMetrics.extinction_rate * 100).toFixed(1)}%</dd></div>
          <div><dt>Average lineage depth</dt><dd>{lineageMetrics.average_lineage_depth.toFixed(2)}</dd></div>
          <div><dt>Generations survived</dt><dd>{lineageMetrics.number_of_generations_survived}</dd></div>
        </dl>
      </section>
    </>
  );
}
