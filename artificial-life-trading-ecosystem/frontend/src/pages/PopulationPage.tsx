import { useState } from 'react';

import { loadGenerationMetrics, loadPopulationSummary } from '../data/repositories';

export function PopulationPage() {
  const generationMetrics = loadGenerationMetrics();
  const { agents } = loadPopulationSummary();
  const [selectedGeneration, setSelectedGeneration] = useState<number>(generationMetrics[generationMetrics.length - 1]?.generation ?? 0);
  const current = generationMetrics.find((generation) => generation.generation === selectedGeneration) ?? generationMetrics[generationMetrics.length - 1];

  return (
    <>
      <section>
        <h2>Population</h2>
        <label htmlFor="generation-select">Generation selector</label>
        <select id="generation-select" value={selectedGeneration} onChange={(event) => setSelectedGeneration(Number(event.target.value))}>
          {generationMetrics.map((generation) => (
            <option key={generation.generation} value={generation.generation}>
              Generation {generation.generation}
            </option>
          ))}
        </select>
      </section>

      <section>
        <h3>Generation metrics</h3>
        {current ? (
          <dl className="summary-grid">
            <div><dt>Population size</dt><dd>{current.population_size}</dd></div>
            <div><dt>Alive</dt><dd>{current.alive_count}</dd></div>
            <div><dt>Dead</dt><dd>{current.dead_count}</dd></div>
            <div><dt>Mean capital</dt><dd>{current.mean_capital.toFixed(2)}</dd></div>
            <div><dt>Median capital</dt><dd>{current.median_capital.toFixed(2)}</dd></div>
            <div><dt>Best capital</dt><dd>{current.best_capital.toFixed(2)}</dd></div>
            <div><dt>Worst capital</dt><dd>{current.worst_capital.toFixed(2)}</dd></div>
            <div><dt>Survival rate</dt><dd>{(current.survival_rate * 100).toFixed(1)}%</dd></div>
            <div><dt>Genetic diversity</dt><dd>{current.genetic_diversity.toFixed(3)}</dd></div>
          </dl>
        ) : (
          <p>Generation metrics are missing.</p>
        )}
      </section>

      <section>
        <h3>Evolutionary metrics</h3>
        {current ? (
          <dl className="summary-grid">
            <div><dt>Mutation count</dt><dd>{current.mutation_count}</dd></div>
            <div><dt>Immigrant count</dt><dd>{current.immigrant_count}</dd></div>
            <div><dt>Sexual reproduction count</dt><dd>{current.sexual_children}</dd></div>
            <div><dt>Asexual reproduction count</dt><dd>{current.asexual_children}</dd></div>
            <div><dt>Zero-trade count</dt><dd>{current.zero_trade_count}</dd></div>
            <div><dt>Dead count</dt><dd>{current.dead_count}</dd></div>
          </dl>
        ) : (
          <p>Evolutionary metrics are unavailable.</p>
        )}
      </section>

      <section>
        <h3>Agents in selected generation</h3>
        <table>
          <thead>
            <tr>
              <th>Agent ID</th>
              <th>Status</th>
              <th>Ending capital</th>
              <th>Return</th>
            </tr>
          </thead>
          <tbody>
            {agents.filter((agent) => agent.generation === selectedGeneration).map((agent) => (
              <tr key={agent.agent_id}>
                <td>{agent.agent_id}</td>
                <td>{agent.status}</td>
                <td>{agent.ending_capital.toFixed(2)}</td>
                <td>{(agent.return * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
