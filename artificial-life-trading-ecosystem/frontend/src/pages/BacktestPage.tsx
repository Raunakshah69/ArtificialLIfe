import { loadStrategyComparison } from '../data/repositories';

export function BacktestPage() {
  const strategies = loadStrategyComparison();

  return (
    <>
      <section>
        <h2>Development evaluation</h2>
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Ending capital</th>
              <th>Cumulative return</th>
              <th>Maximum drawdown</th>
              <th>Trade count</th>
              <th>Sharpe ratio</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((strategy) => (
              <tr key={strategy.label}>
                <td>{strategy.label}</td>
                <td>{strategy.ending_capital.toFixed(2)}</td>
                <td>{(strategy.cumulative_return * 100).toFixed(2)}%</td>
                <td>{(strategy.maximum_drawdown * 100).toFixed(2)}%</td>
                <td>{strategy.trade_count}</td>
                <td>{strategy.sharpe_ratio === null ? 'Not valid' : strategy.sharpe_ratio.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section>
        <h2>Final held-out test</h2>
        <p>Final held-out evaluation has not been run.</p>
      </section>
    </>
  );
}
