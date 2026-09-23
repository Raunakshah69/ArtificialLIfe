import { loadReplayData } from '../data/repositories';

export function ReplayPage() {
  const replayData = loadReplayData();

  if (replayData.length === 0) {
    return (
      <section>
        <h2>Replay</h2>
        <p>Replay data is unavailable for this agent.</p>
      </section>
    );
  }

  return (
    <>
      <section>
        <h2>Replay</h2>
        <p>Recorded state only. This replay does not generate new trading decisions.</p>
      </section>
      <section>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Close</th>
              <th>Forecast</th>
              <th>Action</th>
              <th>Position</th>
              <th>Quantity</th>
              <th>Cash</th>
              <th>Equity</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {replayData.map((point, index) => (
              <tr key={`${point.action}-${index}`}>
                <td>{point.date}</td>
                <td>{point.close_price.toFixed(2)}</td>
                <td>{point.forecast}</td>
                <td>{point.action}</td>
                <td>{point.position.toFixed(2)}</td>
                <td>{point.quantity.toFixed(2)}</td>
                <td>{point.cash.toFixed(2)}</td>
                <td>{point.equity.toFixed(2)}</td>
                <td>{point.trade_reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
