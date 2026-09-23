import { loadAgentDetails } from '../data/repositories';

export function AgentsPage() {
  const agents = loadAgentDetails();

  return (
    <>
      <section>
        <h2>Agents</h2>
        <table>
          <thead>
            <tr>
              <th>Agent ID</th>
              <th>Generation</th>
              <th>Status</th>
              <th>Starting capital</th>
              <th>Ending capital</th>
              <th>Return</th>
              <th>Trade count</th>
              <th>Win rate</th>
              <th>Parent A</th>
              <th>Parent B</th>
              <th>Immigrant</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((agent) => (
              <tr key={agent.agent_id}>
                <td>{agent.agent_id}</td>
                <td>{agent.generation}</td>
                <td>{agent.status}</td>
                <td>{agent.starting_capital.toFixed(2)}</td>
                <td>{agent.ending_capital.toFixed(2)}</td>
                <td>{(agent.return * 100).toFixed(2)}%</td>
                <td>{agent.trade_count}</td>
                <td>{(agent.win_rate * 100).toFixed(2)}%</td>
                <td>{agent.parent_a ?? '—'}</td>
                <td>{agent.parent_b ?? '—'}</td>
                <td>{agent.immigrant ? 'Yes' : 'No'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
