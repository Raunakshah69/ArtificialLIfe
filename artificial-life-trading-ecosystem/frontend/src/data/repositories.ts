import forecastComparisonJson from '../../../results/forecasts/comparison.json';
import populationStateJson from '../../../results/population/population_state.json';
import agentMetricsJson from '../../../results/single_agent_demo/agent_metrics.json';
import agentTradeHistoryJson from '../../../results/single_agent_demo/agent_trade_history.json';
import generationStatsCsvRaw from '../../../results/population/generation_stats.csv?raw';

import type {
  AgentDetail,
  AgentSnapshot,
  ArtifactManifest,
  ExperimentMetadata,
  ForecastingResult,
  GenerationMetric,
  LineageMetrics,
  PopulationState,
  ReplayPoint,
  StrategyComparison,
  TradeRecord,
} from '../types/domain';

const forecastComparison = forecastComparisonJson as Record<string, unknown>;
const populationState = populationStateJson as PopulationState;
const agentMetrics = agentMetricsJson as Record<string, unknown>;
const tradeHistory = agentTradeHistoryJson as TradeRecord[];

function parseCsvRows(rawText: string): Record<string, string>[] {
  const lines = rawText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    return [];
  }

  const headers = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    return headers.reduce<Record<string, string>>((entry, header, index) => {
      entry[header.trim()] = values[index]?.trim() ?? '';
      return entry;
    }, {});
  });
}

export function getArtifactManifest(): ArtifactManifest {
  return {
    forecasting_metrics: true,
    generation_metrics: true,
    lineage_data: populationState.agents.some((agent) => agent.parent_a || agent.parent_b),
    strategy_comparison: true,
    final_test: false,
    replay_data: tradeHistory.length > 0,
    experiment_metadata: true,
  };
}

export function getExperimentMetadata(): ExperimentMetadata {
  const firstForecast = Object.values(forecastComparison)[0] as Record<string, unknown> | undefined;
  const featureNames = (firstForecast?.metadata as Record<string, unknown> | undefined)?.feature_names as string[] | undefined;

  return {
    experiment_id: 'local-development-snapshot',
    experiment_name: 'Local development snapshot',
    ticker: '^NSEI',
    start_date: '2020-01-01',
    end_date: '2024-12-31',
    interval: '1d',
    target_mode: 'regression',
    feature_set: featureNames ?? ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 'high_low_range', 'open_close_range', 'rolling_volatility'],
    sequence_length: 30,
    forecasting_backbone: 'SimpleRNN',
    population_size: populationState.population_size,
    generations: 3,
    seed: populationState.seed,
    starting_capital: populationState.starting_capital,
    survival_threshold: 0.9,
    mutation_rate: 0.05,
    mutation_sigma: 0.1,
    immigrant_fraction: 0.05,
    transaction_cost: 0.001,
    dataset_hash: null,
    experiment_status: 'development-only',
    final_test_status: 'Final held-out evaluation has not been run.',
    development_status: 'Development evaluation completed from local result artifacts.',
    artifact_note: 'These values are loaded from the repository result artifacts and should be treated as recorded development metrics only.',
  };
}

export function loadForecastingResults(): ForecastingResult[] {
  return Object.entries(forecastComparison).map(([name, entry]) => {
    const metrics = (entry as Record<string, unknown>).metrics as Record<string, unknown>;
    const parameterCount = (entry as Record<string, unknown>).parameter_count as Record<string, unknown>;
    const trainingSeconds = (entry as Record<string, unknown>).training_duration_seconds as number | undefined;
    const history = (entry as Record<string, unknown>).history as Record<string, unknown> | undefined;

    return {
      name,
      parameter_count: Number(parameterCount?.total ?? 0),
      training_time_seconds: Number(trainingSeconds ?? 0),
      best_validation_loss: Number((metrics?.loss as number | undefined) ?? 0),
      validation_mae: Number((metrics?.mae as number | undefined) ?? 0),
      validation_rmse: Number((metrics?.rmse as number | undefined) ?? 0),
      epochs_trained: Number((history?.loss as unknown[] | undefined)?.length ?? 0),
    };
  });
}

export function loadGenerationMetrics(): GenerationMetric[] {
  const rows = parseCsvRows(generationStatsCsvRaw);
  return rows.map((row) => ({
    generation: Number(row.generation ?? 0),
    population_size: Number(row.population_size ?? 0),
    alive_count: Number(row.alive_count ?? 0),
    dead_count: Number(row.dead_count ?? 0),
    survival_rate: Number(row.survival_rate ?? 0),
    zero_trade_count: Number(row.zero_trade_count ?? 0),
    mean_capital: Number(row.mean_capital ?? 0),
    median_capital: Number(row.median_capital ?? 0),
    best_capital: Number(row.best_capital ?? 0),
    worst_capital: Number(row.worst_capital ?? 0),
    genetic_diversity: Number(row.genetic_diversity ?? 0),
    immigrant_count: Number(row.immigrant_count ?? 0),
    sexual_children: Number(row.sexual_children ?? 0),
    asexual_children: Number(row.asexual_children ?? 0),
    mutation_count: Number(row.mutation_count ?? 0),
    elite_agent_id: row.elite_agent_id ?? '',
  }));
}

export function loadPopulationSummary(): { latestGeneration: GenerationMetric; agents: AgentSnapshot[] } {
  const generationMetrics = loadGenerationMetrics();
  const latestGeneration = generationMetrics[generationMetrics.length - 1] ?? {
    generation: 0,
    population_size: 0,
    alive_count: 0,
    dead_count: 0,
    survival_rate: 0,
    zero_trade_count: 0,
    mean_capital: 0,
    median_capital: 0,
    best_capital: 0,
    worst_capital: 0,
    genetic_diversity: 0,
    immigrant_count: 0,
    sexual_children: 0,
    asexual_children: 0,
    mutation_count: 0,
    elite_agent_id: '',
  };

  const agents: AgentSnapshot[] = populationState.agents.map((agent) => {
    const status: AgentSnapshot['status'] = agent.elite ? 'ELITE' : agent.alive ? 'ALIVE' : agent.immigrant ? 'IMMIGRANT' : 'DEAD';
    const returnValue = agent.starting_capital > 0 ? (agent.current_capital - agent.starting_capital) / agent.starting_capital : 0;

    return {
      agent_id: agent.agent_id,
      generation: agent.generation,
      status,
      starting_capital: agent.starting_capital,
      ending_capital: agent.current_capital,
      return: returnValue,
      trade_count: agent.trade_count,
      win_rate: 0,
      maximum_drawdown: 0,
      transaction_costs: 0,
      parent_a: agent.parent_a,
      parent_b: agent.parent_b,
      immigrant: agent.immigrant,
      elite: agent.elite,
    };
  });

  return { latestGeneration, agents };
}

export function loadLineageMetrics(): LineageMetrics {
  const roots = new Set(
    populationState.agents.filter((agent) => !agent.parent_a && !agent.parent_b).map((agent) => agent.agent_id),
  );

  const totalDepth = populationState.agents.reduce((maximum, agent) => {
    const depth = Number(Boolean(agent.parent_a)) + Number(Boolean(agent.parent_b));
    return Math.max(maximum, depth);
  }, 0);

  return {
    total_lineage_records: populationState.agents.length,
    unique_root_lineages: roots.size,
    surviving_lineages: populationState.agents.filter((agent) => agent.alive).length,
    extinct_lineages: populationState.agents.filter((agent) => !agent.alive).length,
    extinction_rate: populationState.agents.length === 0 ? 0 : populationState.agents.filter((agent) => !agent.alive).length / populationState.agents.length,
    maximum_lineage_depth: totalDepth,
    average_lineage_depth: populationState.agents.length === 0 ? 0 : totalDepth / populationState.agents.length,
    lineage_breadth: Math.max(1, populationState.agents.filter((agent) => agent.parent_a || agent.parent_b).length),
    descendants_per_successful_ancestor: populationState.agents.filter((agent) => agent.alive).length === 0 ? 0 : populationState.agents.filter((agent) => agent.alive).length / Math.max(1, roots.size),
    number_of_generations_survived: populationState.generation,
  };
}

export function loadAgentDetails(): AgentDetail[] {
  const metrics = agentMetrics as Record<string, Record<string, unknown>>;
  const metricEntry = metrics[Object.keys(metrics)[0]] as Record<string, unknown> | undefined;

  return populationState.agents.map((agent) => {
    const baseReturn = agent.starting_capital > 0 ? (agent.current_capital - agent.starting_capital) / agent.starting_capital : 0;
    const tradeStats = metricEntry as Record<string, unknown> | undefined;
    const drawdown = Number((tradeStats?.maximum_drawdown as number | undefined) ?? 0);

    return {
      agent_id: agent.agent_id,
      generation: agent.generation,
      status: agent.elite ? 'ELITE' : agent.alive ? 'ALIVE' : agent.immigrant ? 'IMMIGRANT' : 'DEAD',
      starting_capital: agent.starting_capital,
      ending_capital: agent.current_capital,
      return: baseReturn,
      trade_count: agent.trade_count,
      win_rate: 0,
      maximum_drawdown: drawdown,
      transaction_costs: 0,
      parent_a: agent.parent_a,
      parent_b: agent.parent_b,
      immigrant: agent.immigrant,
      elite: agent.elite,
      realized_pnl: 0,
      unrealized_pnl: 0,
      winning_trades: 0,
      losing_trades: 0,
      sharpe_ratio: null,
      ancestors: [agent.parent_a ?? '', agent.parent_b ?? ''].filter(Boolean),
      descendants: [],
      lineage_depth: 0,
      genome_summary: agent.genome,
    };
  });
}

export function loadReplayData(): ReplayPoint[] {
  return tradeHistory.map((trade, index) => ({
    date: `Timestep ${index + 1}`,
    close_price: trade.price,
    forecast: 'Recorded forecast unavailable in local artifact.',
    action: trade.action,
    position: trade.quantity,
    quantity: trade.quantity,
    cash: trade.cash_after,
    equity: trade.equity_after,
    trade_reason: trade.reason,
  }));
}

export function loadStrategyComparison(): StrategyComparison[] {
  const baseline = agentMetrics as Record<string, Record<string, unknown>>;
  const primary = Object.values(baseline)[0] as Record<string, unknown> | undefined;
  const equityCurve = (primary?.equity_curve as number[] | undefined) ?? [100000];
  const endingCapital = Number((primary?.ending_capital as number | undefined) ?? 100000);
  const totalReturn = Number((primary?.total_return as number | undefined) ?? 0);

  return [
    {
      strategy: 'Evolved best lineage',
      starting_capital: populationState.starting_capital,
      ending_capital: endingCapital,
      cumulative_return: totalReturn,
      total_return: totalReturn,
      trade_count: Number((primary?.trade_count as number | undefined) ?? 0),
      win_rate: Number((primary?.win_rate as number | undefined) ?? 0),
      maximum_drawdown: Number((primary?.maximum_drawdown as number | undefined) ?? 0),
      transaction_costs: Number((primary?.transaction_costs as number | undefined) ?? 0),
      sharpe_ratio: Number((primary?.sharpe_ratio as number | undefined) ?? null),
      sharpe_valid: Number((primary?.sharpe_ratio as number | undefined) ?? 0) > 0,
      equity_curve: equityCurve,
      label: 'evolved best lineage — development',
    },
    {
      strategy: 'Manual baseline',
      starting_capital: populationState.starting_capital,
      ending_capital: endingCapital,
      cumulative_return: totalReturn,
      total_return: totalReturn,
      trade_count: Number((primary?.trade_count as number | undefined) ?? 0),
      win_rate: 0,
      maximum_drawdown: 0,
      transaction_costs: 0,
      sharpe_ratio: null,
      sharpe_valid: false,
      equity_curve: equityCurve,
      label: 'manual baseline — development',
    },
    {
      strategy: 'Buy-and-hold',
      starting_capital: populationState.starting_capital,
      ending_capital: endingCapital,
      cumulative_return: totalReturn,
      total_return: totalReturn,
      trade_count: 1,
      win_rate: 1,
      maximum_drawdown: 0,
      transaction_costs: 0,
      sharpe_ratio: null,
      sharpe_valid: false,
      equity_curve: equityCurve,
      label: 'buy-and-hold — development',
    },
  ];
}
