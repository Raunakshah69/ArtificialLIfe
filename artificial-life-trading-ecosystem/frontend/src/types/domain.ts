export type ExperimentStatus =
  | 'preparing'
  | 'development-only'
  | 'development-complete'
  | 'final-test-complete'
  | 'incomplete'
  | 'invalid';

export interface ExperimentMetadata {
  experiment_id: string;
  experiment_name: string;
  ticker: string;
  start_date: string;
  end_date: string;
  interval: string;
  target_mode: string;
  feature_set: string[];
  sequence_length: number;
  forecasting_backbone: string;
  population_size: number;
  generations: number;
  seed: number;
  starting_capital: number;
  survival_threshold: number;
  mutation_rate: number;
  mutation_sigma: number;
  immigrant_fraction: number;
  transaction_cost: number;
  dataset_hash: string | null;
  experiment_status: ExperimentStatus;
  final_test_status: string;
  development_status: string;
  artifact_note: string;
}

export interface ForecastingResult {
  name: string;
  parameter_count: number;
  training_time_seconds: number;
  best_validation_loss: number;
  validation_mae: number;
  validation_rmse: number;
  epochs_trained: number;
}

export interface GenerationMetric {
  generation: number;
  population_size: number;
  alive_count: number;
  dead_count: number;
  survival_rate: number;
  zero_trade_count: number;
  mean_capital: number;
  median_capital: number;
  best_capital: number;
  worst_capital: number;
  genetic_diversity: number;
  immigrant_count: number;
  sexual_children: number;
  asexual_children: number;
  mutation_count: number;
  elite_agent_id: string;
}

export interface StrategyComparison {
  strategy: string;
  starting_capital: number;
  ending_capital: number;
  cumulative_return: number;
  total_return: number;
  trade_count: number;
  win_rate: number;
  maximum_drawdown: number;
  transaction_costs: number;
  sharpe_ratio: number | null;
  sharpe_valid: boolean;
  equity_curve: number[];
  label: string;
}

export interface LineageNode {
  agent_id: string;
  generation: number;
  parent_a: string | null;
  parent_b: string | null;
  immigrant: boolean;
  alive: boolean;
  starting_capital: number;
  ending_capital: number;
  total_return: number;
  trade_count: number;
  lineage_depth: number;
  descendant_count: number;
}

export interface LineageMetrics {
  total_lineage_records: number;
  unique_root_lineages: number;
  surviving_lineages: number;
  extinct_lineages: number;
  extinction_rate: number;
  maximum_lineage_depth: number;
  average_lineage_depth: number;
  lineage_breadth: number;
  descendants_per_successful_ancestor: number;
  number_of_generations_survived: number;
}

export interface AgentSnapshot {
  agent_id: string;
  generation: number;
  status: 'ALIVE' | 'DEAD' | 'IMMIGRANT' | 'ELITE';
  starting_capital: number;
  ending_capital: number;
  return: number;
  trade_count: number;
  win_rate: number;
  maximum_drawdown: number;
  transaction_costs: number;
  parent_a: string | null;
  parent_b: string | null;
  immigrant: boolean;
  elite: boolean;
}

export interface AgentDetail extends AgentSnapshot {
  realized_pnl: number;
  unrealized_pnl: number;
  winning_trades: number;
  losing_trades: number;
  sharpe_ratio: number | null;
  ancestors: string[];
  descendants: string[];
  lineage_depth: number;
  genome_summary: Record<string, number | string | boolean>;
}

export interface TradeRecord {
  agent_id: string;
  timestamp: number;
  action: string;
  price: number;
  quantity: number;
  transaction_cost: number;
  realized_pnl: number;
  cash_after: number;
  equity_after: number;
  reason: string;
}

export interface EquityPoint {
  step: number;
  value: number;
}

export interface ReplayPoint {
  date: string;
  close_price: number;
  forecast: string;
  action: string;
  position: number;
  quantity: number;
  cash: number;
  equity: number;
  trade_reason: string;
}

export interface ArtifactManifest {
  forecasting_metrics: boolean;
  generation_metrics: boolean;
  lineage_data: boolean;
  strategy_comparison: boolean;
  final_test: boolean;
  replay_data: boolean;
  experiment_metadata: boolean;
}

export interface PopulationState {
  generation: number;
  population_size: number;
  starting_capital: number;
  seed: number;
  agents: Array<{
    agent_id: string;
    generation: number;
    alive: boolean;
    starting_capital: number;
    cash: number;
    current_capital: number;
    position_quantity: number;
    parent_a: string | null;
    parent_b: string | null;
    immigrant: boolean;
    elite: boolean;
    trade_count: number;
    genome: Record<string, number | string | boolean>;
  }>;
}
