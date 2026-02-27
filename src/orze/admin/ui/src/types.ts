/* ── API response types ── */

export interface ActiveRun {
  idea_id: string;
  gpu: number;
  elapsed_min: number;
  host: string;
  title?: string;
}

export interface StatusResponse {
  timestamp: string;
  host: string;
  iteration: number;
  active: ActiveRun[];
  free_gpus: number[];
  free_gpus_by_host: Record<string, number[]>;
  queue_depth: number;
  completed: number;
  failed: number;
  disk_free_gb: number;
  top_results: TopResult[];
}

export interface TopResult {
  idea_id: string;
  title: string;
  [metric: string]: any;
}

export interface GpuInfo {
  index: number;
  name: string;
  memory_used_mib: number;
  memory_total_mib: number;
  utilization_pct: number;
  temperature_c: number;
}

export interface Heartbeat {
  host: string;
  pid: number;
  timestamp: string;
  epoch: number;
  active: ActiveRun[];
  free_gpus: number[];
  gpu_info?: GpuInfo[];
  disk_free_gb?: number;
  os?: string;
  status: 'online' | 'degraded' | 'offline';
  heartbeat_age_sec: number;
}

export interface NodesResponse {
  heartbeats: Heartbeat[];
  local_gpus: GpuInfo[];
}

export interface Run {
  idea_id: string;
  status: 'RUNNING' | 'QUEUED' | 'COMPLETED' | 'FAILED' | 'ERROR' | string;
  host?: string;
  gpu?: number;
  metric?: number;
  metric_name?: string;
  started_at?: string;
  elapsed_min?: number;
  title?: string;
  tags?: string[];
  claimed_by?: string;
  claimed_gpu?: number;
  error?: string;
}

export interface RunsResponse {
  active: ActiveRun[];
  top_results?: TopResult[];
  recent: Run[];
}

export interface RunDetail {
  idea_id: string;
  metrics: Record<string, any> | null;
  claim: { claimed_by: string; claimed_at: string; gpu: number } | null;
  config?: Record<string, any>;
  title?: string;
}

export interface RunLog {
  idea_id: string;
  log: string;
}

export interface LeaderboardEntry {
  idea_id: string;
  title: string;
  metric_value: number;
  metric_name: string;
  training_time?: number;
}

export interface LeaderboardResponse {
  top: LeaderboardEntry[];
  metric: string;
}

export interface Alert {
  type: string;
  idea_id?: string;
  host?: string;
  error?: string;
  minutes_ago?: number;
  disk_free_gb?: number;
}

export interface AlertsResponse {
  alerts: Alert[];
  count: number;
}

export interface Idea {
  id: string;
  title: string;
  priority?: string;
  status: string;
  config?: any;
}

export interface IdeasResponse {
  ideas: Record<string, any>;
  count: number;
}

export interface QueueItem {
  idea_id: string;
  title: string;
  priority: string;
  status: string;
  config: Record<string, any>;
  category: string;
  parent: string;
  hypothesis: string;
  sweep_parent?: string;
  huggingface?: {
    model_id: string;
    url: string;
    feature_dim: number;
    img_size: number;
    source: string;
  };
}

export interface QueueResponse {
  queue: QueueItem[];
  total: number;
  total_all: number;
  page: number;
  per_page: number;
  total_pages: number;
  counts: Record<string, number>;
}

export type Tab = 'overview' | 'nodes' | 'runs' | 'queue' | 'leaderboard' | 'alerts' | 'settings';
