import { useState, useEffect } from 'react';
import type {
  StatusResponse,
  NodesResponse,
  RunsResponse,
  LeaderboardResponse,
  LeaderboardViewsResponse,
  AlertsResponse,
  QueueResponse,
} from './types';

export type Polled<T> = T & { _loading: boolean };

// Module-level cache: survives tab unmount/remount
const _responseCache = new Map<string, any>();

function usePolling<T>(url: string, interval: number, initial: T): Polled<T> {
  const cached = _responseCache.get(url);
  const [data, setData] = useState<T>(cached ?? initial);
  const [loading, setLoading] = useState(!cached);
  useEffect(() => {
    let active = true;
    const f = () =>
      fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(r.statusText);
          return r.json();
        })
        .then((d) => {
          _responseCache.set(url, d);
          if (active) { setData(d); setLoading(false); }
        })
        .catch(() => {});
    f();
    const id = setInterval(f, interval);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [url, interval]);
  return { ...data, _loading: loading };
}

const EMPTY_STATUS: StatusResponse = {
  timestamp: '',
  host: '',
  iteration: 0,
  active: [],
  free_gpus: [],
  free_gpus_by_host: {},
  queue_depth: 0,
  completed: 0,
  failed: 0,
  disk_free_gb: 0,
  top_results: [],
};

const EMPTY_NODES: NodesResponse = { heartbeats: [], local_gpus: [] };
const EMPTY_RUNS: RunsResponse = { active: [], recent: [] };
const EMPTY_LB: LeaderboardResponse = { top: [], metric: '' };
const EMPTY_ALERTS: AlertsResponse = { alerts: [], count: 0 };

export function useStatus() {
  return usePolling<StatusResponse>('/api/status', 5000, EMPTY_STATUS);
}

export function useNodes() {
  return usePolling<NodesResponse>('/api/nodes', 5000, EMPTY_NODES);
}

export function useRuns() {
  return usePolling<RunsResponse>('/api/runs', 5000, EMPTY_RUNS);
}

const EMPTY_VIEWS: LeaderboardViewsResponse = { views: [] };

export function useLeaderboardViews() {
  return usePolling<LeaderboardViewsResponse>('/api/leaderboard/views', 30000, EMPTY_VIEWS);
}

export function useLeaderboard(view?: string) {
  const url = view ? `/api/leaderboard?view=${encodeURIComponent(view)}` : '/api/leaderboard';
  return usePolling<LeaderboardResponse>(url, 15000, EMPTY_LB);
}

export function useAlerts() {
  return usePolling<AlertsResponse>('/api/alerts', 10000, EMPTY_ALERTS);
}

const EMPTY_QUEUE: QueueResponse = {
  queue: [], total: 0, total_all: 0,
  page: 1, per_page: 50, total_pages: 1, counts: {},
};

export function useQueue(page: number, statusFilter: string, search: string) {
  const params = new URLSearchParams({ page: String(page), per_page: '50' });
  if (statusFilter && statusFilter !== 'all') params.set('status_filter', statusFilter);
  if (search) params.set('search', search);
  return usePolling<QueueResponse>(`/api/queue?${params}`, 10000, EMPTY_QUEUE);
}

export function useConfig() {
  return usePolling<Record<string, any>>('/api/config', 30000, {});
}
