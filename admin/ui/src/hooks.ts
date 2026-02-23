import { useState, useEffect } from 'react';
import type {
  StatusResponse,
  FleetResponse,
  RunsResponse,
  LeaderboardResponse,
  AlertsResponse,
  QueueResponse,
} from './types';

function usePolling<T>(url: string, interval: number, initial: T): T {
  const [data, setData] = useState<T>(initial);
  useEffect(() => {
    let active = true;
    const f = () =>
      fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(r.statusText);
          return r.json();
        })
        .then((d) => {
          if (active) setData(d);
        })
        .catch(() => {});
    f();
    const id = setInterval(f, interval);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [url, interval]);
  return data;
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

const EMPTY_FLEET: FleetResponse = { heartbeats: [], local_gpus: [] };
const EMPTY_RUNS: RunsResponse = { active: [], recent: [] };
const EMPTY_LB: LeaderboardResponse = { top: [], metric: '' };
const EMPTY_ALERTS: AlertsResponse = { alerts: [], count: 0 };

export function useStatus() {
  return usePolling<StatusResponse>('/api/status', 5000, EMPTY_STATUS);
}

export function useFleet() {
  return usePolling<FleetResponse>('/api/fleet', 5000, EMPTY_FLEET);
}

export function useRuns() {
  return usePolling<RunsResponse>('/api/runs', 5000, EMPTY_RUNS);
}

export function useLeaderboard() {
  return usePolling<LeaderboardResponse>('/api/leaderboard', 15000, EMPTY_LB);
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
