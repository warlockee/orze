import { useState, useRef, useEffect, useCallback, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  Award,
  ChevronDown,
  ChevronRight,
  Clock,
  Cpu,
  HardDrive,
  Layers,
  Monitor,
  Play,
  Settings,
  Shield,
  Square,
  Thermometer,
  Trash2,
  TrendingUp,
  Zap,
  XCircle,
  RefreshCw,
  Server,
  BarChart3,
  Bell,
  FileText,
  ListOrdered,
  Search,
  Filter,
} from 'lucide-react';

import { useStatus, useFleet, useRuns, useLeaderboard, useAlerts, useConfig, useQueue } from './hooks';
import { stopAll, killRun, fetchRunDetail, fetchRunLog } from './api';
import type {
  Tab,
  Heartbeat,
  GpuInfo,
  Run,
  ActiveRun,
  LeaderboardEntry,
  Alert,
  RunDetail,
  QueueItem,
} from './types';

/* ─── tiny primitives ─── */

function Badge({ children, color = 'gray' }: { children: ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    green: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    red: 'bg-red-500/20 text-red-300 border-red-500/30',
    yellow: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    blue: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    purple: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    gray: 'bg-gray-500/20 text-gray-300 border-gray-500/30',
    cyan: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${colors[color] ?? colors.gray}`}
    >
      {children}
    </span>
  );
}

function Pill({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-gray-500">{label}</span>
      <span className="text-sm font-semibold text-gray-100">{value}</span>
      {sub && <span className="text-[10px] text-gray-500">{sub}</span>}
    </div>
  );
}

function Card({
  children,
  className = '',
  onClick,
}: {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22 }}
      onClick={onClick}
      className={`glass rounded-xl p-4 ${onClick ? 'cursor-pointer hover:border-white/10' : ''} ${className}`}
    >
      {children}
    </motion.div>
  );
}

function ProgressBar({ value, max = 100, color = 'emerald' }: { value: number; max?: number; color?: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="h-1.5 w-full rounded-full bg-white/5">
      <motion.div
        className={`h-full rounded-full bg-${color}-500`}
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.5 }}
      />
    </div>
  );
}

function MiniSpark({ data, color = '#34d399', w = 60, h = 20 }: { data: number[]; color?: string; w?: number; h?: number }) {
  if (data.length < 2) return <div style={{ width: w, height: h }} />;
  const mn = Math.min(...data);
  const mx = Math.max(...data);
  const range = mx - mn || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - mn) / range) * h;
    return `${x},${y}`;
  });
  return (
    <svg width={w} height={h} className="inline-block">
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={pts.join(' ')} />
    </svg>
  );
}

function IconKpi({
  icon: Icon,
  label,
  value,
  sub,
  spark,
}: {
  icon: any;
  label: string;
  value: string | number;
  sub?: string;
  spark?: number[];
}) {
  return (
    <Card className="flex items-start gap-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white/5">
        <Icon size={18} className="text-gray-400" />
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] uppercase tracking-wider text-gray-500">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold">{value}</span>
          {spark && spark.length > 1 && <MiniSpark data={spark} />}
        </div>
        {sub && <span className="text-[10px] text-gray-500">{sub}</span>}
      </div>
    </Card>
  );
}

function Segmented({
  items,
  value,
  onChange,
}: {
  items: { key: string; label: string; icon?: any }[];
  value: string;
  onChange: (k: string) => void;
}) {
  return (
    <div className="flex gap-1 rounded-lg bg-white/5 p-1">
      {items.map((it) => {
        const active = it.key === value;
        return (
          <button
            key={it.key}
            onClick={() => onChange(it.key)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-all ${
              active ? 'bg-white/10 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {it.icon && <it.icon size={14} />}
            {it.label}
          </button>
        );
      })}
    </div>
  );
}

function Reveal({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="glass rounded-xl">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-4 py-3 text-sm font-medium text-gray-300 hover:text-white"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {title}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden px-4 pb-4"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function GlowBlob() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute -left-40 -top-40 h-[600px] w-[600px] rounded-full bg-purple-600/10 blur-[160px]" />
      <div className="absolute -bottom-40 -right-40 h-[500px] w-[500px] rounded-full bg-cyan-600/10 blur-[140px]" />
    </div>
  );
}

function FloatingGrid() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-20 opacity-[0.03]">
      <div
        className="h-full w-full"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />
    </div>
  );
}

function Table({
  columns,
  rows,
  onRowClick,
}: {
  columns: { key: string; label: string; w?: string }[];
  rows: Record<string, any>[];
  onRowClick?: (row: Record<string, any>) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-white/5">
            {columns.map((c) => (
              <th
                key={c.key}
                className="pb-2 pr-4 text-[10px] uppercase tracking-wider text-gray-500 font-medium"
                style={c.w ? { width: c.w } : {}}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              onClick={() => onRowClick?.(row)}
              className={`border-b border-white/[0.03] ${onRowClick ? 'cursor-pointer hover:bg-white/[0.02]' : ''}`}
            >
              {columns.map((c) => (
                <td key={c.key} className="py-2 pr-4 text-gray-300">
                  {row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── helpers ─── */

function statusColor(s: string): string {
  if (s === 'online') return 'green';
  if (s === 'degraded') return 'yellow';
  if (s === 'offline') return 'red';
  if (s === 'RUNNING') return 'blue';
  if (s === 'QUEUED') return 'purple';
  if (s === 'COMPLETED') return 'green';
  if (s === 'FAILED' || s === 'ERROR') return 'red';
  return 'gray';
}

function fmtTime(mins: number | undefined): string {
  if (mins == null) return '-';
  if (mins < 60) return `${Math.round(mins)}m`;
  return `${(mins / 60).toFixed(1)}h`;
}

function fmtMetric(v: number | undefined): string {
  if (v == null) return '-';
  return v.toFixed(4);
}

function ago(mins: number | undefined): string {
  if (mins == null) return '';
  if (mins < 1) return 'just now';
  if (mins < 60) return `${Math.round(mins)}m ago`;
  return `${(mins / 60).toFixed(1)}h ago`;
}

function fmtRunName(ideaId: string, title?: string): string {
  const base = ideaId.split('~')[0]; // idea-1234~sweep_param -> idea-1234
  if (title) return `${base} · ${title}`;
  return base;
}

/* ─── main panel ─── */

export default function OrzeAdminPanel() {
  const [tab, setTab] = useState<Tab>('overview');
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runLog, setRunLog] = useState<string>('');
  const [confirmStopAll, setConfirmStopAll] = useState(false);
  const [queueStatusFilter, setQueueStatusFilter] = useState('all');
  const [queueSearch, setQueueSearch] = useState('');
  const [queueSearchDebounced, setQueueSearchDebounced] = useState('');
  const [queuePage, setQueuePage] = useState(1);
  const [expandedQueue, setExpandedQueue] = useState<Set<string>>(new Set());

  // API hooks
  const status = useStatus();
  const fleet = useFleet();
  const runs = useRuns();
  const leaderboard = useLeaderboard();
  const alertsData = useAlerts();
  const config = useConfig();
  const queueData = useQueue(queuePage, queueStatusFilter, queueSearchDebounced);

  // Debounce search input to avoid spamming API
  useEffect(() => {
    const t = setTimeout(() => {
      setQueueSearchDebounced(queueSearch);
      setQueuePage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [queueSearch]);

  // Sparkline accumulator for GPU util
  const gpuSparkRef = useRef<number[]>([]);
  const iterSparkRef = useRef<number[]>([]);
  const queueSparkRef = useRef<number[]>([]);

  useEffect(() => {
    if (!fleet.local_gpus.length) return;
    const avg =
      fleet.local_gpus.reduce((s, g) => s + g.utilization_pct, 0) / fleet.local_gpus.length;
    gpuSparkRef.current = [...gpuSparkRef.current.slice(-19), avg];
  }, [fleet]);

  useEffect(() => {
    if (status.iteration > 0) {
      iterSparkRef.current = [...iterSparkRef.current.slice(-19), status.iteration];
    }
    if (status.queue_depth != null) {
      queueSparkRef.current = [...queueSparkRef.current.slice(-19), status.queue_depth];
    }
  }, [status]);

  // Fetch run detail when selecting a run
  useEffect(() => {
    if (!selectedRun) {
      setRunDetail(null);
      setRunLog('');
      return;
    }
    fetchRunDetail(selectedRun)
      .then(setRunDetail)
      .catch(() => setRunDetail(null));
    fetchRunLog(selectedRun)
      .then((d) => setRunLog(d.log || ''))
      .catch(() => setRunLog(''));
  }, [selectedRun]);

  const handleStopAll = useCallback(async () => {
    await stopAll();
    setConfirmStopAll(false);
  }, []);

  const handleKillRun = useCallback(
    async (id: string) => {
      if (!confirm(`Kill run ${id}?`)) return;
      await killRun(id);
    },
    [],
  );

  // Derived values
  const hosts = fleet.heartbeats;
  const localGpus = fleet.local_gpus;
  const gpuAvg =
    localGpus.length > 0
      ? Math.round(localGpus.reduce((s, g) => s + g.utilization_pct, 0) / localGpus.length)
      : 0;
  const memAvg =
    localGpus.length > 0
      ? Math.round(
          localGpus.reduce((s, g) => s + (g.memory_used_mib / g.memory_total_mib) * 100, 0) /
            localGpus.length,
        )
      : 0;
  const tempMax =
    localGpus.length > 0
      ? Math.max(...localGpus.map((g) => g.temperature_c))
      : 0;

  const activeRuns = runs.active;
  const completedRuns = runs.recent;
  const allRuns: Run[] = [
    ...activeRuns.map((r: ActiveRun) => ({
      idea_id: r.idea_id,
      status: 'RUNNING' as const,
      host: r.host,
      gpu: r.gpu,
      elapsed_min: r.elapsed_min,
      title: r.title,
    })),
    ...completedRuns,
  ];

  const alerts = alertsData.alerts;
  const lbEntries = leaderboard.top;

  const tabs: { key: Tab; label: string; icon: any }[] = [
    { key: 'overview', label: 'Overview', icon: Activity },
    { key: 'fleet', label: 'Fleet', icon: Server },
    { key: 'runs', label: 'Runs', icon: Play },
    { key: 'queue', label: 'Queue', icon: ListOrdered },
    { key: 'leaderboard', label: 'Leaderboard', icon: Award },
    { key: 'alerts', label: 'Alerts', icon: Bell },
    { key: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="relative min-h-screen bg-gray-950 text-gray-100">
      <GlowBlob />
      <FloatingGrid />

      {/* header */}
      <header className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-cyan-500">
              <Zap size={16} className="text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight">ORZE</h1>
              <p className="text-[10px] text-gray-500">Admin Panel</p>
            </div>
          </div>

          <Segmented
            items={tabs.map((t) => ({ key: t.key, label: t.label, icon: t.icon }))}
            value={tab}
            onChange={(k) => setTab(k as Tab)}
          />

          <div className="flex items-center gap-3">
            {alerts.length > 0 && (
              <button
                onClick={() => setTab('alerts')}
                className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 hover:bg-white/10"
              >
                <Bell size={14} />
                <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[9px] font-bold">
                  {alerts.length}
                </span>
              </button>
            )}

            {/* Stop All */}
            <button
              onClick={() => setConfirmStopAll(true)}
              className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/20 transition-colors border border-red-500/20"
            >
              <Square size={12} />
              Stop All
            </button>
          </div>
        </div>
      </header>

      {/* confirm stop-all modal */}
      <AnimatePresence>
        {confirmStopAll && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={() => setConfirmStopAll(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass rounded-2xl p-6 max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/20">
                  <AlertTriangle size={20} className="text-red-400" />
                </div>
                <div>
                  <h3 className="font-bold text-white">Stop All Runs?</h3>
                  <p className="text-xs text-gray-400">This will write a stop sentinel. All farm instances will halt.</p>
                </div>
              </div>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setConfirmStopAll(false)}
                  className="rounded-lg px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleStopAll}
                  className="rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 transition-colors"
                >
                  Confirm Stop
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* body */}
      <main className="mx-auto max-w-7xl px-6 py-6">
        <AnimatePresence mode="wait">
          {tab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              {/* KPI row */}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
                <IconKpi icon={Cpu} label="GPU Util" value={`${gpuAvg}%`} spark={gpuSparkRef.current} />
                <IconKpi icon={HardDrive} label="VRAM" value={`${memAvg}%`} />
                <IconKpi icon={Thermometer} label="Temp Peak" value={`${tempMax}C`} />
                <IconKpi icon={Layers} label="Queue" value={status.queue_depth} spark={queueSparkRef.current} />
                <IconKpi icon={TrendingUp} label="Completed" value={status.completed} />
                <IconKpi icon={AlertTriangle} label="Failed" value={status.failed} />
              </div>

              {/* Two-col: hosts + active runs */}
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Fleet summary */}
                <Card>
                  <div className="mb-3 flex items-center justify-between">
                    <h2 className="text-sm font-bold">Fleet</h2>
                    <Badge color={hosts.length > 0 ? 'green' : 'gray'}>{hosts.length} hosts</Badge>
                  </div>
                  <div className="space-y-2">
                    {hosts.length === 0 && (
                      <p className="text-xs text-gray-500">No heartbeats received yet</p>
                    )}
                    {hosts.map((h) => (
                      <div key={h.host} className="flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div
                            className={`h-2 w-2 rounded-full ${
                              h.status === 'online'
                                ? 'bg-emerald-400'
                                : h.status === 'degraded'
                                ? 'bg-amber-400'
                                : 'bg-red-400'
                            }`}
                          />
                          <span className="text-xs font-medium">{h.host}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-gray-500">
                            {h.free_gpus.length} free GPU{h.free_gpus.length !== 1 ? 's' : ''}
                          </span>
                          <Badge color={statusColor(h.status)}>{h.status}</Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* Active runs */}
                <Card>
                  <div className="mb-3 flex items-center justify-between">
                    <h2 className="text-sm font-bold">Active Runs</h2>
                    <Badge color="blue">{activeRuns.length} running</Badge>
                  </div>
                  <div className="space-y-2">
                    {activeRuns.length === 0 && (
                      <p className="text-xs text-gray-500">No active runs</p>
                    )}
                    {activeRuns.map((r) => (
                      <div key={`${r.idea_id}-${r.gpu}`} className="flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-2">
                        <div className="flex items-center gap-2">
                          <Play size={12} className="text-blue-400" />
                          <span className="text-xs font-medium">{fmtRunName(r.idea_id, r.title)}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-gray-500">GPU {r.gpu}</span>
                          <span className="text-[10px] text-gray-500">{fmtTime(r.elapsed_min)}</span>
                          <span className="text-[10px] text-gray-500">{r.host}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>

              {/* Top results mini-leaderboard */}
              {status.top_results.length > 0 && (
                <Card>
                  <h2 className="mb-3 text-sm font-bold">Top Results</h2>
                  <div className="space-y-1">
                    {status.top_results.slice(0, 5).map((r, i) => (
                      <div key={r.idea_id} className="flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-2">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-gray-500">#{i + 1}</span>
                          <span className="text-xs font-medium">{r.idea_id}</span>
                          {r.title && <span className="text-[10px] text-gray-500 truncate max-w-[200px]">{r.title}</span>}
                        </div>
                        <span className="text-xs font-mono text-emerald-400">
                          {(() => {
                            // Find the first numeric value that looks like a metric
                            for (const [k, v] of Object.entries(r)) {
                              if (k !== 'idea_id' && k !== 'title' && typeof v === 'number') return v.toFixed(4);
                            }
                            return '-';
                          })()}
                        </span>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Disk space */}
              <Card className="max-w-xs">
                <div className="flex items-center gap-2 mb-2">
                  <HardDrive size={14} className="text-gray-400" />
                  <span className="text-xs font-medium">Disk</span>
                </div>
                <span className="text-lg font-bold">{status.disk_free_gb.toFixed(0)} GB</span>
                <span className="text-[10px] text-gray-500 ml-1">free</span>
              </Card>
            </motion.div>
          )}

          {tab === 'fleet' && (
            <motion.div
              key="fleet"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <h2 className="text-lg font-bold">Fleet</h2>

              {hosts.length === 0 && (
                <Card>
                  <p className="text-sm text-gray-500">No hosts reporting. Waiting for heartbeats...</p>
                </Card>
              )}

              {hosts.map((h) => (
                <Card key={h.host}>
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Monitor size={16} className="text-gray-400" />
                      <span className="font-bold text-sm">{h.host}</span>
                      <Badge color={statusColor(h.status)}>{h.status}</Badge>
                    </div>
                    <span className="text-[10px] text-gray-500">
                      heartbeat {Math.round(h.heartbeat_age_sec)}s ago
                    </span>
                  </div>

                  <div className="flex gap-6 mb-3">
                    <Pill label="PID" value={h.pid} />
                    <Pill label="Free GPUs" value={h.free_gpus.join(', ') || 'none'} />
                    {h.disk_free_gb != null && <Pill label="Disk Free" value={`${h.disk_free_gb.toFixed(0)} GB`} />}
                    {h.os && <Pill label="OS" value={h.os} />}
                  </div>

                  {h.active.length > 0 && (
                    <Reveal title={`Active Runs (${h.active.length})`} defaultOpen>
                      <div className="space-y-1">
                        {h.active.map((r) => (
                          <div key={`${r.idea_id}-${r.gpu}`} className="flex items-center justify-between text-xs">
                            <span className="text-gray-300">{fmtRunName(r.idea_id)}</span>
                            <div className="flex gap-3">
                              <span className="text-gray-500">GPU {r.gpu}</span>
                              <span className="text-gray-500">{fmtTime(r.elapsed_min)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </Reveal>
                  )}

                  {/* GPU cards */}
                  {h.gpu_info && h.gpu_info.length > 0 && (
                    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {h.gpu_info.map((g) => (
                        <GpuCard key={g.index} gpu={g} />
                      ))}
                    </div>
                  )}
                </Card>
              ))}

              {/* Local GPU details */}
              {localGpus.length > 0 && (
                <Card>
                  <h3 className="mb-3 text-sm font-bold">Local GPUs (this host)</h3>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {localGpus.map((g) => (
                      <GpuCard key={g.index} gpu={g} />
                    ))}
                  </div>
                </Card>
              )}
            </motion.div>
          )}

          {tab === 'runs' && (
            <motion.div
              key="runs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold">Runs</h2>
                <div className="flex gap-2">
                  <Badge color="blue">{activeRuns.length} running</Badge>
                  <Badge color="green">{completedRuns.filter((r) => r.status === 'COMPLETED').length} completed</Badge>
                  <Badge color="red">{completedRuns.filter((r) => r.status === 'FAILED' || r.status === 'ERROR').length} failed</Badge>
                </div>
              </div>

              <div className="grid gap-6 lg:grid-cols-3">
                {/* runs list */}
                <div className="lg:col-span-2 space-y-2">
                  {allRuns.length === 0 && (
                    <Card>
                      <p className="text-sm text-gray-500">No runs yet</p>
                    </Card>
                  )}
                  {allRuns.map((r) => (
                    <Card
                      key={r.idea_id}
                      onClick={() => setSelectedRun(r.idea_id)}
                      className={`${selectedRun === r.idea_id ? 'border-purple-500/30' : ''}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Badge color={statusColor(r.status)}>{r.status}</Badge>
                          <span className="text-sm font-medium">{fmtRunName(r.idea_id, r.title)}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          {r.host && <span className="text-[10px] text-gray-500">{r.host}</span>}
                          {r.gpu != null && <span className="text-[10px] text-gray-500">GPU {r.gpu}</span>}
                          <span className="text-[10px] text-gray-500">{fmtTime(r.elapsed_min)}</span>
                          {r.metric != null && (
                            <span className="text-xs font-mono text-emerald-400">{fmtMetric(r.metric)}</span>
                          )}
                          {r.status === 'RUNNING' && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleKillRun(r.idea_id);
                              }}
                              className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-red-400 hover:bg-red-500/10 transition-colors"
                            >
                              <XCircle size={10} />
                              Kill
                            </button>
                          )}
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>

                {/* run detail sidebar */}
                <div className="space-y-4">
                  {selectedRun ? (
                    <>
                      <Card>
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="text-sm font-bold">{selectedRun}</h3>
                          <button onClick={() => setSelectedRun(null)} className="text-gray-500 hover:text-white">
                            <XCircle size={14} />
                          </button>
                        </div>
                        {runDetail ? (
                          <div className="space-y-2 text-xs">
                            {runDetail.claim && (
                              <div className="space-y-1">
                                <Pill label="Host" value={runDetail.claim.claimed_by} />
                                <Pill label="GPU" value={runDetail.claim.gpu} />
                                <Pill label="Claimed At" value={runDetail.claim.claimed_at} />
                              </div>
                            )}
                            {runDetail.metrics && (
                              <Reveal title="Metrics" defaultOpen>
                                <pre className="text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-48">
                                  {JSON.stringify(runDetail.metrics, null, 2)}
                                </pre>
                              </Reveal>
                            )}
                            <button
                              onClick={() => handleKillRun(selectedRun)}
                              className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/20 transition-colors border border-red-500/20 mt-2"
                            >
                              <XCircle size={12} />
                              Kill Run
                            </button>
                          </div>
                        ) : (
                          <p className="text-xs text-gray-500">Loading...</p>
                        )}
                      </Card>
                      {runLog && (
                        <Card>
                          <h3 className="mb-2 text-sm font-bold flex items-center gap-2">
                            <FileText size={14} />
                            Log
                          </h3>
                          <pre className="text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-96 font-mono">
                            {runLog}
                          </pre>
                        </Card>
                      )}
                    </>
                  ) : (
                    <Card>
                      <p className="text-xs text-gray-500">Select a run to view details</p>
                    </Card>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {tab === 'queue' && (
            <motion.div
              key="queue"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <QueueTab
                queue={queueData.queue}
                counts={queueData.counts}
                total={queueData.total}
                totalAll={queueData.total_all}
                page={queueData.page}
                totalPages={queueData.total_pages}
                statusFilter={queueStatusFilter}
                onStatusFilter={(s) => { setQueueStatusFilter(s); setQueuePage(1); }}
                search={queueSearch}
                onSearch={setQueueSearch}
                onPageChange={setQueuePage}
                expanded={expandedQueue}
                onToggle={(id) => {
                  setExpandedQueue((prev) => {
                    const next = new Set(prev);
                    if (next.has(id)) next.delete(id);
                    else next.add(id);
                    return next;
                  });
                }}
              />
            </motion.div>
          )}

          {tab === 'leaderboard' && (
            <motion.div
              key="leaderboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold">Leaderboard</h2>
                <Badge color="purple">{lbEntries.length} entries</Badge>
              </div>

              <Card>
                <Table
                  columns={[
                    { key: 'rank', label: '#', w: '40px' },
                    { key: 'idea_id', label: 'Idea' },
                    { key: 'title', label: 'Title' },
                    { key: 'metric', label: leaderboard.metric || 'Metric' },
                    { key: 'time', label: 'Training Time' },
                  ]}
                  rows={lbEntries.map((e: any, i: number) => ({
                    rank: (
                      <span className={i < 3 ? 'text-amber-400 font-bold' : ''}>
                        {i + 1}
                      </span>
                    ),
                    idea_id: (
                      <span className="font-mono text-xs">{e.idea_id}</span>
                    ),
                    title: (
                      <span className="text-xs text-gray-400 truncate block max-w-[300px]">
                        {e.title || '-'}
                      </span>
                    ),
                    metric: (
                      <span className="font-mono text-emerald-400">
                        {typeof e.metric_value === 'number'
                          ? e.metric_value.toFixed(4)
                          : typeof e.auc_roc === 'number'
                          ? e.auc_roc.toFixed(4)
                          : '-'}
                      </span>
                    ),
                    time: (
                      <span className="text-xs text-gray-500">
                        {e.training_time ? fmtTime(e.training_time / 60) : '-'}
                      </span>
                    ),
                    _raw: e,
                  }))}
                  onRowClick={(row) => {
                    setSelectedRun(row._raw.idea_id);
                    setTab('runs');
                  }}
                />
              </Card>
            </motion.div>
          )}

          {tab === 'alerts' && (
            <motion.div
              key="alerts"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold">Alerts</h2>
                <Badge color={alerts.length > 0 ? 'red' : 'green'}>
                  {alerts.length === 0 ? 'All clear' : `${alerts.length} alert${alerts.length !== 1 ? 's' : ''}`}
                </Badge>
              </div>

              {alerts.length === 0 && (
                <Card>
                  <div className="flex items-center gap-3 py-4">
                    <Shield size={20} className="text-emerald-400" />
                    <span className="text-sm text-gray-400">No active alerts</span>
                  </div>
                </Card>
              )}

              <div className="space-y-3">
                {alerts.map((a, i) => (
                  <AlertCard key={i} alert={a} />
                ))}
              </div>
            </motion.div>
          )}

          {tab === 'settings' && (
            <motion.div
              key="settings"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <h2 className="text-lg font-bold">Configuration</h2>
              <Card>
                <pre className="text-xs text-gray-400 whitespace-pre-wrap overflow-auto max-h-[70vh] font-mono">
                  {Object.keys(config).length > 0
                    ? JSON.stringify(config, null, 2)
                    : 'Loading configuration...'}
                </pre>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

/* ─── sub-components ─── */

function priorityColor(p: string): string {
  if (p === 'critical') return 'red';
  if (p === 'high') return 'yellow';
  if (p === 'medium') return 'blue';
  return 'gray';
}

function queueStatusColor(s: string): string {
  if (s === 'pending') return 'purple';
  if (s === 'running') return 'blue';
  if (s === 'completed') return 'green';
  if (s === 'failed' || s === 'error') return 'red';
  return 'gray';
}

function Pagination({ page, totalPages, onPageChange }: { page: number; totalPages: number; onPageChange: (p: number) => void }) {
  if (totalPages <= 1) return null;
  const btn = "rounded-lg px-2.5 py-1 text-xs text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed bg-white/5 transition-colors";
  return (
    <div className="flex items-center justify-center gap-2">
      <button onClick={() => onPageChange(1)} disabled={page <= 1} className={btn}>First</button>
      <button onClick={() => onPageChange(page - 1)} disabled={page <= 1} className={btn}>Prev</button>
      <span className="text-xs text-gray-400">Page {page} / {totalPages}</span>
      <button onClick={() => onPageChange(page + 1)} disabled={page >= totalPages} className={btn}>Next</button>
      <button onClick={() => onPageChange(totalPages)} disabled={page >= totalPages} className={btn}>Last</button>
    </div>
  );
}

function QueueTab({
  queue,
  counts,
  total,
  totalAll,
  page,
  totalPages,
  statusFilter,
  onStatusFilter,
  search,
  onSearch,
  onPageChange,
  expanded,
  onToggle,
}: {
  queue: QueueItem[];
  counts: Record<string, number>;
  total: number;
  totalAll: number;
  page: number;
  totalPages: number;
  statusFilter: string;
  onStatusFilter: (s: string) => void;
  search: string;
  onSearch: (s: string) => void;
  onPageChange: (p: number) => void;
  expanded: Set<string>;
  onToggle: (id: string) => void;
}) {
  // Group sweep variants under parents (from server-paginated results)
  const parents = new Map<string, QueueItem[]>();
  const standalone: QueueItem[] = [];
  for (const item of queue) {
    if (item.sweep_parent) {
      const group = parents.get(item.sweep_parent) || [];
      group.push(item);
      parents.set(item.sweep_parent, group);
    } else {
      standalone.push(item);
    }
  }

  type GroupedItem = { item: QueueItem; children: QueueItem[] };
  const grouped: GroupedItem[] = [];
  const usedParents = new Set<string>();
  for (const item of standalone) {
    const children = parents.get(item.idea_id) || [];
    usedParents.add(item.idea_id);
    grouped.push({ item, children });
  }
  for (const [parentId, children] of parents) {
    if (!usedParents.has(parentId)) {
      grouped.push({ item: children[0], children: children.slice(1) });
    }
  }

  return (
    <>
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Queue</h2>
        <Badge color="purple">{totalAll} total</Badge>
      </div>

      {/* Summary badges */}
      <div className="flex gap-3 flex-wrap">
        <Badge color="purple">{counts.pending || 0} pending</Badge>
        <Badge color="blue">{counts.running || 0} running</Badge>
        <Badge color="green">{counts.completed || 0} completed</Badge>
        <Badge color="red">{(counts.failed || 0) + (counts.error || 0)} failed</Badge>
      </div>

      {/* Filters */}
      <div className="flex gap-3 items-center flex-wrap">
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-gray-500" />
          <select
            value={statusFilter}
            onChange={(e) => onStatusFilter(e.target.value)}
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-1.5 text-xs text-gray-300 outline-none focus:border-purple-500/50"
          >
            <option value="all">All</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <Search size={14} className="text-gray-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => onSearch(e.target.value)}
            placeholder="Search ideas..."
            className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-1.5 text-xs text-gray-300 placeholder-gray-600 outline-none focus:border-purple-500/50"
          />
        </div>
        <span className="text-[10px] text-gray-500">{total} results</span>
      </div>

      {/* Pagination (top) */}
      <Pagination page={page} totalPages={totalPages} onPageChange={onPageChange} />

      {/* Queue list */}
      <div className="space-y-2">
        {grouped.length === 0 && (
          <Card>
            <p className="text-sm text-gray-500">No ideas match the current filter</p>
          </Card>
        )}
        {grouped.map(({ item, children }) => (
          <QueueCard
            key={item.idea_id}
            item={item}
            children={children}
            expanded={expanded}
            onToggle={onToggle}
          />
        ))}
      </div>

      {/* Pagination (bottom) */}
      <Pagination page={page} totalPages={totalPages} onPageChange={onPageChange} />
    </>
  );
}

function QueueCard({
  item,
  children,
  expanded,
  onToggle,
}: {
  item: QueueItem;
  children: QueueItem[];
  expanded: Set<string>;
  onToggle: (id: string) => void;
}) {
  const isExpanded = expanded.has(item.idea_id);
  const hasSweep = children.length > 0;

  return (
    <div>
      <Card onClick={() => onToggle(item.idea_id)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge color={priorityColor(item.priority)}>{item.priority}</Badge>
            <Badge color={queueStatusColor(item.status)}>{item.status}</Badge>
            <span className="text-sm font-medium">{fmtRunName(item.idea_id, item.title)}</span>
            {hasSweep && (
              <span className="text-[10px] text-gray-500">+{children.length} sweep variants</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {item.category && (
              <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-gray-400">
                {item.category}
              </span>
            )}
            {item.parent && item.parent !== 'none' && (
              <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-gray-400">
                {item.parent}
              </span>
            )}
            {isExpanded ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
          </div>
        </div>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="mt-3 space-y-3 border-t border-white/5 pt-3">
                {item.hypothesis && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider text-gray-500">Hypothesis</span>
                    <p className="text-xs text-gray-400 mt-0.5">{item.hypothesis}</p>
                  </div>
                )}
                {item.huggingface && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider text-gray-500">HuggingFace</span>
                    <div className="mt-1 flex items-center gap-3 rounded-lg bg-white/[0.02] p-2">
                      <a
                        href={item.huggingface.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-purple-400 hover:text-purple-300 font-mono underline underline-offset-2"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {item.huggingface.model_id}
                      </a>
                      <span className="text-[10px] text-gray-500">
                        {item.huggingface.feature_dim}d
                      </span>
                      <span className="text-[10px] text-gray-500">
                        {item.huggingface.img_size}px
                      </span>
                      <Badge color="purple">{item.huggingface.source}</Badge>
                    </div>
                  </div>
                )}
                {Object.keys(item.config).length > 0 && (
                  <div>
                    <span className="text-[10px] uppercase tracking-wider text-gray-500">Config</span>
                    <pre className="mt-1 text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-48 rounded-lg bg-white/[0.02] p-2 font-mono">
                      {JSON.stringify(item.config, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Sweep children */}
      {hasSweep && isExpanded && (
        <div className="ml-6 mt-1 space-y-1">
          {children.map((child) => (
            <Card key={child.idea_id} onClick={() => onToggle(child.idea_id)} className="py-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Badge color={queueStatusColor(child.status)}>{child.status}</Badge>
                  <span className="text-xs font-mono text-gray-400">{child.idea_id.split('~').slice(1).join('~')}</span>
                </div>
                {expanded.has(child.idea_id) ? <ChevronDown size={12} className="text-gray-500" /> : <ChevronRight size={12} className="text-gray-500" />}
              </div>
              <AnimatePresence>
                {expanded.has(child.idea_id) && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <pre className="mt-2 text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-32 rounded-lg bg-white/[0.02] p-2 font-mono">
                      {JSON.stringify(child.config, null, 2)}
                    </pre>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const memPct = Math.round((gpu.memory_used_mib / gpu.memory_total_mib) * 100);
  const tempColor = gpu.temperature_c > 80 ? 'text-red-400' : gpu.temperature_c > 65 ? 'text-amber-400' : 'text-emerald-400';

  return (
    <div className="rounded-lg bg-white/[0.03] p-3 border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-medium text-gray-400">GPU {gpu.index}</span>
        <span className={`text-[10px] font-medium ${tempColor}`}>{gpu.temperature_c}C</span>
      </div>
      <div className="space-y-1.5">
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <span className="text-[9px] text-gray-500">Util</span>
            <span className="text-[9px] text-gray-400">{gpu.utilization_pct}%</span>
          </div>
          <ProgressBar value={gpu.utilization_pct} color={gpu.utilization_pct > 90 ? 'emerald' : gpu.utilization_pct > 50 ? 'blue' : 'gray'} />
        </div>
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <span className="text-[9px] text-gray-500">VRAM</span>
            <span className="text-[9px] text-gray-400">{memPct}%</span>
          </div>
          <ProgressBar value={memPct} color={memPct > 90 ? 'amber' : 'purple'} />
        </div>
      </div>
      <p className="mt-1.5 text-[9px] text-gray-600 truncate">{gpu.name}</p>
    </div>
  );
}

function AlertCard({ alert }: { alert: Alert }) {
  const iconMap: Record<string, any> = {
    failure: XCircle,
    stale_host: Clock,
    low_disk: HardDrive,
  };
  const colorMap: Record<string, string> = {
    failure: 'red',
    stale_host: 'yellow',
    low_disk: 'yellow',
  };
  const Icon = iconMap[alert.type] || AlertTriangle;
  const color = colorMap[alert.type] || 'gray';

  return (
    <Card>
      <div className="flex items-start gap-3">
        <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-${color}-500/10`}>
          <Icon size={16} className={`text-${color}-400`} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <Badge color={color}>{alert.type.replace('_', ' ')}</Badge>
            {alert.minutes_ago != null && (
              <span className="text-[10px] text-gray-500">{ago(alert.minutes_ago)}</span>
            )}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            {alert.type === 'failure' && (
              <>
                <span className="font-medium text-gray-300">{alert.idea_id}</span>
                {alert.error && <span className="ml-2">{alert.error}</span>}
              </>
            )}
            {alert.type === 'stale_host' && (
              <span>
                Host <span className="font-medium text-gray-300">{alert.host}</span> not responding
              </span>
            )}
            {alert.type === 'low_disk' && (
              <span>
                Only <span className="font-medium text-amber-300">{alert.disk_free_gb} GB</span> free
              </span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
