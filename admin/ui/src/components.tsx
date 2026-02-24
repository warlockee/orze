import { useState, type ReactNode } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Clock,
  HardDrive,
  XCircle,
} from 'lucide-react';
import type { GpuInfo, Alert } from './types';

/* ─── tiny primitives ─── */

export function Badge({ children, color = 'gray' }: { children: ReactNode; color?: string }) {
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
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${colors[color] ?? colors.gray}`}>
      {children}
    </span>
  );
}

export function Pill({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-gray-500">{label}</span>
      <span className="text-sm font-semibold text-gray-100">{value}</span>
      {sub && <span className="text-[10px] text-gray-500">{sub}</span>}
    </div>
  );
}

export function Card({
  children,
  className = '',
  onClick,
}: {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={`glass rounded-xl p-4 card-animate ${onClick ? 'cursor-pointer hover:border-white/10' : ''} ${className}`}
    >
      {children}
    </div>
  );
}

const barColors: Record<string, string> = {
  emerald: 'bg-emerald-500',
  blue: 'bg-blue-500',
  purple: 'bg-purple-500',
  amber: 'bg-amber-500',
  gray: 'bg-gray-500',
  red: 'bg-red-500',
};

export function ProgressBar({ value, max = 100, color = 'emerald' }: { value: number; max?: number; color?: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="h-1.5 w-full rounded-full bg-white/5">
      <div
        className={`h-full rounded-full progress-fill ${barColors[color] ?? barColors.emerald}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function MiniSpark({ data, color = '#34d399', w = 60, h = 20 }: { data: number[]; color?: string; w?: number; h?: number }) {
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

export function IconKpi({
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

export function Segmented({
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

export function Reveal({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
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
      <div className={`reveal-grid ${open ? 'open' : ''}`}>
        <div>
          <div className="px-4 pb-4">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

export function GlowBlob() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute -left-40 -top-40 h-[600px] w-[600px] rounded-full bg-purple-600/10 blur-[160px]" />
      <div className="absolute -bottom-40 -right-40 h-[500px] w-[500px] rounded-full bg-cyan-600/10 blur-[140px]" />
    </div>
  );
}

export function FloatingGrid() {
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

export function Table({
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

export function Pagination({ page, totalPages, onPageChange }: { page: number; totalPages: number; onPageChange: (p: number) => void }) {
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

/* ─── sub-components ─── */

export function GpuCard({ gpu }: { gpu: GpuInfo }) {
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

const alertBgColors: Record<string, string> = {
  red: 'bg-red-500/10',
  yellow: 'bg-yellow-500/10',
  gray: 'bg-gray-500/10',
};

const alertTextColors: Record<string, string> = {
  red: 'text-red-400',
  yellow: 'text-yellow-400',
  gray: 'text-gray-400',
};

export function AlertCard({ alert }: { alert: Alert }) {
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
        <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${alertBgColors[color] ?? alertBgColors.gray}`}>
          <Icon size={16} className={alertTextColors[color] ?? alertTextColors.gray} />
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
              <span>Host <span className="font-medium text-gray-300">{alert.host}</span> not responding</span>
            )}
            {alert.type === 'low_disk' && (
              <span>Only <span className="font-medium text-amber-300">{alert.disk_free_gb} GB</span> free</span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ─── helpers ─── */

export function statusColor(s: string): string {
  if (s === 'online') return 'green';
  if (s === 'degraded') return 'yellow';
  if (s === 'offline') return 'red';
  if (s === 'RUNNING') return 'blue';
  if (s === 'QUEUED') return 'purple';
  if (s === 'COMPLETED') return 'green';
  if (s === 'FAILED' || s === 'ERROR') return 'red';
  return 'gray';
}

export function fmtTime(mins: number | undefined): string {
  if (mins == null) return '-';
  if (mins < 60) return `${Math.round(mins)}m`;
  return `${(mins / 60).toFixed(1)}h`;
}

export function fmtMetric(v: number | undefined): string {
  if (v == null) return '-';
  return v.toFixed(4);
}

export function ago(mins: number | undefined): string {
  if (mins == null) return '';
  if (mins < 1) return 'just now';
  if (mins < 60) return `${Math.round(mins)}m ago`;
  return `${(mins / 60).toFixed(1)}h ago`;
}

export function fmtRunName(ideaId: string, title?: string): string {
  const base = ideaId.split('~')[0];
  if (title) return `${base} · ${title}`;
  return base;
}

export function priorityColor(p: string): string {
  if (p === 'critical') return 'red';
  if (p === 'high') return 'yellow';
  if (p === 'medium') return 'blue';
  return 'gray';
}

export function queueStatusColor(s: string): string {
  if (s === 'pending') return 'purple';
  if (s === 'running') return 'blue';
  if (s === 'completed') return 'green';
  if (s === 'failed' || s === 'error') return 'red';
  return 'gray';
}
