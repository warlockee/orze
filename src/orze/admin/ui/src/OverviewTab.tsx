import { useRef, useEffect } from 'react';
import { Cpu, HardDrive, Thermometer, Layers, TrendingUp, AlertTriangle, Play } from 'lucide-react';
import { useStatus, useNodes, useRuns } from './hooks';
import { Badge, Card, IconKpi, LoadingState, statusColor, fmtRunName, fmtTime } from './components';

export default function OverviewTab() {
  const status = useStatus();
  const nodes = useNodes();
  const runs = useRuns();

  const gpuSparkRef = useRef<number[]>([]);
  const queueSparkRef = useRef<number[]>([]);

  const localGpus = nodes.local_gpus;

  useEffect(() => {
    if (!localGpus.length) return;
    const avg = localGpus.reduce((s, g) => s + g.utilization_pct, 0) / localGpus.length;
    gpuSparkRef.current = [...gpuSparkRef.current.slice(-19), avg];
  }, [nodes]);

  useEffect(() => {
    if (status.queue_depth != null) {
      queueSparkRef.current = [...queueSparkRef.current.slice(-19), status.queue_depth];
    }
  }, [status]);

  if (status._loading && nodes._loading) return <LoadingState label="Loading overview…" />;

  const hosts = nodes.heartbeats;
  const gpuAvg = localGpus.length > 0
    ? Math.round(localGpus.reduce((s, g) => s + g.utilization_pct, 0) / localGpus.length)
    : 0;
  const memAvg = localGpus.length > 0
    ? Math.round(localGpus.reduce((s, g) => s + (g.memory_used_mib / g.memory_total_mib) * 100, 0) / localGpus.length)
    : 0;
  const tempMax = localGpus.length > 0 ? Math.max(...localGpus.map((g) => g.temperature_c)) : 0;
  const activeRuns = runs.active;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <IconKpi icon={Cpu} label="GPU Util" value={`${gpuAvg}%`} spark={gpuSparkRef.current} />
        <IconKpi icon={HardDrive} label="VRAM" value={`${memAvg}%`} />
        <IconKpi icon={Thermometer} label="Temp Peak" value={`${tempMax}C`} />
        <IconKpi icon={Layers} label="Queue" value={status.queue_depth} spark={queueSparkRef.current} />
        <IconKpi icon={TrendingUp} label="Completed" value={status.completed} />
        <IconKpi icon={AlertTriangle} label="Failed" value={status.failed} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold">Nodes</h2>
            <Badge color={hosts.length > 0 ? 'green' : 'gray'}>{hosts.length} hosts</Badge>
          </div>
          <div className="space-y-2">
            {hosts.length === 0 && <p className="text-xs text-gray-500">No heartbeats received yet</p>}
            {hosts.map((h) => (
              <div key={h.host} className="flex items-center justify-between rounded-lg bg-white/[0.02] px-3 py-2">
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${
                    h.status === 'online' ? 'bg-emerald-400' : h.status === 'degraded' ? 'bg-amber-400' : 'bg-red-400'
                  }`} />
                  <span className="text-xs font-medium">{h.host}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] text-gray-500">{h.free_gpus.length} free GPU{h.free_gpus.length !== 1 ? 's' : ''}</span>
                  <Badge color={statusColor(h.status)}>{h.status}</Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold">Active Runs</h2>
            <Badge color="blue">{activeRuns.length} running</Badge>
          </div>
          <div className="space-y-2">
            {activeRuns.length === 0 && <p className="text-xs text-gray-500">No active runs</p>}
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

      <Card className="max-w-xs">
        <div className="flex items-center gap-2 mb-2">
          <HardDrive size={14} className="text-gray-400" />
          <span className="text-xs font-medium">Disk</span>
        </div>
        <span className="text-lg font-bold">{status.disk_free_gb.toFixed(0)} GB</span>
        <span className="text-[10px] text-gray-500 ml-1">free</span>
      </Card>
    </div>
  );
}
