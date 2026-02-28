import { Monitor } from 'lucide-react';
import { useNodes } from './hooks';
import { Badge, Card, Pill, Reveal, GpuCard, LoadingState, statusColor, fmtRunName, fmtTime } from './components';

export default function NodesTab() {
  const nodes = useNodes();
  if (nodes._loading) return <LoadingState label="Loading nodes…" />;
  const hosts = nodes.heartbeats;
  const localGpus = nodes.local_gpus;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-bold">Nodes</h2>
      {hosts.length === 0 && (
        <Card><p className="text-sm text-gray-500">No hosts reporting. Waiting for heartbeats...</p></Card>
      )}
      {hosts.map((h) => (
        <Card key={h.host}>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Monitor size={16} className="text-gray-400" />
              <span className="font-bold text-sm">{h.host}</span>
              <Badge color={statusColor(h.status)}>{h.status}</Badge>
            </div>
            <span className="text-[10px] text-gray-500">heartbeat {Math.round(h.heartbeat_age_sec)}s ago</span>
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
          {h.gpu_info && h.gpu_info.length > 0 && (
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {h.gpu_info.map((g) => (<GpuCard key={g.index} gpu={g} />))}
            </div>
          )}
        </Card>
      ))}
      {localGpus.length > 0 && (
        <Card>
          <h3 className="mb-3 text-sm font-bold">Local GPUs (this host)</h3>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {localGpus.map((g) => (<GpuCard key={g.index} gpu={g} />))}
          </div>
        </Card>
      )}
    </div>
  );
}
