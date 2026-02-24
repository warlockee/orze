import { useState, useEffect, useCallback } from 'react';
import { Play, XCircle, FileText } from 'lucide-react';
import { useRuns } from './hooks';
import { fetchRunDetail, fetchRunLog, killRun } from './api';
import { Badge, Card, Pill, Reveal, statusColor, fmtRunName, fmtTime, fmtMetric } from './components';
import type { Run, ActiveRun, RunDetail } from './types';

export default function RunsTab() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runLog, setRunLog] = useState<string>('');

  const activeRuns = runs.active;
  const completedRuns = runs.recent;
  const allRuns: Run[] = [
    ...activeRuns.map((r: ActiveRun) => ({
      idea_id: r.idea_id, status: 'RUNNING' as const, host: r.host, gpu: r.gpu, elapsed_min: r.elapsed_min, title: r.title,
    })),
    ...completedRuns,
  ];

  useEffect(() => {
    if (!selectedRun) { setRunDetail(null); setRunLog(''); return; }
    fetchRunDetail(selectedRun).then(setRunDetail).catch(() => setRunDetail(null));
    fetchRunLog(selectedRun).then((d) => setRunLog(d.log || '')).catch(() => setRunLog(''));
  }, [selectedRun]);

  const handleKillRun = useCallback(async (id: string) => {
    if (!confirm(`Kill run ${id}?`)) return;
    await killRun(id);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Runs</h2>
        <div className="flex gap-2">
          <Badge color="blue">{activeRuns.length} running</Badge>
          <Badge color="green">{completedRuns.filter((r) => r.status === 'COMPLETED').length} completed</Badge>
          <Badge color="red">{completedRuns.filter((r) => r.status === 'FAILED' || r.status === 'ERROR').length} failed</Badge>
        </div>
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-2">
          {allRuns.length === 0 && <Card><p className="text-sm text-gray-500">No runs yet</p></Card>}
          {allRuns.map((r) => (
            <Card key={r.idea_id} onClick={() => setSelectedRun(r.idea_id)} className={selectedRun === r.idea_id ? 'border-purple-500/30' : ''}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Badge color={statusColor(r.status)}>{r.status}</Badge>
                  <span className="text-sm font-medium">{fmtRunName(r.idea_id, r.title)}</span>
                </div>
                <div className="flex items-center gap-3">
                  {r.host && <span className="text-[10px] text-gray-500">{r.host}</span>}
                  {r.gpu != null && <span className="text-[10px] text-gray-500">GPU {r.gpu}</span>}
                  <span className="text-[10px] text-gray-500">{fmtTime(r.elapsed_min)}</span>
                  {r.metric != null && <span className="text-xs font-mono text-emerald-400">{fmtMetric(r.metric)}</span>}
                  {r.status === 'RUNNING' && (
                    <button onClick={(e) => { e.stopPropagation(); handleKillRun(r.idea_id); }}
                      className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-red-400 hover:bg-red-500/10 transition-colors">
                      <XCircle size={10} /> Kill
                    </button>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
        <div className="space-y-4">
          {selectedRun ? (
            <>
              <Card>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold">{selectedRun}</h3>
                  <button onClick={() => setSelectedRun(null)} className="text-gray-500 hover:text-white"><XCircle size={14} /></button>
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
                        <pre className="text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-48">{JSON.stringify(runDetail.metrics, null, 2)}</pre>
                      </Reveal>
                    )}
                    <button onClick={() => handleKillRun(selectedRun)}
                      className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/20 transition-colors border border-red-500/20 mt-2">
                      <XCircle size={12} /> Kill Run
                    </button>
                  </div>
                ) : <p className="text-xs text-gray-500">Loading...</p>}
              </Card>
              {runLog && (
                <Card>
                  <h3 className="mb-2 text-sm font-bold flex items-center gap-2"><FileText size={14} /> Log</h3>
                  <pre className="text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-96 font-mono">{runLog}</pre>
                </Card>
              )}
            </>
          ) : <Card><p className="text-xs text-gray-500">Select a run to view details</p></Card>}
        </div>
      </div>
    </div>
  );
}
