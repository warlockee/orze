import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { useIdeaDetail } from './IdeaDetailContext';
import { fetchIdeaDetail } from './api';
import { Badge, Pill, Reveal, priorityColor } from './components';
import type { IdeaDetail } from './types';

export default function IdeaDetailModal() {
  const { activeIdeaId, closeIdea } = useIdeaDetail();
  const [detail, setDetail] = useState<IdeaDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!activeIdeaId) { setDetail(null); return; }
    setLoading(true);
    fetchIdeaDetail(activeIdeaId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [activeIdeaId]);

  useEffect(() => {
    if (!activeIdeaId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeIdea(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [activeIdeaId, closeIdea]);

  if (!activeIdeaId) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm modal-overlay"
      onClick={closeIdea}
    >
      <div
        className="glass rounded-2xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto modal-content"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-mono text-sm font-bold text-white">{activeIdeaId}</h3>
            {detail?.title && <p className="text-xs text-gray-400 mt-0.5">{detail.title}</p>}
          </div>
          <button onClick={closeIdea} className="text-gray-500 hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-8">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-700 border-t-purple-400" />
          </div>
        )}

        {detail && !loading && (
          <div className="space-y-4">
            {/* Badges */}
            <div className="flex flex-wrap gap-2">
              {detail.priority && <Badge color={priorityColor(detail.priority)}>{detail.priority}</Badge>}
              {detail.category && <Badge color="cyan">{detail.category}</Badge>}
              {detail.origin && (
                <Badge color="gray">{detail.origin === 'ideas.md' ? 'active' : 'archived'}</Badge>
              )}
            </div>

            {/* Parent + research cycle */}
            <div className="grid grid-cols-2 gap-3">
              <Pill
                label="Parent"
                value={detail.parent && detail.parent !== 'none' ? detail.parent : 'Original idea'}
              />
              <Pill
                label="Research Cycle"
                value={detail.research_cycle || '-'}
              />
            </div>

            {/* Hypothesis */}
            {detail.hypothesis && (
              <div>
                <span className="text-[10px] uppercase tracking-wider text-gray-500">Hypothesis</span>
                <p className="text-xs text-gray-300 mt-1 leading-relaxed">{detail.hypothesis}</p>
              </div>
            )}

            {/* Config overrides */}
            {detail.config && Object.keys(detail.config).length > 0 && (
              <Reveal title="Config Overrides" defaultOpen={false}>
                <pre className="text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-48 font-mono">
                  {JSON.stringify(detail.config, null, 2)}
                </pre>
              </Reveal>
            )}

            {/* Metrics */}
            {detail.metrics && (
              <Reveal title="Evaluation Metrics" defaultOpen>
                {detail.metrics.test_metrics && (
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    {Object.entries(detail.metrics.test_metrics).map(([k, v]) => (
                      <Pill
                        key={k}
                        label={k}
                        value={typeof v === 'number' ? (v as number).toFixed(4) : String(v)}
                      />
                    ))}
                  </div>
                )}
                {detail.metrics.training_time != null && (
                  <Pill
                    label="Training Time"
                    value={`${(detail.metrics.training_time / 60).toFixed(1)} min`}
                  />
                )}
                {detail.metrics.status === 'FAILED' && (
                  <div className="text-xs text-red-400 mt-1">
                    Failed: {detail.metrics.reason || detail.metrics.error || 'unknown'}
                  </div>
                )}
              </Reveal>
            )}

            {/* Not found */}
            {!detail.found && (
              <div className="text-xs text-gray-500 text-center py-4">
                No metadata found for this idea. It may have been pruned.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
