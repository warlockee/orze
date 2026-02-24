import { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, Search, Filter } from 'lucide-react';
import { useQueue } from './hooks';
import { Badge, Card, Pagination, LoadingState, fmtRunName, priorityColor, queueStatusColor } from './components';
import type { QueueItem } from './types';

export default function QueueTabView() {
  const [statusFilter, setStatusFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [searchDebounced, setSearchDebounced] = useState('');
  const [page, setPage] = useState(1);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const queueData = useQueue(page, statusFilter, searchDebounced);

  useEffect(() => {
    const t = setTimeout(() => { setSearchDebounced(search); setPage(1); }, 300);
    return () => clearTimeout(t);
  }, [search]);

  if (queueData._loading) return <LoadingState label="Loading queue…" />;

  const onToggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const parents = new Map<string, QueueItem[]>();
  const standalone: QueueItem[] = [];
  for (const item of queueData.queue) {
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
    if (!usedParents.has(parentId)) grouped.push({ item: children[0], children: children.slice(1) });
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Queue</h2>
        <Badge color="purple">{queueData.total_all} total</Badge>
      </div>
      <div className="flex gap-3 flex-wrap">
        <Badge color="purple">{queueData.counts.pending || 0} pending</Badge>
        <Badge color="blue">{queueData.counts.running || 0} running</Badge>
        <Badge color="green">{queueData.counts.completed || 0} completed</Badge>
        <Badge color="red">{(queueData.counts.failed || 0) + (queueData.counts.error || 0)} failed</Badge>
      </div>
      <div className="flex gap-3 items-center flex-wrap">
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-gray-500" />
          <select value={statusFilter} onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-1.5 text-xs text-gray-300 outline-none focus:border-purple-500/50">
            <option value="all">All</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        <div className="flex items-center gap-2 flex-1 max-w-xs">
          <Search size={14} className="text-gray-500" />
          <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search ideas..."
            className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-1.5 text-xs text-gray-300 placeholder-gray-600 outline-none focus:border-purple-500/50" />
        </div>
        <span className="text-[10px] text-gray-500">{queueData.total} results</span>
      </div>
      <Pagination page={queueData.page} totalPages={queueData.total_pages} onPageChange={setPage} />
      <div className="space-y-2">
        {grouped.length === 0 && <Card><p className="text-sm text-gray-500">No ideas match the current filter</p></Card>}
        {grouped.map(({ item, children }) => (
          <QueueCard key={item.idea_id} item={item} children={children} expanded={expanded} onToggle={onToggle} />
        ))}
      </div>
      <Pagination page={queueData.page} totalPages={queueData.total_pages} onPageChange={setPage} />
    </div>
  );
}

function QueueCard({ item, children, expanded, onToggle }: { item: QueueItem; children: QueueItem[]; expanded: Set<string>; onToggle: (id: string) => void }) {
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
            {hasSweep && <span className="text-[10px] text-gray-500">+{children.length} sweep variants</span>}
          </div>
          <div className="flex items-center gap-2">
            {item.category && <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-gray-400">{item.category}</span>}
            {item.parent && item.parent !== 'none' && <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-gray-400">{item.parent}</span>}
            {isExpanded ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
          </div>
        </div>
        <div className={`reveal-grid ${isExpanded ? 'open' : ''}`} onClick={(e) => e.stopPropagation()}>
          <div>
            <div className="mt-3 space-y-3 border-t border-white/5 pt-3">
              {item.hypothesis && <div><span className="text-[10px] uppercase tracking-wider text-gray-500">Hypothesis</span><p className="text-xs text-gray-400 mt-0.5">{item.hypothesis}</p></div>}
              {item.huggingface && (
                <div>
                  <span className="text-[10px] uppercase tracking-wider text-gray-500">HuggingFace</span>
                  <div className="mt-1 flex items-center gap-3 rounded-lg bg-white/[0.02] p-2">
                    <a href={item.huggingface.url} target="_blank" rel="noopener noreferrer" className="text-xs text-purple-400 hover:text-purple-300 font-mono underline underline-offset-2" onClick={(e) => e.stopPropagation()}>{item.huggingface.model_id}</a>
                    <span className="text-[10px] text-gray-500">{item.huggingface.feature_dim}d</span>
                    <span className="text-[10px] text-gray-500">{item.huggingface.img_size}px</span>
                    <Badge color="purple">{item.huggingface.source}</Badge>
                  </div>
                </div>
              )}
              {Object.keys(item.config).length > 0 && (
                <div>
                  <span className="text-[10px] uppercase tracking-wider text-gray-500">Config</span>
                  <pre className="mt-1 text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-48 rounded-lg bg-white/[0.02] p-2 font-mono">{JSON.stringify(item.config, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
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
              <div className={`reveal-grid ${expanded.has(child.idea_id) ? 'open' : ''}`} onClick={(e) => e.stopPropagation()}>
                <div>
                  <pre className="mt-2 text-[10px] text-gray-400 whitespace-pre-wrap overflow-auto max-h-32 rounded-lg bg-white/[0.02] p-2 font-mono">{JSON.stringify(child.config, null, 2)}</pre>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
