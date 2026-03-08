const BASE = '';

async function request(url: string, opts?: RequestInit): Promise<any> {
  const res = await fetch(`${BASE}${url}`, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function stopAll(): Promise<void> {
  await fetch('/api/actions/stop', { method: 'POST' });
}

export async function killRun(ideaId: string): Promise<void> {
  await fetch('/api/actions/kill', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idea_id: ideaId }),
  });
}

export async function addIdea(
  title: string,
  config: Record<string, any>,
  priority: string,
): Promise<any> {
  return request('/api/ideas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, config, priority }),
  });
}

export async function fetchRunDetail(ideaId: string): Promise<any> {
  return request(`/api/run/detail?idea_id=${encodeURIComponent(ideaId)}`);
}

export async function fetchRunLog(ideaId: string): Promise<any> {
  return request(`/api/run/log?idea_id=${encodeURIComponent(ideaId)}`);
}

export async function fetchIdeaDetail(ideaId: string): Promise<any> {
  // Try the dedicated endpoint first, fall back to composing from existing endpoints
  try {
    const res = await fetch(`/api/idea/detail?idea_id=${encodeURIComponent(ideaId)}`);
    if (res.ok) {
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) return res.json();
    }
  } catch { /* endpoint may not exist yet */ }

  // Fallback: compose from /api/queue + /api/run/detail
  const result: any = { idea_id: ideaId, found: false };

  // Try queue data (has hypothesis, category, parent, config)
  try {
    const q = await request(`/api/queue?search=${encodeURIComponent(ideaId.split('~')[0])}&per_page=100`);
    const match = (q.queue || []).find((item: any) => item.idea_id === ideaId || item.idea_id === ideaId.split('~')[0]);
    if (match) {
      result.found = true;
      result.title = match.title;
      result.priority = match.priority;
      result.category = match.category;
      result.parent = match.parent;
      result.hypothesis = match.hypothesis;
      result.config = match.config;
      result.origin = 'ideas.md';
    }
  } catch { /* ignore */ }

  // Try run detail (has metrics, claim)
  try {
    const rd = await request(`/api/run/detail?idea_id=${encodeURIComponent(ideaId)}`);
    if (rd.metrics) { result.found = true; result.metrics = rd.metrics; }
    if (rd.claim) result.claim = rd.claim;
  } catch { /* ignore */ }

  return result;
}
