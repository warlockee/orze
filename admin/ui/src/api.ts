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
