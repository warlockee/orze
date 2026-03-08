import { useState } from 'react';
import { useLeaderboard, useLeaderboardViews } from './hooks';
import { Badge, Card, Table, LoadingState, fmtTime, IdeaLink } from './components';

export default function LeaderboardTab() {
  const [activeView, setActiveView] = useState('');
  const views = useLeaderboardViews();
  const leaderboard = useLeaderboard(activeView || undefined);
  if (leaderboard._loading) return <LoadingState label="Loading leaderboard…" />;
  const lbEntries = leaderboard.top;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">{leaderboard.title || 'Leaderboard'}</h2>
        <div className="flex items-center gap-3">
          {views.views.length > 0 && (
            <div className="flex gap-1">
              <button
                onClick={() => setActiveView('')}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  !activeView
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                All
              </button>
              {views.views.map((v) => (
                <button
                  key={v}
                  onClick={() => setActiveView(v)}
                  className={`px-3 py-1 rounded text-xs font-medium capitalize transition-colors ${
                    activeView === v
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          )}
          <Badge color="purple">{lbEntries.length} entries</Badge>
        </div>
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
            rank: <span className={i < 3 ? 'text-amber-400 font-bold' : ''}>{i + 1}</span>,
            idea_id: <IdeaLink ideaId={e.idea_id} />,
            title: <span className="text-xs text-gray-400 truncate block max-w-[300px]">{e.title || '-'}</span>,
            metric: (
              <span className="font-mono text-emerald-400">
                {typeof e.metric_value === 'number' ? e.metric_value.toFixed(4) : '-'}
              </span>
            ),
            time: <span className="text-xs text-gray-500">{e.training_time ? fmtTime(e.training_time / 60) : '-'}</span>,
          }))}
        />
      </Card>
    </div>
  );
}
