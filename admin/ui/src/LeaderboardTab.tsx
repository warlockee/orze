import { useLeaderboard } from './hooks';
import { Badge, Card, Table, LoadingState, fmtTime } from './components';

export default function LeaderboardTab() {
  const leaderboard = useLeaderboard();
  if (leaderboard._loading) return <LoadingState label="Loading leaderboard…" />;
  const lbEntries = leaderboard.top;

  return (
    <div className="space-y-6">
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
            rank: <span className={i < 3 ? 'text-amber-400 font-bold' : ''}>{i + 1}</span>,
            idea_id: <span className="font-mono text-xs">{e.idea_id}</span>,
            title: <span className="text-xs text-gray-400 truncate block max-w-[300px]">{e.title || '-'}</span>,
            metric: (
              <span className="font-mono text-emerald-400">
                {typeof e.metric_value === 'number' ? e.metric_value.toFixed(4)
                  : typeof e.auc_roc === 'number' ? e.auc_roc.toFixed(4) : '-'}
              </span>
            ),
            time: <span className="text-xs text-gray-500">{e.training_time ? fmtTime(e.training_time / 60) : '-'}</span>,
          }))}
        />
      </Card>
    </div>
  );
}
