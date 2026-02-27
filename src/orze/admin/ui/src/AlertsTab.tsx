import { Shield } from 'lucide-react';
import { useAlerts } from './hooks';
import { Badge, Card, AlertCard, LoadingState } from './components';

export default function AlertsTab() {
  const alertsData = useAlerts();
  if (alertsData._loading) return <LoadingState label="Loading alerts…" />;
  const alerts = alertsData.alerts;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Alerts</h2>
        <Badge color={alerts.length > 0 ? 'red' : 'green'}>
          {alerts.length === 0 ? 'All clear' : `${alerts.length} alert${alerts.length !== 1 ? 's' : ''}`}
        </Badge>
      </div>
      {alerts.length === 0 && (
        <Card>
          <div className="flex items-center gap-3 py-4">
            <Shield size={20} className="text-emerald-400" />
            <span className="text-sm text-gray-400">No active alerts</span>
          </div>
        </Card>
      )}
      <div className="space-y-3">
        {alerts.map((a, i) => (<AlertCard key={i} alert={a} />))}
      </div>
    </div>
  );
}
