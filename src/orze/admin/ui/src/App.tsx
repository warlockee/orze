import { useState, useCallback, lazy, Suspense } from 'react';
import {
  Activity,
  AlertTriangle,
  Award,
  Bell,
  ListOrdered,
  Play,
  Server,
  Settings,
  Square,
  Zap,
} from 'lucide-react';

import { useAlerts } from './hooks';
import { stopAll } from './api';
import { Segmented, GlowBlob, FloatingGrid } from './components';
import type { Tab } from './types';

const OverviewTab = lazy(() => import('./OverviewTab'));
const FleetTab = lazy(() => import('./FleetTab'));
const RunsTab = lazy(() => import('./RunsTab'));
const QueueTabView = lazy(() => import('./QueueTabView'));
const LeaderboardTab = lazy(() => import('./LeaderboardTab'));
const AlertsTab = lazy(() => import('./AlertsTab'));
const SettingsTab = lazy(() => import('./SettingsTab'));

function TabFallback() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-600 border-t-purple-400" />
    </div>
  );
}

export default function OrzeAdminPanel() {
  const [tab, setTab] = useState<Tab>('overview');
  const [confirmStopAll, setConfirmStopAll] = useState(false);

  const alertsData = useAlerts();
  const alerts = alertsData.alerts;

  const handleStopAll = useCallback(async () => {
    await stopAll();
    setConfirmStopAll(false);
  }, []);

  const tabs: { key: Tab; label: string; icon: any }[] = [
    { key: 'overview', label: 'Overview', icon: Activity },
    { key: 'fleet', label: 'Nodes', icon: Server },
    { key: 'runs', label: 'Runs', icon: Play },
    { key: 'queue', label: 'Queue', icon: ListOrdered },
    { key: 'leaderboard', label: 'Leaderboard', icon: Award },
    { key: 'alerts', label: 'Alerts', icon: Bell },
    { key: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="relative min-h-screen bg-gray-950 text-gray-100">
      <GlowBlob />
      <FloatingGrid />

      <header className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-cyan-500">
              <Zap size={16} className="text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight">ORZE</h1>
              <p className="text-[10px] text-gray-500">Admin Panel</p>
            </div>
          </div>

          <Segmented
            items={tabs.map((t) => ({ key: t.key, label: t.label, icon: t.icon }))}
            value={tab}
            onChange={(k) => setTab(k as Tab)}
          />

          <div className="flex items-center gap-3">
            {alerts.length > 0 && (
              <button
                onClick={() => setTab('alerts')}
                className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 hover:bg-white/10"
              >
                <Bell size={14} />
                <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[9px] font-bold">
                  {alerts.length}
                </span>
              </button>
            )}

            <button
              onClick={() => setConfirmStopAll(true)}
              className="flex items-center gap-1.5 rounded-lg bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/20 transition-colors border border-red-500/20"
            >
              <Square size={12} />
              Stop All
            </button>
          </div>
        </div>
      </header>

      {/* stop-all modal */}
      {confirmStopAll && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm modal-overlay"
          onClick={() => setConfirmStopAll(false)}
        >
          <div
            className="glass rounded-2xl p-6 max-w-md mx-4 modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/20">
                <AlertTriangle size={20} className="text-red-400" />
              </div>
              <div>
                <h3 className="font-bold text-white">Stop All Runs?</h3>
                <p className="text-xs text-gray-400">This will write a stop sentinel. All farm instances will halt.</p>
              </div>
            </div>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setConfirmStopAll(false)}
                className="rounded-lg px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleStopAll}
                className="rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 transition-colors"
              >
                Confirm Stop
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="mx-auto max-w-7xl px-6 py-6">
        <Suspense fallback={<TabFallback />}>
          {tab === 'overview' && <OverviewTab />}
          {tab === 'fleet' && <FleetTab />}
          {tab === 'runs' && <RunsTab />}
          {tab === 'queue' && <QueueTabView />}
          {tab === 'leaderboard' && <LeaderboardTab />}
          {tab === 'alerts' && <AlertsTab />}
          {tab === 'settings' && <SettingsTab />}
        </Suspense>
      </main>
    </div>
  );
}
