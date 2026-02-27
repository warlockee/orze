import { useConfig } from './hooks';
import { Card, LoadingState } from './components';

export default function SettingsTab() {
  const config = useConfig();
  if (config._loading) return <LoadingState label="Loading config…" />;
  return (
    <div className="space-y-6">
      <h2 className="text-lg font-bold">Configuration</h2>
      <Card>
        <pre className="text-xs text-gray-400 whitespace-pre-wrap overflow-auto max-h-[70vh] font-mono">
          {JSON.stringify(config, null, 2)}
        </pre>
      </Card>
    </div>
  );
}
