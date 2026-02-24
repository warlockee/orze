import { useConfig } from './hooks';
import { Card } from './components';

export default function SettingsTab() {
  const config = useConfig();
  return (
    <div className="space-y-6">
      <h2 className="text-lg font-bold">Configuration</h2>
      <Card>
        <pre className="text-xs text-gray-400 whitespace-pre-wrap overflow-auto max-h-[70vh] font-mono">
          {Object.keys(config).length > 0 ? JSON.stringify(config, null, 2) : 'Loading configuration...'}
        </pre>
      </Card>
    </div>
  );
}
