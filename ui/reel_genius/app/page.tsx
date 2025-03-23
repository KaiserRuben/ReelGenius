import { getPlatforms, PlatformConfig } from '@/lib/api';
import CreateVideoForm from '@/components/CreateVideoForm';

export default async function Home() {
  // Server-side data fetching
  const platformData = await getPlatforms().catch(() => null);
  
  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">🎬 AI Video Generator</h1>
        <p className="text-muted-foreground">Transform content into engaging videos with AI</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <CreateVideoForm />
        </div>
        
        <div>
          <div className="card p-4">
            <h2 className="text-lg font-semibold mb-3">Quick Tips</h2>
            <ul className="space-y-2 text-sm">
              <li className="flex gap-2">
                <span className="text-primary">🔍</span>
                <span>More detailed content produces better videos</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary">⏱️</span>
                <span>Generation typically takes 2-5 minutes</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary">💡</span>
                <span>Choose the right platform for optimal formatting</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary">🎨</span>
                <span>Adjust visual settings for different styles</span>
              </li>
            </ul>
          </div>
          
          {platformData && (
            <div className="card p-4 mt-4">
              <h2 className="text-lg font-semibold mb-3">Platform Information</h2>
              <div className="space-y-3">
                {Object.entries(platformData.configs || {}).map(([platform, config]: [string, PlatformConfig]) => (
                  <div key={platform} className="text-sm">
                    <h3 className="font-medium uppercase">{platform.replace('_', ' ')}</h3>
                    <div className="grid grid-cols-2 gap-1 mt-1 text-xs text-muted-foreground">
                      <span>Aspect Ratio:</span>
                      <span>{config.aspect_ratio}</span>
                      <span>Duration:</span>
                      <span>{config.min_duration}s - {config.max_duration}s</span>
                      <span>Resolution:</span>
                      <span>{config.resolution}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}