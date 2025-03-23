import { getTasks, TaskStatusResponse } from '@/lib/api';
import Link from 'next/link';

export const dynamic = 'force-dynamic';

export default async function AnalyticsPage() {
  // Server-side data fetching
  const tasksData = await getTasks(
    undefined, // status
    undefined, // platform
    100, // limit
    0, // skip
    true, // include_details
    true // include_media_info
  ).catch((error) => {
    console.error('Error fetching tasks:', error);
    return { tasks: [], total: 0, limit: 100, skip: 0, has_more: false };
  });
  
  // Calculate stats from tasks
  const stats = {
    total: tasksData.total,
    completed: 0,
    failed: 0,
    running: 0,
    queued: 0,
    totalDuration: 0,
    avgDuration: 0,
    platformStats: {} as Record<string, number>,
    totalVideoSize: 0,
    totalImageSize: 0,
    totalAudioSize: 0,
    completionRate: 0,
    averageExecutionTime: 0,
    executionTimes: [] as number[],
    creationDates: {} as Record<string, number>
  };
  
  tasksData.tasks.forEach((task: TaskStatusResponse) => {
    // Status counts
    if (task.status === 'completed') stats.completed++;
    else if (task.status === 'failed') stats.failed++;
    else if (task.status === 'running') stats.running++;
    else if (task.status === 'queued') stats.queued++;
    
    // Platform stats
    if (task.platform) {
      stats.platformStats[task.platform] = (stats.platformStats[task.platform] || 0) + 1;
    }
    
    // Execution time
    if (task.execution_time) {
      stats.executionTimes.push(task.execution_time);
    }
    
    // Duration
    if (task.result?.metadata?.duration) {
      stats.totalDuration += task.result.metadata.duration;
    }
    
    // Media sizes
    if (task.media_info) {
      stats.totalVideoSize += task.media_info.video_size_bytes || 0;
      stats.totalImageSize += task.media_info.total_image_size_bytes || 0;
      stats.totalAudioSize += task.media_info.total_audio_size_bytes || 0;
    }
    
    // Creation date (for timeline)
    if (task.created_at) {
      const date = new Date(task.created_at * 1000);
      const dateKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
      stats.creationDates[dateKey] = (stats.creationDates[dateKey] || 0) + 1;
    }
  });
  
  // Calculate derived stats
  stats.completionRate = stats.total > 0 ? (stats.completed / stats.total) * 100 : 0;
  stats.avgDuration = stats.completed > 0 ? stats.totalDuration / stats.completed : 0;
  stats.averageExecutionTime = stats.executionTimes.length > 0
    ? stats.executionTimes.reduce((sum, time) => sum + time, 0) / stats.executionTimes.length
    : 0;
  
  // Format bytes to human-readable
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };
  
  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Analytics Dashboard</h1>
        <p className="text-muted-foreground">View statistics and insights about your generated videos</p>
      </div>
      
      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        <div className="card p-6">
          <h3 className="text-muted-foreground text-sm font-medium mb-1">Total Videos</h3>
          <p className="text-3xl font-bold">{stats.total}</p>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-muted-foreground">Completion rate</span>
            <span className="text-sm font-medium">{stats.completionRate.toFixed(1)}%</span>
          </div>
        </div>
        
        <div className="card p-6">
          <h3 className="text-muted-foreground text-sm font-medium mb-1">Total Duration</h3>
          <p className="text-3xl font-bold">{stats.totalDuration.toFixed(1)}s</p>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-muted-foreground">Average duration</span>
            <span className="text-sm font-medium">{stats.avgDuration.toFixed(1)}s</span>
          </div>
        </div>
        
        <div className="card p-6">
          <h3 className="text-muted-foreground text-sm font-medium mb-1">Avg. Gen. Time</h3>
          <p className="text-3xl font-bold">{stats.averageExecutionTime.toFixed(1)}s</p>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-muted-foreground">Processing efficiency</span>
            <span className="text-sm font-medium">
              {stats.averageExecutionTime > 0
                ? (stats.avgDuration / stats.averageExecutionTime).toFixed(1)
                : 'N/A'}x
            </span>
          </div>
        </div>
        
        <div className="card p-6">
          <h3 className="text-muted-foreground text-sm font-medium mb-1">Storage Used</h3>
          <p className="text-3xl font-bold">{formatBytes(stats.totalVideoSize + stats.totalImageSize + stats.totalAudioSize)}</p>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-muted-foreground">Video storage</span>
            <span className="text-sm font-medium">{formatBytes(stats.totalVideoSize)}</span>
          </div>
        </div>
      </div>
      
      {/* Status breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="card p-6">
          <h2 className="text-lg font-medium mb-4">Status Breakdown</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Completed</span>
                <span className="text-sm font-medium">{stats.completed} ({Math.round(stats.completed / stats.total * 100) || 0}%)</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div className="bg-green-500 h-2.5 rounded-full" style={{ width: `${Math.round(stats.completed / stats.total * 100) || 0}%` }}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Failed</span>
                <span className="text-sm font-medium">{stats.failed} ({Math.round(stats.failed / stats.total * 100) || 0}%)</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div className="bg-red-500 h-2.5 rounded-full" style={{ width: `${Math.round(stats.failed / stats.total * 100) || 0}%` }}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Running</span>
                <span className="text-sm font-medium">{stats.running} ({Math.round(stats.running / stats.total * 100) || 0}%)</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div className="bg-blue-500 h-2.5 rounded-full" style={{ width: `${Math.round(stats.running / stats.total * 100) || 0}%` }}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Queued</span>
                <span className="text-sm font-medium">{stats.queued} ({Math.round(stats.queued / stats.total * 100) || 0}%)</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div className="bg-yellow-500 h-2.5 rounded-full" style={{ width: `${Math.round(stats.queued / stats.total * 100) || 0}%` }}></div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <h2 className="text-lg font-medium mb-4">Platform Distribution</h2>
          {Object.keys(stats.platformStats).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(stats.platformStats).map(([platform, count]) => (
                <div key={platform}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">{platform.replace('_', ' ').toUpperCase()}</span>
                    <span className="text-sm font-medium">{count} ({Math.round(count / stats.total * 100)}%)</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                    <div className="bg-primary h-2.5 rounded-full" style={{ width: `${Math.round(count / stats.total * 100)}%` }}></div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-12">
              No platform data available
            </div>
          )}
        </div>
      </div>
      
      {/* Generation timeline */}
      <div className="card p-6 mb-6">
        <h2 className="text-lg font-medium mb-4">Generation Timeline</h2>
        {Object.keys(stats.creationDates).length > 0 ? (
          <div>
            <div className="flex items-end h-48 gap-1">
              {Object.entries(stats.creationDates)
                .sort(([dateA], [dateB]) => dateA.localeCompare(dateB))
                .map(([date, count]) => (
                  <div key={date} className="flex flex-col items-center flex-1 min-w-0">
                    <div className="relative w-full">
                      <div 
                        className="absolute bottom-0 w-full bg-primary/80 rounded-t"
                        style={{ 
                          height: `${Math.max(15, Math.min(100, (count / Math.max(...Object.values(stats.creationDates))) * 100))}%` 
                        }}
                      ></div>
                    </div>
                    <div className="mt-2 text-xs text-muted-foreground whitespace-nowrap overflow-hidden text-ellipsis w-full text-center">
                      {date.split('-').slice(1).join('/')}
                    </div>
                  </div>
                ))
              }
            </div>
          </div>
        ) : (
          <div className="text-center text-muted-foreground py-12">
            No timeline data available
          </div>
        )}
      </div>
      
      {/* Actions */}
      <div className="flex justify-center gap-4">
        <Link href="/" className="button-primary">
          Create New Video
        </Link>
        <Link href="/history" className="button-secondary">
          View All Videos
        </Link>
      </div>
    </div>
  );
}