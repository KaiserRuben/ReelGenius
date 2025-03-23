import { getTasks, TaskStatusResponse } from '@/lib/api';
import Link from 'next/link';
import StatusBadge from '@/components/StatusBadge';

export const dynamic = 'force-dynamic';

export default async function HistoryPage() {
  // Server-side data fetching
  const tasksData = await getTasks(
    undefined, // status
    undefined, // platform
    20, // limit
    0, // skip
    true, // include_details
    false // include_media_info
  ).catch((error) => {
    console.error('Error fetching tasks:', error);
    return { tasks: [], total: 0, limit: 20, skip: 0, has_more: false };
  });
  
  // Format timestamp
  const formatTime = (timestamp?: number) => {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp * 1000);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };
  
  // Calculate time ago
  const getTimeAgo = (timestamp?: number) => {
    if (!timestamp) return '';
    
    const seconds = Math.floor((Date.now() / 1000) - timestamp);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Video History</h1>
        <p className="text-muted-foreground">View and manage your generated videos</p>
      </div>
      
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-accent/50">
                <th className="px-4 py-3 text-left text-sm font-medium">Task ID</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Platform</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Created</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Duration</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {tasksData.tasks.map((task: TaskStatusResponse) => (
                <tr key={task.task_id} className="hover:bg-accent/30">
                  <td className="px-4 py-3 text-sm">
                    <div className="font-mono">{task.task_id.substring(0, 8)}...</div>
                    {task.content_summary && (
                      <div className="text-xs text-muted-foreground truncate max-w-40">
                        {task.content_summary}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    {task.platform || 'N/A'}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={task.status} />
                    {task.status === 'running' && task.progress !== undefined && (
                      <div className="mt-1 w-24 h-1 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500"
                          style={{ width: `${Math.max(2, task.progress * 100)}%` }}
                        />
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    <div>{formatTime(task.created_at)}</div>
                    <div className="text-xs text-muted-foreground">
                      {getTimeAgo(task.created_at)}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    {task.result?.metadata?.duration 
                      ? `${task.result.metadata.duration.toFixed(1)}s` 
                      : task.execution_time 
                        ? `${task.execution_time.toFixed(1)}s` 
                        : 'N/A'
                    }
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex gap-2">
                      <Link 
                        href={`/task/${task.task_id}`}
                        className="px-2 py-1 text-xs bg-primary/10 text-primary rounded hover:bg-primary/20"
                      >
                        View
                      </Link>
                      {task.status === 'completed' && task.result?.video_path && (
                        <a 
                          href={`/api/video/${task.task_id}`}
                          className="px-2 py-1 text-xs bg-accent/70 text-foreground rounded hover:bg-accent"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Download
                        </a>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
              
              {tasksData.tasks.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                    No videos found. Generate your first video!
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="mt-4 flex justify-between items-center">
        <p className="text-sm text-muted-foreground">
          Showing {tasksData.tasks.length} of {tasksData.total} videos
        </p>
        <div className="flex gap-2">
          <Link
            href="/history?page=1"
            className={`px-3 py-1 text-sm rounded border border-border ${
              tasksData.skip === 0 ? 'opacity-50 pointer-events-none' : 'hover:bg-accent'
            }`}
          >
            Previous
          </Link>
          <Link
            href={`/history?page=${Math.floor(tasksData.skip / tasksData.limit) + 2}`}
            className={`px-3 py-1 text-sm rounded border border-border ${
              !tasksData.has_more ? 'opacity-50 pointer-events-none' : 'hover:bg-accent'
            }`}
          >
            Next
          </Link>
        </div>
      </div>
    </div>
  );
}