import { useState, useEffect } from 'react';
import { getTasks, TaskData } from '@/lib/api';
import { extractData } from '@/lib/types';
import Link from 'next/link';

export default function StatusUpdates() {
  const [recentTasks, setRecentTasks] = useState<TaskData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTasks, setActiveTasks] = useState(0);
  
  useEffect(() => {
    const loadTasks = async () => {
      try {
        setLoading(true);
        const response = await getTasks(undefined, undefined, 5, 0, true);
        
        const tasksData = extractData(response);
        if (tasksData && tasksData.tasks) {
          setRecentTasks(tasksData.tasks);
          
          // Count active tasks
          const active = tasksData.tasks.filter(
            task => task.status === 'running' || task.status === 'queued'
          ).length;
          setActiveTasks(active);
        }
      } catch (err) {
        setError('Failed to load recent tasks');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    // Load initially
    loadTasks();
    
    // Set up polling interval if there are active tasks
    const interval = setInterval(() => {
      if (activeTasks > 0) {
        loadTasks();
      }
    }, 10000);
    
    return () => clearInterval(interval);
  }, [activeTasks]);
  
  if (loading) {
    return (
      <div className="flex justify-center items-center h-24">
        <div className="animate-pulse text-primary">Loading status updates...</div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="text-center p-4 text-sm text-destructive">
        {error}
      </div>
    );
  }
  
  if (recentTasks.length === 0) {
    return (
      <div className="text-center p-4 text-sm text-muted-foreground">
        No recent tasks found. Start by creating a new video!
      </div>
    );
  }
  
  // Format timestamp
  const formatTime = (timestamp?: number) => {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp * 1000);
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };
  
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium mb-3">Recent Tasks</h3>
      
      <div className="space-y-2">
        {recentTasks.map((task) => (
          <Link 
            key={task.task_id}
            href={`/task/${task.task_id}`}
            className="flex items-center justify-between p-2 hover:bg-muted/50 rounded-md transition-colors"
          >
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                task.status === 'completed' ? 'bg-green-500' :
                task.status === 'running' ? 'bg-blue-500 animate-pulse' :
                task.status === 'queued' ? 'bg-yellow-500' :
                'bg-red-500'
              }`} />
              <div className="flex-1 min-w-0">
                <p className="truncate text-sm">
                  {task.result?.metadata?.title || task.content_summary || task.task_id.substring(0, 8)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {task.platform} • {formatTime(task.created_at)}
                </p>
              </div>
            </div>
            
            <div className="text-right">
              {task.status === 'running' && (
                <span className="text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 px-2 py-0.5 rounded-full">
                  {Math.round((task.progress || 0) * 100)}%
                </span>
              )}
              
              {task.status === 'completed' && task.result?.metadata?.duration && (
                <span className="text-xs text-muted-foreground">
                  {task.result.metadata.duration.toFixed(0)}s
                </span>
              )}
              
              {task.status === 'failed' && (
                <span className="text-xs bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300 px-2 py-0.5 rounded-full">
                  Failed
                </span>
              )}
              
              {task.status === 'queued' && (
                <span className="text-xs bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-300 px-2 py-0.5 rounded-full">
                  Queued
                </span>
              )}
            </div>
          </Link>
        ))}
      </div>
      
      <div className="pt-2 border-t">
        <Link 
          href="/history"
          className="text-xs text-primary hover:underline flex items-center justify-center gap-1"
        >
          <span>View all tasks</span>
          <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </Link>
      </div>
    </div>
  );
}