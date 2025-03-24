import { getTaskStatus, getSceneImageUrl, getSceneAudioUrl, getVideoUrl, SceneMediaInfo, TaskData } from '@/lib/api';
import { extractData } from '@/lib/types';
import StatusBadge from '@/components/StatusBadge';
import VideoPlayer from '@/components/VideoPlayer';
import MediaGallery, { MediaItem } from '@/components/MediaGallery';
import AlgorithmMetricsPanel from '@/components/AlgorithmMetricsPanel';
import PromptTemplateViewer from '@/components/PromptTemplateViewer';
import SceneDetailsViewer from '@/components/SceneDetailsViewer';
import ContentStrategy from '@/components/ContentStrategy';
import CacheStats from '@/components/CacheStats';
import TaskProgress from '@/components/TaskProgress';
import Link from 'next/link';
import { notFound } from 'next/navigation';

export const dynamic = 'force-dynamic';

interface PageProps {
  params: Promise<{ taskId: string }>
}

export default async function TaskDetailPage({
  params
}: PageProps) {
  const { taskId } = await params;
  
  // Server-side data fetching with new standardized response format
  const response = await getTaskStatus(taskId, true).catch((error) => {
    console.error('Error fetching task:', error);
    return null;
  });
  
  if (!response) {
    notFound();
  }
  
  // Extract the task data from our standardized response
  let taskData: TaskData;
  try {
    const extractedData = extractData(response);
    if (!extractedData) {
      notFound();
    }
    taskData = extractedData as TaskData;
  } catch (error) {
    console.error('Error processing task data:', error);
    return (
      <div className="max-w-5xl mx-auto">
        <div className="card p-6 border-destructive">
          <h1 className="text-3xl font-bold mb-2">Error Loading Task</h1>
          <p className="text-muted-foreground mb-4">
            {response.error || "Failed to load task data"}
          </p>
          <Link href="/history" className="button-primary inline-block">
            Back to History
          </Link>
        </div>
      </div>
    );
  }
  
  // Format timestamp
  const formatTime = (timestamp?: number) => {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp * 1000);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }).format(date);
  };
  
  // Format file size
  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'N/A';
    
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };
  
  // Prepare media items for gallery if video is completed
  const mediaItems: MediaItem[] = [];
  
  if (taskData.status === 'completed' && taskData.result?.scenes) {
    // Add scene images and audio from the new data structure
    taskData.result.scenes.forEach((scene: SceneMediaInfo, index: number) => {
      if (scene.image_url) {
        mediaItems.push({
          type: 'image',
          url: scene.image_url,
          title: `Scene ${index + 1} Image`,
          index,
        });
      } else if (scene.index !== undefined) {
        // Fall back to generated URL if direct URL not provided
        mediaItems.push({
          type: 'image',
          url: getSceneImageUrl(taskId, scene.index),
          title: `Scene ${index + 1} Image`,
          index: scene.index
        });
      }
      
      if (scene.audio_url) {
        mediaItems.push({
          type: 'audio',
          url: scene.audio_url,
          title: `Scene ${index + 1} Audio`,
          index: mediaItems.length,
        });
      } else if (scene.index !== undefined) {
        // Fall back to generated URL if direct URL not provided
        mediaItems.push({
          type: 'audio',
          url: getSceneAudioUrl(taskId, scene.index),
          title: `Scene ${index + 1} Audio`,
          index: mediaItems.length
        });
      }
    });
    
    // Add hook audio if available (falling back to traditional path if needed)
    if (taskData.result.hook_audio_url) {
      mediaItems.push({
        type: 'audio',
        url: taskData.result.hook_audio_url,
        title: 'Hook Audio',
        index: mediaItems.length
      });
    } else if (taskData.result.hook_audio_path) {
      mediaItems.push({
        type: 'audio',
        url: getSceneAudioUrl(taskId, 'hook'),
        title: 'Hook Audio',
        index: mediaItems.length
      });
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Video Task Details</h1>
        <div className="flex items-center gap-3">
          <p className="text-muted-foreground">Task ID: {taskId}</p>
          <StatusBadge status={taskData.status} size="md" />
          {taskData.result?.cache_stats?.money_saved > 0 && (
            <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 px-2 py-1 rounded-full">
              Saved ${taskData.result.cache_stats.money_saved.toFixed(2)}
            </span>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="md:col-span-2 space-y-6">
          {/* Progress indicator for ongoing tasks */}
          {(taskData.status === 'running' || taskData.status === 'queued') && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Generation Progress</h2>
              
              <TaskProgress taskData={taskData} />
            </div>
          )}
          
          {/* Video player for completed videos */}
          {taskData.status === 'completed' && (taskData.result?.video_url || taskData.result?.video_file) && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Generated Video</h2>
              <VideoPlayer 
                src={taskData.result.video_url || taskData.result.video_file?.url || getVideoUrl(taskId)}
                title={taskData.result.metadata?.title || 'Generated Video'}
              />
              
              <div className="mt-4 flex justify-center">
                <a
                  href={taskData.result.video_url || taskData.result.video_file?.url || getVideoUrl(taskId)}
                  download
                  className="button-primary"
                >
                  Download Video
                </a>
              </div>
              
              {taskData.result.video_file?.size_bytes && (
                <div className="mt-2 text-sm text-center text-muted-foreground">
                  File size: {formatFileSize(taskData.result.video_file.size_bytes)}
                </div>
              )}
            </div>
          )}
          
          {/* Media gallery for completed videos */}
          {mediaItems.length > 0 && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Media Gallery</h2>
              <MediaGallery items={mediaItems} taskId={taskId} />
            </div>
          )}
          
          {/* Scene Details Viewer */}
          {taskData.status === 'completed' && taskData.result?.scenes && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Scene Details</h2>
              <SceneDetailsViewer 
                scenes={taskData.result.scenes} 
                showPrompts={true} 
              />
            </div>
          )}
          
          {/* Metadata for completed videos */}
          {taskData.status === 'completed' && taskData.result?.metadata && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Video Metadata</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Title</h3>
                  <p>{taskData.result.metadata.title || 'Untitled'}</p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Platform</h3>
                  <p>{taskData.result.metadata.platform || taskData.platform || 'Unknown'}</p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Duration</h3>
                  <p>{taskData.result.metadata.duration?.toFixed(1) || 'Unknown'} seconds</p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Resolution</h3>
                  <p>{taskData.result.metadata.resolution || 'Unknown'}</p>
                </div>
                
                <div className="md:col-span-2">
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Description</h3>
                  <p className="whitespace-pre-line">{taskData.result.metadata.description || 'No description'}</p>
                </div>
                
                <div className="md:col-span-2">
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Hashtags</h3>
                  <div className="flex flex-wrap gap-1">
                    {taskData.result.metadata.hashtags?.map((tag: string, index: number) => (
                      <span 
                        key={index}
                        className="inline-block px-2 py-1 bg-primary/10 text-primary text-xs rounded-full"
                      >
                        {tag}
                      </span>
                    )) || 'No hashtags'}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Algorithm Optimization Metrics */}
          {taskData.status === 'completed' && taskData.result?.algorithm_metrics && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Algorithm Optimization Metrics</h2>
              <AlgorithmMetricsPanel data={taskData.result} />
            </div>
          )}
          
          {/* Content Strategy & Analysis */}
          {taskData.status === 'completed' && (taskData.result?.content_strategy || taskData.result?.content_analysis || taskData.result?.visual_plan) && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Content Strategy</h2>
              <ContentStrategy 
                contentStrategy={taskData.result?.content_strategy}
                contentAnalysis={taskData.result?.content_analysis}
                visualPlan={taskData.result?.visual_plan}
              />
            </div>
          )}
          
          {/* Cache Statistics */}
          {taskData.status === 'completed' && taskData.result?.cache_stats && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Cache Performance</h2>
              <CacheStats cacheStats={taskData.result.cache_stats} />
            </div>
          )}
          
          {/* Prompt Templates Viewer */}
          {taskData.status === 'completed' && taskData.result?.prompt_templates && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Prompt Templates</h2>
              <PromptTemplateViewer templates={taskData.result.prompt_templates} />
            </div>
          )}
          
          {/* Error information for failed tasks */}
          {taskData.status === 'failed' && taskData.error && (
            <div className="card p-6 border-destructive">
              <h2 className="text-lg font-medium mb-4 text-destructive">Error Details</h2>
              <div className="p-3 bg-destructive/10 rounded-md">
                <p className="text-destructive whitespace-pre-line">{taskData.error}</p>
              </div>
            </div>
          )}
        </div>
        
        {/* Sidebar */}
        <div className="space-y-6">
          {/* Task information */}
          <div className="card p-6">
            <h2 className="text-lg font-medium mb-4">Task Information</h2>
            <div className="space-y-3">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Status</h3>
                <StatusBadge status={taskData.status} />
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Created</h3>
                <p className="text-sm">{formatTime(taskData.created_at)}</p>
              </div>
              
              {taskData.updated_at && (
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Last Updated</h3>
                  <p className="text-sm">{formatTime(taskData.updated_at)}</p>
                </div>
              )}
              
              {taskData.platform && (
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Platform</h3>
                  <p className="text-sm">{taskData.platform}</p>
                </div>
              )}
              
              {taskData.execution_time && (
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Execution Time</h3>
                  <p className="text-sm">{taskData.execution_time.toFixed(1)} seconds</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Video details */}
          {taskData.status === 'completed' && taskData.result && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Video Details</h2>
              <div className="space-y-3">
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">File Size</h3>
                  <p className="text-sm">
                    {formatFileSize(taskData.result.video_file?.size_bytes || taskData.result.metadata?.file_size)}
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Scene Count</h3>
                  <p className="text-sm">
                    {taskData.result.scene_count || taskData.result.scenes?.length || 0} scenes
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Category</h3>
                  <p className="text-sm">
                    {taskData.result.metadata?.category || 'Uncategorized'}
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Has Hook</h3>
                  <p className="text-sm">
                    {taskData.result.hook_audio_url || taskData.result.hook_audio_path ? 'Yes' : 'No'}
                  </p>
                </div>
                
                {taskData.execution_time && (
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground mb-1">Generation Time</h3>
                    <p className="text-sm">
                      {Math.round(taskData.execution_time)} seconds
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Content preview */}
          {taskData.content_summary && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Content Preview</h2>
              <div className="max-h-40 overflow-y-auto text-sm text-muted-foreground">
                {taskData.content_summary}
              </div>
            </div>
          )}
          
          {/* Actions */}
          <div className="card p-6">
            <h2 className="text-lg font-medium mb-4">Actions</h2>
            <div className="space-y-3">
              <Link
                href="/history"
                className="button-secondary w-full flex justify-center"
              >
                Back to History
              </Link>
              
              {taskData.status === 'completed' && (taskData.result?.video_url || taskData.result?.video_file) && (
                <a
                  href={taskData.result.video_url || taskData.result.video_file?.url || getVideoUrl(taskId)}
                  download
                  className="button-primary w-full flex justify-center"
                >
                  Download Video
                </a>
              )}
              
              <Link
                href="/"
                className="text-primary text-sm hover:underline w-full flex justify-center"
              >
                Create New Video
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}