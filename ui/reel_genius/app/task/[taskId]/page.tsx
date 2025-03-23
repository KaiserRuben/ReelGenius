import { getTaskStatus, getSceneImageUrl, getSceneAudioUrl, getVideoUrl, SceneMediaInfo } from '@/lib/api';
import StatusBadge from '@/components/StatusBadge';
import ProgressIndicator from '@/components/ProgressIndicator';
import VideoPlayer from '@/components/VideoPlayer';
import MediaGallery, { MediaItem } from '@/components/MediaGallery';
import Link from 'next/link';

export const dynamic = 'force-dynamic';

interface PageProps {
  params: Promise<{ taskId: string }>
}

export default async function TaskDetailPage({
  params
}: PageProps) {
  const { taskId } = await params;
  
  // Server-side data fetching
  const taskData = await getTaskStatus(taskId).catch((error) => {
    console.error('Error fetching task:', error);
    return null;
  });
  
  if (!taskData) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="card p-6">
          <h1 className="text-3xl font-bold mb-2">Task Not Found</h1>
          <p className="text-muted-foreground mb-4">
            The task with ID {taskId} could not be found or has been deleted.
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
  
  if (taskData.status === 'completed' && taskData.result?.processed_scenes) {
    // Add scene images
    taskData.result.processed_scenes.forEach((scene: SceneMediaInfo, index: number) => {
      if (scene.image_path) {
        mediaItems.push({
          type: 'image',
          url: getSceneImageUrl(taskId, index),
          title: `Scene ${index + 1} Image`,
          index
        });
      }
      
      if (scene.voice_path) {
        mediaItems.push({
          type: 'audio',
          url: getSceneAudioUrl(taskId, index),
          title: `Scene ${index + 1} Audio`,
          index: mediaItems.length
        });
      }
    });
    
    // Add hook audio if available
    if (taskData.result.hook_audio_path) {
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
        <h1 className="text-3xl font-bold mb-2">Task Details</h1>
        <div className="flex items-center gap-3">
          <p className="text-muted-foreground">Task ID: {taskId}</p>
          <StatusBadge status={taskData.status} size="md" />
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="md:col-span-2 space-y-6">
          {/* Progress indicator for ongoing tasks */}
          {(taskData.status === 'running' || taskData.status === 'queued') && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Generation Progress</h2>
              <ProgressIndicator
                progress={taskData.progress || 0}
                status={taskData.status}
                height={10}
                showDetailedProgress={true}
                result={taskData.result}
              />
              
              <div className="mt-6 flex items-center justify-between text-sm text-muted-foreground">
                <p>
                  Task has been running for {
                    taskData.created_at 
                      ? `${Math.floor((Date.now() / 1000 - taskData.created_at) / 60)} minutes`
                      : 'an unknown amount of time'
                  }.
                </p>
                
                {taskData.result?.execution_time && (
                  <p>
                    Estimated completion: {
                      taskData.progress && taskData.progress > 0.05
                        ? `${Math.ceil(taskData.result.execution_time * (1 / taskData.progress - 1))} seconds`
                        : 'Calculating...'
                    }
                  </p>
                )}
              </div>
              
              {/* Live generation updates */}
              {taskData.status === 'running' && taskData.result && (
                <div className="mt-4 p-3 bg-muted/50 rounded-md">
                  <h3 className="text-sm font-medium mb-2">Generation Updates</h3>
                  
                  {taskData.progress < 0.25 && taskData.result.input_analysis && (
                    <div className="text-xs space-y-1">
                      <p>
                        <span className="font-medium">Content Topic:</span> {taskData.result.input_analysis.topic || 'Analyzing...'}
                      </p>
                      {taskData.result.input_analysis.keywords && (
                        <p>
                          <span className="font-medium">Keywords:</span> {taskData.result.input_analysis.keywords.join(', ')}
                        </p>
                      )}
                      {taskData.result.content_strategy && (
                        <p>
                          <span className="font-medium">Approach:</span> {taskData.result.content_strategy.tone || 'Standard'} tone for {taskData.result.content_strategy.target_audience || 'general audience'}
                        </p>
                      )}
                    </div>
                  )}
                  
                  {taskData.progress >= 0.25 && taskData.progress < 0.5 && taskData.result.script && (
                    <div className="text-xs space-y-1">
                      <p>
                        <span className="font-medium">Script Created:</span> {taskData.result.script.scenes ? `${taskData.result.script.scenes.length} scenes planned` : 'In progress...'}
                      </p>
                      {taskData.result.script.title && (
                        <p>
                          <span className="font-medium">Title:</span> {taskData.result.script.title}
                        </p>
                      )}
                      {taskData.result.script.hook && (
                        <p>
                          <span className="font-medium">Hook:</span> "{taskData.result.script.hook}"
                        </p>
                      )}
                    </div>
                  )}
                  
                  {taskData.progress >= 0.5 && taskData.progress < 0.75 && taskData.result.processed_scenes && (
                    <div className="text-xs space-y-1">
                      <p>
                        <span className="font-medium">Scene Progress:</span> Generated {taskData.result.processed_scenes.length} of {taskData.result.script?.scenes?.length || '?'} scenes
                      </p>
                      {taskData.result.processed_scenes.length > 0 && (
                        <div className="mt-2">
                          <span className="font-medium">Latest Scene:</span> 
                          <p className="italic mt-1">"{taskData.result.processed_scenes[taskData.result.processed_scenes.length - 1].text}"</p>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {taskData.progress >= 0.75 && (
                    <div className="text-xs">
                      <p><span className="font-medium">Finalizing Video:</span> Combining media assets...</p>
                      {taskData.result.metadata && (
                        <p className="mt-1">
                          <span className="font-medium">Estimated Length:</span> {taskData.result.metadata.duration?.toFixed(1) || 'Calculating...'} seconds
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          
          {/* Video player for completed videos */}
          {taskData.status === 'completed' && taskData.result?.video_path && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Generated Video</h2>
              <VideoPlayer 
                src={getVideoUrl(taskId)}
                title={taskData.result.metadata?.title || 'Generated Video'}
              />
              
              <div className="mt-4 flex justify-center">
                <a
                  href={getVideoUrl(taskId)}
                  download
                  className="button-primary"
                >
                  Download Video
                </a>
              </div>
            </div>
          )}
          
          {/* Media gallery for completed videos */}
          {mediaItems.length > 0 && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Media Gallery</h2>
              <MediaGallery items={mediaItems} taskId={taskId} />
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
          {taskData.status === 'completed' && taskData.result?.video_path && (
            <div className="card p-6">
              <h2 className="text-lg font-medium mb-4">Video Details</h2>
              <div className="space-y-3">
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">File Size</h3>
                  <p className="text-sm">
                    {formatFileSize(taskData.result.metadata?.file_size)}
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-muted-foreground mb-1">Scene Count</h3>
                  <p className="text-sm">
                    {taskData.result.processed_scenes?.length || 0} scenes
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
                    {taskData.result.hook_audio_path ? 'Yes' : 'No'}
                  </p>
                </div>
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
              
              {taskData.status === 'completed' && taskData.result?.video_path && (
                <a
                  href={getVideoUrl(taskId)}
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