import { Suspense } from 'react';
import { notFound } from 'next/navigation';
import { getTaskStatus, getVideoUrl, getSceneImageUrl, getSceneAudioUrl } from '@/lib/api';
import { isValidTaskId, VideoResponse, SceneMedia } from '@/lib/types';
import VideoPlayer from '@/components/VideoPlayer';
import StatusBadge from '@/components/StatusBadge';
import ProgressIndicator from '@/components/ProgressIndicator';

interface TaskDetailParams {
  params: {
    taskId: string;
  };
}

export default async function TaskDetailPage({ params }: TaskDetailParams) {
  const paramsData = await params;
  const { taskId } = paramsData;
  
  // Validate taskId format
  if (!isValidTaskId(taskId)) {
    notFound();
  }
  
  try {
    // Fetch task data with full details
    const response = await getTaskStatus(taskId, true);
    
    if (!response || response.status === 'error') {
      throw new Error(response?.error || 'Failed to load task');
    }
    
    const taskData = response.data as VideoResponse;
    
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col mb-6">
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-3xl font-bold">Task {taskId.substring(0, 8)}</h1>
            <StatusBadge status={taskData.status} size="lg" />
          </div>
          <p className="text-muted-foreground">
            Platform: <span className="font-medium">{taskData.platform || 'Not specified'}</span> | 
            Created: <span className="font-medium">{taskData.created_at ? new Date(taskData.created_at * 1000).toLocaleString() : 'Unknown'}</span>
          </p>
        </div>
        
        {/* Task status and progress */}
        <div className="card p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Status</h2>
          <ProgressIndicator 
            progress={taskData.progress || 0} 
            status={taskData.status} 
            showDetailedProgress={true} 
            result={taskData.status === 'running' ? taskData : null}
          />
          
          {taskData.execution_time && (
            <div className="mt-4 text-sm">
              <p className="text-muted-foreground">
                Total processing time: <span className="font-medium">{Math.round(taskData.execution_time)} seconds</span>
              </p>
            </div>
          )}
          
          {taskData.error && (
            <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 rounded-md">
              <h3 className="font-semibold">Error</h3>
              <p>{taskData.error}</p>
            </div>
          )}
        </div>
        
        {/* Content summary */}
        {taskData.content_summary && (
          <div className="card p-6 mb-6">
            <h2 className="text-xl font-semibold mb-2">Content</h2>
            <p className="text-muted-foreground whitespace-pre-wrap">{taskData.content_summary}</p>
          </div>
        )}
        
        {/* Video player for completed tasks */}
        {taskData.status === 'completed' && taskData.video && (
          <div className="card p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Generated Video</h2>
            <Suspense fallback={<div className="animate-pulse h-64 bg-muted rounded-lg"></div>}>
              <VideoPlayer
                src={getVideoUrl(taskId)}
                title={taskData.metadata?.title || 'Generated Video'}
                controls={true}
                width="100%"
                height="auto"
              />
            </Suspense>
            
            {taskData.metadata && (
              <div className="mt-4 space-y-2">
                <h3 className="font-semibold">Metadata</h3>
                
                {taskData.metadata.title && (
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Title:</span>
                    <span className="col-span-2 font-medium">{taskData.metadata.title}</span>
                  </div>
                )}
                
                {taskData.metadata.description && (
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Description:</span>
                    <span className="col-span-2">{taskData.metadata.description}</span>
                  </div>
                )}
                
                {taskData.metadata.hashtags && taskData.metadata.hashtags.length > 0 && (
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Hashtags:</span>
                    <div className="col-span-2 flex flex-wrap gap-1">
                      {taskData.metadata.hashtags.map((tag, idx) => (
                        <span key={idx} className="bg-secondary text-secondary-foreground rounded-md px-2 py-0.5 text-xs">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {taskData.metadata.duration && (
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Duration:</span>
                    <span className="col-span-2">{taskData.metadata.duration.toFixed(2)} seconds</span>
                  </div>
                )}
                
                {taskData.metadata.resolution && (
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Resolution:</span>
                    <span className="col-span-2">{taskData.metadata.resolution}</span>
                  </div>
                )}
                
                {taskData.video.size_bytes && (
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">File size:</span>
                    <span className="col-span-2">{Math.round(taskData.video.size_bytes / 1024 / 1024 * 100) / 100} MB</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        
        {/* Scenes section for completed tasks */}
        {taskData.status === 'completed' && taskData.scenes && taskData.scenes.length > 0 && (
          <div className="card p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Scenes</h2>
            <div className="space-y-6">
              {taskData.scenes.map((scene: SceneMedia) => (
                <div key={scene.index} className="p-4 border border-border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold">Scene {scene.index + 1}</h3>
                    <span className="text-xs text-muted-foreground">({scene.duration.toFixed(2)}s)</span>
                  </div>
                  
                  {/* Scene text */}
                  <p className="text-sm mb-4">{scene.text}</p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Scene image */}
                    {scene.image_url && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Image</h4>
                        <div className="relative aspect-video bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden">
                          <img 
                            src={getSceneImageUrl(taskId, scene.index.toString())} 
                            alt={`Scene ${scene.index + 1}`}
                            className="object-cover w-full h-full"
                          />
                        </div>
                        
                        {scene.image_prompt && (
                          <div className="text-xs text-muted-foreground mt-1">
                            <strong>Prompt:</strong> {scene.image_prompt}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Scene audio */}
                    {scene.audio_url && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Audio</h4>
                        <audio 
                          src={getSceneAudioUrl(taskId, scene.index.toString())} 
                          controls 
                          className="w-full" 
                        />
                        <div className="text-xs text-muted-foreground">
                          Duration: {scene.duration.toFixed(2)}s
                          {scene.audio_size_bytes && ` | Size: ${Math.round(scene.audio_size_bytes / 1024)} KB`}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Media info for completed tasks */}
        {taskData.status === 'completed' && taskData.media_info && (
          <div className="card p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Media Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Total Scenes:</span>
                  <span className="font-medium">{taskData.scene_count || taskData.scenes?.length || 0}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Images:</span>
                  <span className="font-medium">{taskData.media_info.image_count || 0}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Audio Clips:</span>
                  <span className="font-medium">{taskData.media_info.audio_count || 0}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Has Hook:</span>
                  <span className="font-medium">{taskData.media_info.has_hook ? 'Yes' : 'No'}</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Video Size:</span>
                  <span className="font-medium">
                    {taskData.media_info.video_size_bytes 
                      ? `${Math.round(taskData.media_info.video_size_bytes / 1024 / 1024 * 100) / 100} MB` 
                      : 'Unknown'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Images Size:</span>
                  <span className="font-medium">
                    {taskData.media_info.total_image_size_bytes 
                      ? `${Math.round(taskData.media_info.total_image_size_bytes / 1024 / 1024 * 100) / 100} MB` 
                      : 'Unknown'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <span className="text-muted-foreground">Audio Size:</span>
                  <span className="font-medium">
                    {taskData.media_info.total_audio_size_bytes 
                      ? `${Math.round(taskData.media_info.total_audio_size_bytes / 1024 / 1024 * 100) / 100} MB` 
                      : 'Unknown'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  } catch (error) {
    console.error('Error loading task:', error);
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card p-6">
          <h1 className="text-2xl font-bold mb-4 text-destructive">Error Loading Task</h1>
          <p>Failed to load task information. The task may not exist or there was a network error.</p>
          <p className="text-sm text-muted-foreground mt-2">
            Task ID: {taskId}
          </p>
        </div>
      </div>
    );
  }
}