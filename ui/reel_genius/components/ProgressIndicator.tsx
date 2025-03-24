'use client';

import { useState, useEffect } from 'react';
import { TaskStatus } from '@/lib/types';

export interface ProcessStage {
  id: string;
  name: string;
  progressStart: number;
  progressEnd: number;
  description?: string;
  status?: 'pending' | 'active' | 'completed' | 'error';
}

interface ProgressIndicatorProps {
  progress: number;
  status: TaskStatus;
  showLabel?: boolean;
  height?: number;
  showDetailedProgress?: boolean;
  result?: any; // Optional result data for detailed progress
}

const DEFAULT_STAGES: ProcessStage[] = [
  {
    id: 'analyze',
    name: 'Analyzing content',
    progressStart: 0,
    progressEnd: 0.25,
    description: 'Analyzing content and planning strategy'
  },
  {
    id: 'script',
    name: 'Generating script',
    progressStart: 0.25,
    progressEnd: 0.5,
    description: 'Creating script and dialogue for the video'
  },
  {
    id: 'visuals',
    name: 'Creating visuals',
    progressStart: 0.5,
    progressEnd: 0.75,
    description: 'Generating images and preparing audio'
  },
  {
    id: 'assembly',
    name: 'Assembling video',
    progressStart: 0.75,
    progressEnd: 1,
    description: 'Combining media and finalizing video'
  }
];

export default function ProgressIndicator({
  progress,
  status,
  showLabel = true,
  height = 8,
  showDetailedProgress = false,
  result
}: ProgressIndicatorProps) {
  const [stage, setStage] = useState('');
  const [stages, setStages] = useState<ProcessStage[]>(DEFAULT_STAGES);
  
  useEffect(() => {
    if (status === 'queued') {
      setStage('Queued');
    } else if (status === 'running') {
      if (progress < 0.25) {
        setStage('Analyzing content');
      } else if (progress < 0.5) {
        setStage('Generating script');
      } else if (progress < 0.75) {
        setStage('Creating visuals');
      } else {
        setStage('Assembling video');
      }
    } else if (status === 'completed') {
      setStage('Complete');
    } else if (status === 'failed') {
      setStage('Failed');
    }
    
    // Update stages status based on progress
    const updatedStages = stages.map(stage => {
      let stageStatus: 'pending' | 'active' | 'completed' | 'error' = 'pending';
      
      if (progress >= stage.progressEnd) {
        stageStatus = 'completed';
      } else if (progress >= stage.progressStart) {
        stageStatus = 'active';
      }
      
      if (status === 'failed') {
        if (progress >= stage.progressStart) {
          stageStatus = 'error';
        }
      }
      
      return {
        ...stage,
        status: stageStatus
      };
    });
    
    setStages(updatedStages);
  }, [progress, status]);
  
  // Calculate stage-specific progress
  const getCurrentStageProgress = () => {
    const currentStage = stages.find(s => 
      progress >= s.progressStart && progress < s.progressEnd
    );
    
    if (!currentStage) return 100;
    
    const stageProgress = (progress - currentStage.progressStart) / 
      (currentStage.progressEnd - currentStage.progressStart) * 100;
    
    return Math.min(Math.max(stageProgress, 0), 100);
  };
  
  return (
    <div className="w-full">
      <div 
        className="progress-bar rounded-full overflow-hidden bg-muted"
        style={{ height: `${height}px` }}
      >
        <div 
          className="progress-bar-fill transition-width duration-500 ease-in-out"
          style={{ 
            width: `${Math.max(2, Math.min(progress * 100, 100))}%`,
            backgroundColor: 
              status === 'completed' ? 'var(--primary)' : 
              status === 'failed' ? 'var(--destructive)' : 
              'var(--primary)',
            height: `${height}px`
          }}
        />
      </div>
      
      {showLabel && (
        <div className="flex justify-between items-center mt-1 text-xs">
          <span>{stage}</span>
          <span>{Math.round(progress * 100)}%</span>
        </div>
      )}
      
      {showDetailedProgress && (
        <div className="mt-4 space-y-4">
          {stages.map((stage) => (
            <div key={stage.id} className="space-y-1">
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div 
                    className={`w-3 h-3 rounded-full mr-2 ${
                      stage.status === 'completed' ? 'bg-green-500' : 
                      stage.status === 'active' ? 'bg-primary animate-pulse' : 
                      stage.status === 'error' ? 'bg-destructive' : 
                      'bg-muted'
                    }`}
                  />
                  <span className="font-medium text-sm">{stage.name}</span>
                </div>
                
                {stage.status === 'active' && (
                  <span className="text-xs text-muted-foreground">
                    {Math.round(getCurrentStageProgress())}%
                  </span>
                )}
                
                {stage.status === 'completed' && (
                  <span className="text-xs text-green-500">✓ Complete</span>
                )}
              </div>
              
              {stage.description && (
                <p className="text-xs text-muted-foreground pl-5">{stage.description}</p>
              )}
              
              {stage.status === 'active' && (
                <div 
                  className="progress-bar ml-5 mt-1 rounded-full overflow-hidden bg-muted"
                  style={{ height: '4px' }}
                >
                  <div 
                    className="progress-bar-fill transition-width duration-300 ease-in-out"
                    style={{ 
                      width: `${getCurrentStageProgress()}%`,
                      backgroundColor: 'var(--primary)',
                      height: '4px'
                    }}
                  />
                </div>
              )}
              
              {/* Show scene count for visual creation stage */}
              {stage.id === 'visuals' && stage.status === 'active' && result?.processed_scenes && (
                <div className="pl-5 text-xs text-muted-foreground mt-1">
                  Generated {result.processed_scenes.length} scenes so far
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}