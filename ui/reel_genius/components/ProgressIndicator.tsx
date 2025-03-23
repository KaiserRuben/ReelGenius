'use client';

import { useState, useEffect } from 'react';

import { TaskStatus } from './StatusBadge';

interface ProgressIndicatorProps {
  progress: number;
  status: TaskStatus;
  showLabel?: boolean;
  height?: number;
}

export default function ProgressIndicator({
  progress,
  status,
  showLabel = true,
  height = 8
}: ProgressIndicatorProps) {
  const [stage, setStage] = useState('');
  
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
  }, [progress, status]);
  
  return (
    <div className="w-full">
      <div 
        className="progress-bar"
        style={{ height: `${height}px` }}
      >
        <div 
          className="progress-bar-fill"
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
    </div>
  );
}