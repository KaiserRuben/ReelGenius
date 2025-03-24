"use client"
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import ProgressIndicator from './ProgressIndicator';
import { TaskData } from '@/lib/api';
import { fadeIn, slideUp, staggerChildren, listItemVariant } from './animations';

interface TaskProgressProps {
  taskData: TaskData;
  refreshInterval?: number;
}

export default function TaskProgress({ taskData }: TaskProgressProps) {
  const [timeSinceStart, setTimeSinceStart] = useState<number>(0);
  const [currentStageTask, setCurrentStageTask] = useState<string | null>(null);
  
  useEffect(() => {
    // Calculate time since task started
    if (taskData.created_at) {
      const startTimeMs = taskData.created_at * 1000; // Convert to milliseconds
      const updateTimer = () => {
        const now = Date.now();
        setTimeSinceStart(Math.floor((now - startTimeMs) / 1000)); // in seconds
      };
      
      updateTimer(); // Initial call
      const timerId = setInterval(updateTimer, 1000);
      
      return () => clearInterval(timerId);
    }
  }, [taskData.created_at]);
  
  useEffect(() => {
    // Determine current stage task based on progress
    if (taskData.status === 'running') {
      const progress = taskData.progress || 0;
      
      if (progress < 0.25) {
        // Content analysis stage
        if (taskData.result?.content_analysis) {
          setCurrentStageTask('Analyzing content structure and topic');
        } else if (taskData.result?.content_strategy) {
          setCurrentStageTask('Developing content strategy');
        } else {
          setCurrentStageTask('Processing input content');
        }
      } else if (progress < 0.5) {
        // Script generation stage
        if (taskData.result?.script?.hook) {
          setCurrentStageTask('Creating scene-by-scene script');
        } else {
          setCurrentStageTask('Engineering hook and narrative structure');
        }
      } else if (progress < 0.75) {
        // Visual generation stage
        const sceneCount = taskData.result?.processed_scenes?.length || 0;
        const totalScenes = taskData.result?.script?.scenes?.length || 1;
        
        if (sceneCount > 0) {
          setCurrentStageTask(`Generating scene ${sceneCount}/${totalScenes} assets`);
        } else {
          setCurrentStageTask('Creating visual assets');
        }
      } else {
        // Assembly stage
        if (taskData.result?.metadata?.duration) {
          setCurrentStageTask('Finalizing video encoding');
        } else if (taskData.result?.processed_scenes?.length) {
          setCurrentStageTask('Combining audio and visuals');
        } else {
          setCurrentStageTask('Assembling video components');
        }
      }
    } else {
      setCurrentStageTask(null);
    }
  }, [taskData]);

  // Format time string (MM:SS)
  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };
  
  // Calculate estimated time remaining
  const getEstimatedTimeRemaining = (): string => {
    if (!taskData.progress || taskData.progress === 0) return 'Calculating...';
    
    if (taskData.time_estimates?.estimated_remaining_seconds) {
      return formatTime(Math.ceil(taskData.time_estimates.estimated_remaining_seconds));
    }
    
    // If we have execution time but no time estimates, calculate based on progress
    if (taskData.result?.execution_time && taskData.progress > 0.05) {
      const totalEstimated = taskData.result.execution_time / taskData.progress;
      const remaining = totalEstimated - taskData.result.execution_time;
      return formatTime(Math.ceil(remaining));
    }
    
    return 'Calculating...';
  };

  return (
    <motion.div 
      className="space-y-4"
      initial="hidden"
      animate="visible"
      variants={staggerChildren}
    >
      {/* Progress bar */}
      <motion.div variants={fadeIn}>
        <ProgressIndicator
          progress={taskData.progress || 0}
          status={taskData.status}
          height={8}
          showDetailedProgress={true}
          result={taskData.result}
        />
      </motion.div>
      
      {/* Current task and time information */}
      {taskData.status === 'running' && (
        <motion.div 
          className="mt-4 p-3 bg-muted/50 rounded-md shadow-sm"
          variants={slideUp}
        >
          <div className="flex justify-between items-center text-sm mb-2">
            <span className="font-medium">Current task:</span>
            <motion.span 
              className="text-primary font-medium"
              key={currentStageTask}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {currentStageTask || 'Processing...'}
            </motion.span>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
            <div>
              <span className="block">Runtime:</span>
              <motion.span 
                className="font-mono"
                animate={{ color: timeSinceStart > 120 ? '#f59e0b' : '#64748b' }}
                transition={{ duration: 0.5 }}
              >
                {formatTime(timeSinceStart)}
              </motion.span>
            </div>
            
            <div>
              <span className="block">Est. remaining:</span>
              <span className="font-mono">{getEstimatedTimeRemaining()}</span>
            </div>
          </div>
        </motion.div>
      )}
      
      {/* Generation updates */}
      {taskData.status === 'running' && taskData.result && (
        <motion.div 
          className="mt-4 p-3 bg-muted/50 rounded-md shadow-sm"
          variants={slideUp}
        >
          <h3 className="text-sm font-medium mb-2">Generation Updates</h3>
          
          {/* Content analysis stage updates */}
          {taskData.progress < 0.25 && (
            <motion.div 
              className="text-xs space-y-1"
              variants={staggerChildren}
            >
              {taskData.result.content_analysis?.topic && (
                <motion.p 
                  variants={listItemVariant}
                  custom={0}
                >
                  <span className="font-medium">Content Topic:</span> {taskData.result.content_analysis.topic}
                </motion.p>
              )}
              
              {taskData.result.content_analysis?.keywords && (
                <motion.p 
                  variants={listItemVariant}
                  custom={1}
                >
                  <span className="font-medium">Keywords:</span> {taskData.result.content_analysis.keywords.join(', ')}
                </motion.p>
              )}
              
              {taskData.result.content_strategy?.hook_engineering?.approach && (
                <motion.p 
                  variants={listItemVariant}
                  custom={2}
                >
                  <span className="font-medium">Hook Strategy:</span> {taskData.result.content_strategy.hook_engineering.approach.substring(0, 80)}...
                </motion.p>
              )}
              
              {taskData.result.visual_plan?.style?.description && (
                <motion.p 
                  variants={listItemVariant}
                  custom={3}
                >
                  <span className="font-medium">Visual Style:</span> {taskData.result.visual_plan.style.description.substring(0, 80)}...
                </motion.p>
              )}
            </motion.div>
          )}
          
          {/* Script generation stage updates */}
          {taskData.progress >= 0.25 && taskData.progress < 0.5 && (
            <motion.div 
              className="text-xs space-y-1"
              variants={staggerChildren}
            >
              {taskData.result.script?.title && (
                <motion.p 
                  variants={listItemVariant}
                  custom={0}
                >
                  <span className="font-medium">Title:</span> {taskData.result.script.title}
                </motion.p>
              )}
              
              {taskData.result.script?.hook && (
                <motion.p 
                  variants={listItemVariant}
                  custom={1}
                  className="p-1.5 bg-primary/5 rounded"
                >
                  <span className="font-medium">Hook:</span> <span className="italic">&quot;{taskData.result.script.hook}&quot;</span>
                </motion.p>
              )}
              
              {taskData.result.script?.scenes && (
                <motion.p 
                  variants={listItemVariant}
                  custom={2}
                >
                  <span className="font-medium">Scenes Planned:</span> {taskData.result.script.scenes.length}
                </motion.p>
              )}
              
              {taskData.result.script?.duration && (
                <motion.p 
                  variants={listItemVariant}
                  custom={3}
                >
                  <span className="font-medium">Estimated Duration:</span> {taskData.result.script.duration.toFixed(1)} seconds
                </motion.p>
              )}
            </motion.div>
          )}
          
          {/* Visual generation stage updates */}
          {taskData.progress >= 0.5 && taskData.progress < 0.75 && taskData.result.processed_scenes && (
            <motion.div 
              className="text-xs space-y-2"
              variants={staggerChildren}
            >
              <motion.div 
                variants={listItemVariant}
                custom={0}
                className="flex items-center"
              >
                <span className="font-medium">Scene Progress:</span>
                <div className="ml-2 flex items-center">
                  <span className="whitespace-nowrap">Generated {taskData.result.processed_scenes.length} of {taskData.result.script?.scenes?.length || '?'} scenes</span>
                  <motion.div 
                    className="ml-2 h-1.5 w-1.5 bg-blue-500 rounded-full"
                    animate={{ 
                      scale: [1, 1.5, 1],
                      opacity: [1, 0.8, 1]
                    }}
                    transition={{ 
                      duration: 1.5, 
                      repeat: Infinity,
                      repeatType: "loop"
                    }}
                  />
                </div>
              </motion.div>
              
              {/* Latest scene info */}
              {taskData.result.processed_scenes.length > 0 && (
                <motion.div
                  variants={listItemVariant}
                  custom={1}
                >
                  <span className="font-medium block">Latest Scene:</span>
                  <motion.p 
                    className="italic mt-1 p-1.5 bg-black/10 dark:bg-white/5 rounded"
                    key={taskData.result.processed_scenes.length}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    &quot;{taskData.result.processed_scenes[taskData.result.processed_scenes.length - 1].text}&quot;
                  </motion.p>
                  
                  {/* Cache info */}
                  {taskData.result.cache_stats && (
                    <motion.p 
                      className="mt-1 text-green-600 dark:text-green-400"
                      variants={listItemVariant}
                      custom={2}
                    >
                      <span className="font-medium">Cache Savings:</span> 
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                      >
                        ${taskData.result.cache_stats.money_saved?.toFixed(3) || '0.00'}
                      </motion.span>
                    </motion.p>
                  )}
                </motion.div>
              )}
            </motion.div>
          )}
          
          {/* Assembly stage updates */}
          {taskData.progress >= 0.75 && (
            <motion.div 
              className="text-xs space-y-1"
              variants={staggerChildren}
            >
              <motion.div 
                className="flex items-center space-x-2"
                variants={listItemVariant}
                custom={0}
              >
                <span className="font-medium">Finalizing Video:</span>
                <span>Combining media assets...</span>
                <motion.div 
                  className="flex space-x-1"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <span className="h-1 w-1 bg-primary rounded-full"></span>
                  <span className="h-1 w-1 bg-primary rounded-full"></span>
                  <span className="h-1 w-1 bg-primary rounded-full"></span>
                </motion.div>
              </motion.div>
              
              {taskData.result.metadata && (
                <>
                  {taskData.result.metadata.duration && (
                    <motion.p
                      variants={listItemVariant}
                      custom={1}
                    >
                      <span className="font-medium">Final Duration:</span> {taskData.result.metadata.duration.toFixed(1)} seconds
                    </motion.p>
                  )}
                  
                  {taskData.result.metadata.resolution && (
                    <motion.p
                      variants={listItemVariant}
                      custom={2}
                    >
                      <span className="font-medium">Resolution:</span> {taskData.result.metadata.resolution}
                    </motion.p>
                  )}
                  
                  {/* Completion progress */}
                  <motion.div 
                    className="mt-2"
                    variants={listItemVariant}
                    custom={3}
                  >
                    <div className="w-full h-1 bg-muted rounded-full mb-1 overflow-hidden">
                      <motion.div 
                        className="h-full bg-green-500 rounded-full"
                        style={{ 
                          width: `${((taskData.progress - 0.75) / 0.25) * 100}%` 
                        }}
                        initial={{ width: 0 }}
                        animate={{ 
                          width: `${((taskData.progress - 0.75) / 0.25) * 100}%`
                        }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                    <motion.div 
                      className="flex justify-between items-center"
                      animate={{ opacity: 1 }}
                      initial={{ opacity: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      <span className="text-2xs text-muted-foreground">Almost there...</span>
                      <span className="text-2xs font-medium text-green-600">
                        {Math.round(((taskData.progress - 0.75) / 0.25) * 100)}% complete
                      </span>
                    </motion.div>
                  </motion.div>
                </>
              )}
            </motion.div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}