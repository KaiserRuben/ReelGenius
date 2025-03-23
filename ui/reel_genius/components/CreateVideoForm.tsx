'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { PlatformType } from '@/lib/api';
import ProgressIndicator from './ProgressIndicator';
import VideoPlayer from './VideoPlayer';

// This component uses platformData in the JSX at line 43-63
export default function CreateVideoForm() {
  const router = useRouter();
  const [content, setContent] = useState('');
  const [platform, setPlatform] = useState<PlatformType>('tiktok');
  const [voiceGender, setVoiceGender] = useState<'male' | 'female'>('male');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<string | null>(null);
  const [taskProgress, setTaskProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [example, setExample] = useState(false);
  
  // Advanced settings
  const [colorScheme, setColorScheme] = useState('vibrant');
  const [textAnimation, setTextAnimation] = useState(true);
  const [motionEffects, setMotionEffects] = useState(true);
  const [transitionStyle, setTransitionStyle] = useState('smooth');
  const [voiceStyle, setVoiceStyle] = useState('natural');
  const [speakingRate, setSpeakingRate] = useState(1.1);
  const [imageStyle, setImageStyle] = useState('photorealistic');
  const [useMetaPrompting, setUseMetaPrompting] = useState(true);
  const [chainOfThought, setChainOfThought] = useState(true);
  const [fewShotExamples, setFewShotExamples] = useState(true);
  
  // File upload
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Polling interval
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  useEffect(() => {
    return () => {
      // Cleanup interval on unmount
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
  
  useEffect(() => {
    // Check if example should be loaded
    if (example) {
      setContent(`# Quantum Computing Explained

Quantum computing leverages the principles of quantum mechanics to process information in ways that classical computers cannot. Unlike traditional bits that exist in a state of either 0 or 1, quantum bits or "qubits" can exist in multiple states simultaneously through a property called superposition.

Another key quantum property is entanglement, where qubits become interconnected and the state of one qubit instantaneously affects another, regardless of distance. This allows quantum computers to perform certain calculations exponentially faster than classical computers.

Potential applications include:
- Breaking current encryption methods
- Drug discovery and molecular modeling
- Optimization problems in logistics and finance
- Advanced AI and machine learning

While still in early stages, quantum computing promises to revolutionize computing as we know it.`);
      setExample(false);
    }
  }, [example]);
  
  const startPolling = (id: string) => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    // Set up polling every 3 seconds
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/status/${id}`);
        if (!response.ok) throw new Error('Failed to fetch status');
        
        const data = await response.json();
        setTaskStatus(data.status);
        setTaskProgress(data.progress || 0);
        
        // If completed, stop polling and set video URL
        if (data.status === 'completed' && data.result?.video_path) {
          clearInterval(interval);
          setVideoUrl(`/api/video/${id}`);
        }
        
        // If failed, stop polling and show error
        if (data.status === 'failed') {
          clearInterval(interval);
          setError(data.error || 'Video generation failed');
        }
      } catch (error) {
        console.error('Status polling error:', error);
      }
    }, 3000);
    
    intervalRef.current = interval;
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
      
      // Read the file content
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setContent(event.target.result as string);
        }
      };
      reader.readAsText(e.target.files[0]);
    }
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!content || content.trim().length < 10) {
      setError('Content must be at least 10 characters long');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Build config overrides
      const config_overrides = {
        visual: {
          color_scheme: colorScheme,
          text_animation: textAnimation,
          motion_effects: motionEffects,
          transition_style: transitionStyle
        },
        tts: {
          voice_style: voiceStyle,
          speaking_rate: speakingRate
        },
        image_gen: {
          style: imageStyle,
          candidates_per_prompt: 1
        },
        llm: {
          use_meta_prompting: useMetaPrompting,
          chain_of_thought: chainOfThought,
          few_shot_examples: fewShotExamples
        }
      };
      
      // Submit generation request
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          platform,
          voice_gender: voiceGender,
          config_overrides
        }),
      });
      
      if (!response.ok) {
        throw new Error('Generation request failed');
      }
      
      const data = await response.json();
      setTaskId(data.task_id);
      setTaskStatus('queued');
      setTaskProgress(0);
      
      // Start polling for updates
      startPolling(data.task_id);
      
    } catch (error) {
      console.error('Submission error:', error);
      setError((error as Error).message || 'Failed to submit generation request');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleReset = () => {
    setTaskId(null);
    setTaskStatus(null);
    setTaskProgress(0);
    setVideoUrl(null);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };
  
  const handleUseExample = () => {
    setExample(true);
  };

  return (
    <div className="card p-6">
      {/* Show the form when no task is in progress */}
      {!taskId && (
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label htmlFor="platform" className="block text-sm font-medium mb-1">
                Select Platform
              </label>
              <select
                id="platform"
                value={platform}
                onChange={(e) => setPlatform(e.target.value as PlatformType)}
                disabled={isLoading}
                className="w-full p-2 border border-input rounded-md bg-transparent"
              >
                <option value="tiktok">TikTok</option>
                <option value="youtube_shorts">YouTube Shorts</option>
                <option value="instagram_reels">Instagram Reels</option>
                <option value="general">General</option>
              </select>
            </div>
            
            <div>
              <label htmlFor="voiceGender" className="block text-sm font-medium mb-1">
                Voice Gender
              </label>
              <select
                id="voiceGender"
                value={voiceGender}
                onChange={(e) => setVoiceGender(e.target.value as 'male' | 'female')}
                disabled={isLoading}
                className="w-full p-2 border border-input rounded-md bg-transparent"
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
          </div>
          
          <div className="mb-4">
            <label htmlFor="content" className="block text-sm font-medium mb-1">
              Content
            </label>
            <div className="flex gap-2 mb-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="button-secondary text-xs py-1"
              >
                Upload File
              </button>
              <button
                type="button"
                onClick={handleUseExample}
                className="button-secondary text-xs py-1"
              >
                Use Example
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".txt,.md"
                className="hidden"
              />
            </div>
            <textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              disabled={isLoading}
              className="w-full h-64 p-3 border border-input rounded-md bg-transparent"
              placeholder="Paste your article, script, or any content you want to transform into a video..."
            />
          </div>
          
          <div className="mb-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-primary flex items-center gap-1"
            >
              <span>{showAdvanced ? '▼' : '►'}</span>
              <span>Advanced Settings</span>
            </button>
            
            {showAdvanced && (
              <div className="mt-3 p-3 bg-accent/40 rounded-md">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3 text-sm">
                  {/* Visual Settings */}
                  <div>
                    <h3 className="font-medium mb-2">Visual Settings</h3>
                    <div className="space-y-2">
                      <div>
                        <label htmlFor="colorScheme" className="block text-xs mb-1">
                          Color Scheme
                        </label>
                        <select
                          id="colorScheme"
                          value={colorScheme}
                          onChange={(e) => setColorScheme(e.target.value)}
                          className="w-full p-1.5 text-xs border border-input rounded-md bg-transparent"
                        >
                          <option value="vibrant">Vibrant</option>
                          <option value="muted">Muted</option>
                          <option value="professional">Professional</option>
                          <option value="dark">Dark</option>
                          <option value="light">Light</option>
                        </select>
                      </div>
                      
                      <div className="flex justify-between">
                        <label htmlFor="textAnimation" className="text-xs">
                          Text Animation
                        </label>
                        <input
                          type="checkbox"
                          id="textAnimation"
                          checked={textAnimation}
                          onChange={(e) => setTextAnimation(e.target.checked)}
                          className="rounded-sm"
                        />
                      </div>
                      
                      <div className="flex justify-between">
                        <label htmlFor="motionEffects" className="text-xs">
                          Motion Effects (Ken Burns)
                        </label>
                        <input
                          type="checkbox"
                          id="motionEffects"
                          checked={motionEffects}
                          onChange={(e) => setMotionEffects(e.target.checked)}
                          className="rounded-sm"
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="transitionStyle" className="block text-xs mb-1">
                          Transitions
                        </label>
                        <select
                          id="transitionStyle"
                          value={transitionStyle}
                          onChange={(e) => setTransitionStyle(e.target.value)}
                          className="w-full p-1.5 text-xs border border-input rounded-md bg-transparent"
                        >
                          <option value="smooth">Smooth</option>
                          <option value="sharp">Sharp</option>
                          <option value="creative">Creative</option>
                        </select>
                      </div>
                    </div>
                  </div>
                  
                  {/* Audio & Image Settings */}
                  <div>
                    <h3 className="font-medium mb-2">Audio & Image Settings</h3>
                    <div className="space-y-2">
                      <div>
                        <label htmlFor="voiceStyle" className="block text-xs mb-1">
                          Voice Style
                        </label>
                        <select
                          id="voiceStyle"
                          value={voiceStyle}
                          onChange={(e) => setVoiceStyle(e.target.value)}
                          className="w-full p-1.5 text-xs border border-input rounded-md bg-transparent"
                        >
                          <option value="natural">Natural</option>
                          <option value="enthusiastic">Enthusiastic</option>
                          <option value="serious">Serious</option>
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="speakingRate" className="block text-xs mb-1">
                          Speaking Rate: {speakingRate}
                        </label>
                        <input
                          type="range"
                          id="speakingRate"
                          min="0.8"
                          max="1.3"
                          step="0.1"
                          value={speakingRate}
                          onChange={(e) => setSpeakingRate(parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="imageStyle" className="block text-xs mb-1">
                          Image Style
                        </label>
                        <select
                          id="imageStyle"
                          value={imageStyle}
                          onChange={(e) => setImageStyle(e.target.value)}
                          className="w-full p-1.5 text-xs border border-input rounded-md bg-transparent"
                        >
                          <option value="photorealistic">Photorealistic</option>
                          <option value="3d_render">3D Render</option>
                          <option value="cartoon">Cartoon</option>
                          <option value="sketch">Sketch</option>
                          <option value="painting">Painting</option>
                        </select>
                      </div>
                    </div>
                  </div>
                  
                  {/* AI Model Settings */}
                  <div className="md:col-span-2 mt-2">
                    <h3 className="font-medium mb-2">AI Model Settings</h3>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="flex justify-between">
                        <label htmlFor="useMetaPrompting" className="text-xs">
                          Meta-Prompting
                        </label>
                        <input
                          type="checkbox"
                          id="useMetaPrompting"
                          checked={useMetaPrompting}
                          onChange={(e) => setUseMetaPrompting(e.target.checked)}
                          className="rounded-sm"
                        />
                      </div>
                      
                      <div className="flex justify-between">
                        <label htmlFor="chainOfThought" className="text-xs">
                          Chain of Thought
                        </label>
                        <input
                          type="checkbox"
                          id="chainOfThought"
                          checked={chainOfThought}
                          onChange={(e) => setChainOfThought(e.target.checked)}
                          className="rounded-sm"
                        />
                      </div>
                      
                      <div className="flex justify-between">
                        <label htmlFor="fewShotExamples" className="text-xs">
                          Few-Shot Examples
                        </label>
                        <input
                          type="checkbox"
                          id="fewShotExamples"
                          checked={fewShotExamples}
                          onChange={(e) => setFewShotExamples(e.target.checked)}
                          className="rounded-sm"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {error && (
            <div className="mb-4 p-3 bg-destructive/10 text-destructive border border-destructive/30 rounded-md text-sm">
              {error}
            </div>
          )}
          
          <button
            type="submit"
            disabled={isLoading || !content || content.trim().length < 10}
            className="button-primary w-full py-3"
          >
            {isLoading ? 'Submitting...' : '🚀 Generate Video'}
          </button>
        </form>
      )}
      
      {/* Show the task progress when a task is in progress */}
      {taskId && (
        <div>
          <h2 className="text-lg font-medium mb-4">Generation Progress</h2>
          
          {/* Progress indicator */}
          <div className="mb-6">
            <ProgressIndicator 
              progress={taskProgress} 
              status={taskStatus || 'queued'} 
              showLabel={true}
              height={10}
            />
          </div>
          
          {/* Task details */}
          <div className="mb-6 p-3 bg-accent/40 rounded-md">
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-sm">
              <span className="text-muted-foreground">Task ID:</span>
              <span className="font-mono">{taskId}</span>
              <span className="text-muted-foreground">Platform:</span>
              <span>{platform}</span>
              <span className="text-muted-foreground">Voice Gender:</span>
              <span>{voiceGender}</span>
              <span className="text-muted-foreground">Status:</span>
              <span className={
                taskStatus === 'completed' ? 'text-green-500' :
                taskStatus === 'failed' ? 'text-red-500' :
                taskStatus === 'running' ? 'text-blue-500' :
                'text-yellow-500'
              }>
                {taskStatus}
              </span>
            </div>
          </div>
          
          {/* Error message */}
          {error && (
            <div className="mb-6 p-3 bg-destructive/10 text-destructive border border-destructive/30 rounded-md text-sm">
              {error}
            </div>
          )}
          
          {/* Video player */}
          {videoUrl && (
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-3">Generated Video</h3>
              <VideoPlayer src={videoUrl} title="Generated Video" />
              <div className="mt-3 flex justify-center">
                <a
                  href={videoUrl}
                  download={`video-${taskId}.mp4`}
                  className="button-primary"
                >
                  Download Video
                </a>
              </div>
            </div>
          )}
          
          {/* Action buttons */}
          <div className="flex justify-between">
            <button
              onClick={handleReset}
              className="button-secondary"
            >
              {videoUrl ? 'Create Another Video' : 'Cancel'}
            </button>
            
            {videoUrl && (
              <button
                onClick={() => router.push(`/task/${taskId}`)}
                className="button-primary"
              >
                View Full Details
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}