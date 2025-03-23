'use client';

import { useState } from 'react';

interface VideoPlayerProps {
  src: string;
  title?: string;
  poster?: string;
  width?: string | number;
  height?: string | number;
  controls?: boolean;
}

export default function VideoPlayer({
  src,
  title,
  poster,
  width = '100%',
  height = 'auto',
  controls = true
}: VideoPlayerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Handle errors
  const handleError = () => {
    setError('Failed to load video. Please try again later.');
    setIsLoading(false);
  };
  
  // Handle video loaded
  const handleLoaded = () => {
    setIsLoading(false);
  };
  
  return (
    <div className="video-player relative rounded-lg overflow-hidden bg-black">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-white">
          <div className="text-center p-4">
            <p className="text-red-400 mb-2">⚠️ {error}</p>
            <button
              onClick={() => window.open(src, '_blank')}
              className="button-primary text-sm"
            >
              Open Video Directly
            </button>
          </div>
        </div>
      )}
      
      <video
        src={src}
        poster={poster}
        controls={controls}
        className="w-full rounded-lg"
        style={{ width, height }}
        onLoadedData={handleLoaded}
        onError={handleError}
        playsInline
      >
        Your browser does not support the video tag.
      </video>
      
      {title && (
        <div className="p-2 text-center text-sm font-medium text-gray-700 dark:text-gray-300">
          {title}
        </div>
      )}
    </div>
  );
}