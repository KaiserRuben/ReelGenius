'use client';

import { useState } from 'react';
import Image from 'next/image';

export interface MediaItem {
  type: 'image' | 'audio';
  url: string;
  title?: string;
  index: number;
}

interface MediaGalleryProps {
  items: MediaItem[];
  taskId: string;
}

export default function MediaGallery({ items, taskId }: MediaGalleryProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  
  const handleItemClick = (index: number) => {
    setActiveIndex(index);
    
    // Stop any playing audio
    if (audioElement) {
      audioElement.pause();
      setIsPlaying(false);
    }
  };
  
  const handleAudioPlay = (audioUrl: string) => {
    if (audioElement) {
      if (isPlaying) {
        audioElement.pause();
        setIsPlaying(false);
      } else {
        audioElement.play();
        setIsPlaying(true);
      }
    } else {
      const newAudio = new Audio(audioUrl);
      newAudio.onended = () => setIsPlaying(false);
      newAudio.onpause = () => setIsPlaying(false);
      newAudio.onplay = () => setIsPlaying(true);
      setAudioElement(newAudio);
      newAudio.play();
      setIsPlaying(true);
    }
  };
  
  const activeItem = items[activeIndex];

  return (
    <div className="media-gallery">
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {/* Thumbnails sidebar */}
        <div className="md:col-span-1 overflow-auto max-h-[500px] flex flex-row md:flex-col gap-2 p-2 border border-border rounded-lg">
          {items.map((item, index) => (
            <div
              key={index}
              className={`
                cursor-pointer rounded-md overflow-hidden border-2 transition-all
                ${activeIndex === index ? 'border-primary' : 'border-transparent hover:border-gray-300'}
              `}
              onClick={() => handleItemClick(index)}
            >
              {item.type === 'image' ? (
                <div className="relative h-20 w-32 md:w-full">
                  <Image
                    src={item.url}
                    alt={item.title || `Scene ${index + 1}`}
                    fill
                    sizes="(max-width: 768px) 100px, 150px"
                    className="object-cover"
                  />
                </div>
              ) : (
                <div className="flex items-center justify-center p-2 bg-accent h-20">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="w-8 h-8 text-foreground/70"
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <path d="M12 2a10 10 0 1 0 10 10"></path>
                    <path d="M12 12m-1 0a1 1 0 1 0 2 0a1 1 0 1 0 -2 0"></path>
                    <path d="M7 12a5 5 0 0 1 5 -5"></path>
                    <path d="M12 7v5l3 3"></path>
                  </svg>
                </div>
              )}
              <div className="text-xs p-1 text-center truncate">
                {item.title || `Scene ${index + 1}`}
              </div>
            </div>
          ))}
        </div>
        
        {/* Main display area */}
        <div className="md:col-span-4 border border-border rounded-lg overflow-hidden">
          {activeItem && (
            <div className="p-4">
              <h3 className="text-lg font-medium mb-3">
                {activeItem.title || `Scene ${activeIndex + 1}`}
              </h3>
              
              {activeItem.type === 'image' ? (
                <div className="relative w-full h-[300px] md:h-[400px] bg-accent rounded-lg overflow-hidden">
                  <Image
                    src={activeItem.url}
                    alt={activeItem.title || `Scene ${activeIndex + 1}`}
                    fill
                    sizes="(max-width: 768px) 100%, 800px"
                    className="object-contain"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[300px] md:h-[400px] bg-accent/50 rounded-lg p-8">
                  <button
                    onClick={() => handleAudioPlay(activeItem.url)}
                    className="w-24 h-24 rounded-full bg-primary/10 flex items-center justify-center hover:bg-primary/20 transition-colors"
                  >
                    <svg 
                      xmlns="http://www.w3.org/2000/svg" 
                      className="w-12 h-12 text-primary"
                      viewBox="0 0 24 24" 
                      fill="none" 
                      stroke="currentColor" 
                      strokeWidth="2" 
                      strokeLinecap="round" 
                      strokeLinejoin="round"
                    >
                      {isPlaying ? (
                        <>
                          <path d="M6 4h4"></path>
                          <path d="M6 8h4"></path>
                          <path d="M6 12h4"></path>
                          <path d="M14 4h4"></path>
                          <path d="M14 8h4"></path>
                          <path d="M14 12h4"></path>
                          <path d="M2 20h20"></path>
                        </>
                      ) : (
                        <>
                          <path d="M7 4v16c0 0 10-7 10-8s-10-8-10-8z"></path>
                        </>
                      )}
                    </svg>
                  </button>
                  <p className="mt-4 text-center text-sm text-foreground/70">
                    {isPlaying ? 'Now playing audio...' : 'Click to play audio'}
                  </p>
                  <a
                    href={activeItem.url}
                    download={`audio-${taskId}-${activeIndex}.mp3`}
                    className="mt-4 button-secondary text-sm"
                  >
                    Download Audio
                  </a>
                </div>
              )}
              
              <div className="flex justify-between mt-4">
                <button
                  onClick={() => setActiveIndex(Math.max(0, activeIndex - 1))}
                  disabled={activeIndex === 0}
                  className="px-3 py-1.5 rounded-md border border-border disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => setActiveIndex(Math.min(items.length - 1, activeIndex + 1))}
                  disabled={activeIndex === items.length - 1}
                  className="px-3 py-1.5 rounded-md border border-border disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}