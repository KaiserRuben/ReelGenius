'use client';

import { useState, useEffect } from 'react';
import { PlatformsResponse } from '@/lib/api';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import Image from 'next/image';

export default function Sidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(true);
  const [platformData, setPlatformData] = useState<PlatformsResponse | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    async function fetchPlatforms() {
      try {
        const res = await fetch('/api/platforms');
        if (!res.ok) throw new Error('Failed to fetch platforms');
        const data = await res.json();
        setPlatformData(data);
      } catch (error) {
        console.error('Fetch platforms error:', error);
        setError('Failed to load platform data');
      }
    }
    
    fetchPlatforms();
  }, []);
  
  // For smaller screens
  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Mobile toggle button */}
      <button 
        className="fixed top-4 left-4 z-50 md:hidden bg-primary text-white p-2 rounded-md"
        onClick={toggleSidebar}
      >
        {isOpen ? '✕' : '☰'}
      </button>
      
      {/* Sidebar */}
      <aside className={`
        fixed top-0 left-0 z-40 h-screen bg-card border-r border-border transition-all duration-300 ease-in-out
        ${isOpen ? 'w-64' : 'w-0 -translate-x-full md:w-20 md:translate-x-0'} 
        md:relative md:translate-x-0
      `}>
        <div className="h-full flex flex-col overflow-y-auto">
          {/* Logo and title */}
          <div className="flex items-center justify-center h-16 p-4 border-b border-border">
            {isOpen ? (
              <h1 className="text-xl font-bold">ReelGenius</h1>
            ) : (
              <span className="text-2xl font-bold">🎬</span>
            )}
          </div>
          
          {/* Navigation */}
          <nav className="flex-1 p-4">
            <ul className="space-y-2">
              <li>
                <Link
                  href="/"
                  className={`flex items-center p-2 rounded-lg ${
                    pathname === '/' 
                      ? 'bg-primary/10 text-primary' 
                      : 'text-foreground/70 hover:text-foreground hover:bg-accent'
                  }`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22v-9"></path>
                    <path d="M15.4 19.2l-3.4 2.8-3.4-2.8"></path>
                    <path d="M8 22h8"></path>
                    <rect x="2" y="2" width="20" height="12" rx="2"></rect>
                    <path d="m14 8-4 3v-6l4 3z"></path>
                  </svg>
                  {isOpen && <span className="ml-3">Create Video</span>}
                </Link>
              </li>
              <li>
                <Link
                  href="/history"
                  className={`flex items-center p-2 rounded-lg ${
                    pathname === '/history' 
                      ? 'bg-primary/10 text-primary' 
                      : 'text-foreground/70 hover:text-foreground hover:bg-accent'
                  }`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a10 10 0 1 0 10 10"></path>
                    <path d="m16 8-4 4-2-2"></path>
                    <path d="M22 8m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"></path>
                    <path d="M18 4m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"></path>
                    <path d="M12 2m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"></path>
                    <path d="M8 4m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"></path>
                    <path d="M5 8m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0"></path>
                  </svg>
                  {isOpen && <span className="ml-3">History</span>}
                </Link>
              </li>
              <li>
                <Link
                  href="/analytics"
                  className={`flex items-center p-2 rounded-lg ${
                    pathname === '/analytics' 
                      ? 'bg-primary/10 text-primary' 
                      : 'text-foreground/70 hover:text-foreground hover:bg-accent'
                  }`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 3v18h18"></path>
                    <path d="M18 17V9"></path>
                    <path d="M13 17V5"></path>
                    <path d="M8 17v-3"></path>
                  </svg>
                  {isOpen && <span className="ml-3">Analytics</span>}
                </Link>
              </li>
            </ul>
          </nav>
          
          {/* Platform info */}
          {isOpen && platformData && (
            <div className="p-4 border-t border-border">
              <h3 className="text-sm font-semibold mb-2">Available Platforms</h3>
              <ul className="space-y-1 text-xs">
                {platformData.platforms?.map((platform: string) => (
                  <li key={platform} className="flex items-center">
                    <span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span>
                    <span>{platform.replace('_', ' ').toUpperCase()}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Footer */}
          <div className="p-4 text-xs text-muted-foreground border-t border-border">
            <p>ReelGenius AI Video Generator</p>
            <p className="text-xs mt-1">© Ruben Kaiser 2025</p>
          </div>
        </div>
      </aside>
      
      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
          onClick={toggleSidebar}
        />
      )}
    </>
  );
}