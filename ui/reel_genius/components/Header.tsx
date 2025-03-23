'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Header() {
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'degraded' | 'error' | 'loading'>('loading');
  const pathname = usePathname();
  
  useEffect(() => {
    async function checkApiHealth() {
      try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error('Health check failed');
        
        const data = await res.json();
        setHealthStatus(data.status);
      } catch (error) {
        console.error('Health check error:', error);
        setHealthStatus('error');
      }
    }
    
    checkApiHealth();
    
    // Check health every 30 seconds
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);
  
  // Define status badge color
  const getStatusBadge = () => {
    switch (healthStatus) {
      case 'healthy':
        return (
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-green-500"></span>
            <span className="text-xs text-green-600 dark:text-green-400">API Connected</span>
          </span>
        );
      case 'degraded':
        return (
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-yellow-500"></span>
            <span className="text-xs text-yellow-600 dark:text-yellow-400">API Degraded</span>
          </span>
        );
      case 'error':
        return (
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500"></span>
            <span className="text-xs text-red-600 dark:text-red-400">API Error</span>
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-gray-500 animate-pulse"></span>
            <span className="text-xs text-gray-600 dark:text-gray-400">Checking API...</span>
          </span>
        );
    }
  };

  return (
    <header className="sticky top-0 z-10 border-b border-border bg-background/95 backdrop-blur">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center">
          <h1 className="text-xl font-semibold hidden sm:block">
            {pathname === '/' && 'Generate Video'}
            {pathname === '/history' && 'Video History'}
            {pathname === '/analytics' && 'Analytics'}
            {pathname.startsWith('/task/') && 'Task Details'}
          </h1>
        </div>
        
        <div className="flex items-center gap-4">
          {getStatusBadge()}
          
          <nav className="flex items-center space-x-1">
            <Link 
              href="/" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                pathname === '/' 
                  ? 'bg-primary/10 text-primary' 
                  : 'text-foreground/70 hover:text-foreground hover:bg-accent'
              }`}
            >
              Create
            </Link>
            <Link 
              href="/history" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                pathname === '/history' 
                  ? 'bg-primary/10 text-primary' 
                  : 'text-foreground/70 hover:text-foreground hover:bg-accent'
              }`}
            >
              History
            </Link>
            <Link 
              href="/analytics" 
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                pathname === '/analytics' 
                  ? 'bg-primary/10 text-primary' 
                  : 'text-foreground/70 hover:text-foreground hover:bg-accent'
              }`}
            >
              Analytics
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}