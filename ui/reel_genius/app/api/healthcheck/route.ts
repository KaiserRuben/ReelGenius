import { NextResponse } from 'next/server';

/**
 * Health check endpoint for the UI
 * Returns a simple 200 OK response with some basic system information
 */
export async function GET() {
  try {
    // Basic system info
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      environment: process.env.NODE_ENV || 'development',
      version: process.env.npm_package_version || 'unknown'
    };

    // Check connection to backend
    let backendStatus: 'connected' | 'error' = 'error';
    let backendError: string | null = null;
    
    try {
      // Use relative URL for client-side fetch
      const response = await fetch('/api/health', { 
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache' },
        next: { revalidate: 0 }
      });
      
      if (response.ok) {
        backendStatus = 'connected';
      } else {
        backendError = `Status: ${response.status} ${response.statusText}`;
      }
    } catch (error) {
      backendError = error instanceof Error ? error.message : String(error);
    }

    return NextResponse.json({
      ...health,
      backend: {
        status: backendStatus,
        error: backendError
      }
    });
  } catch (error) {
    // Even on error, return 200 for the healthcheck
    // but include the error details
    return NextResponse.json({
      status: 'degraded',
      error: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString()
    });
  }
}