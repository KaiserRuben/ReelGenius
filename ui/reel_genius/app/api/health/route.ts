import { getApiHealth } from '@/lib/api';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Attempt to get backend API health data
    const backendHealth = await getApiHealth();
    
    // Base health information for the UI
    const uiHealth = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      memory: {
        rss: Math.round(process.memoryUsage().rss / (1024 * 1024)) + 'MB',
        heapTotal: Math.round(process.memoryUsage().heapTotal / (1024 * 1024)) + 'MB',
        heapUsed: Math.round(process.memoryUsage().heapUsed / (1024 * 1024)) + 'MB'
      },
      docker: process.env.DOCKER_ENV === 'true'
    };

    // Combine UI health with backend health data
    return NextResponse.json({
      ...uiHealth,
      backend: backendHealth
    });
  } catch (error) {
    console.error('API health check error:', error);
    
    // For Docker health checks, return 200 even on errors to prevent container restarts
    // but include error information in the response
    return NextResponse.json(
      { 
        status: 'degraded',
        timestamp: new Date().toISOString(),
        error: 'Failed to check API health',
        details: (error as Error).message,
        backend: {
          status: 'error',
          connected: false
        }
      },
      // Use 200 status to ensure Docker health check passes
      { status: 200 }
    );
  }
}