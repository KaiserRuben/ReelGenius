// Determine whether we're running in Docker
const isDebug = process.env.NEXT_PUBLIC_DEBUG === 'true';

// Use the Docker service name if in Docker environment
// Try multiple possible URLs for Docker connectivity
const backendUrls = [
      'http://app:8000',                // Docker service name
      'http://reelgenius-backend:8000', // Container name
      'http://localhost:8000',          // Host machine's localhost
      'http://host.docker.internal:8000' // Special Docker DNS for host machine
    ]

if (isDebug) {
  console.log('Debug mode enabled');
  console.log('Backend URLs to try:', backendUrls);
}

// Initialize API_URL with the first option
let API_URL = backendUrls[0];

import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ taskId: string }> }
) {
  try {
    const { taskId } = await params;
    
    if (!taskId) {
      return NextResponse.json(
        { error: 'Task ID is required' },
        { status: 400 }
      );
    }
    
    if (isDebug) {
      console.log(`Attempting to fetch status for task ${taskId}`);
    }
    
    let response = null;
    let lastError = null;

    // Try all backend URLs until one works
    for (const backendUrl of backendUrls) {
      try {
        if (isDebug) {
          console.log(`Trying backend URL: ${backendUrl}/status/${taskId}`);
        }
        
        // Direct call to the backend API
        response = await fetch(`${backendUrl}/status/${taskId}`);
        
        if (isDebug) {
          console.log(`Response received from ${backendUrl} with status: ${response.status}`);
        }
        
        // If we got a successful response, update the API_URL for future calls
        if (response.ok) {
          API_URL = backendUrl;
          if (isDebug) {
            console.log(`Success! Using ${API_URL} for future requests.`);
          }
          break;
        } else {
          console.error(`Error from ${backendUrl}: ${response.status}`);
          // Continue to try next URL if this one didn't work
        }
      } catch (error: unknown) {
        lastError = error;
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error(`Fetch error with ${backendUrl}:`, errorMessage);
      }
    }

    // If we've tried all URLs and none worked
    if (!response) {
      console.error('All backend URLs failed');
      const errorDetails = lastError instanceof Error ? lastError.message : 'Unknown error';
      throw new Error(`Failed to connect to any backend API: ${errorDetails}`);
    }
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Backend API error:', errorData);
      throw new Error(`Backend API returned ${response.status}: ${errorData.error || response.statusText}`);
    }
    
    const statusData = await response.json();
    return NextResponse.json(statusData);
  } catch (error) {
    console.error('Task status fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch task status', details: (error as Error).message },
      { status: 500 }
    );
  }
}