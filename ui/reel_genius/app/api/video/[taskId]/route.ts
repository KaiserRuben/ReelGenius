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
    
    if (isDebug) {
      console.log(`Attempting to fetch video for task ${taskId}`);
    }
    
    let response = null;
    let lastError = null;

    // Try all backend URLs until one works
    for (const backendUrl of backendUrls) {
      try {
        const videoUrl = `${backendUrl}/video/${taskId}`;
        
        if (isDebug) {
          console.log(`Trying backend URL: ${videoUrl}`);
        }
        
        // Direct call to the backend API
        response = await fetch(videoUrl);
        
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
      throw new Error(`Failed to fetch video: ${response.status}`);
    }
    
    const contentType = response.headers.get('content-type') || 'video/mp4';
    const contentLength = response.headers.get('content-length');
    const contentDisposition = response.headers.get('content-disposition');
    
    const buffer = await response.arrayBuffer();
    
    const headers = new Headers();
    headers.set('Content-Type', contentType);
    if (contentLength) headers.set('Content-Length', contentLength);
    if (contentDisposition) headers.set('Content-Disposition', contentDisposition);
    headers.set('Cache-Control', 'public, max-age=3600');
    
    return new NextResponse(buffer, {
      status: 200,
      headers
    });
  } catch (error) {
    console.error('Video fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch video', details: (error as Error).message },
      { status: 500 }
    );
  }
}