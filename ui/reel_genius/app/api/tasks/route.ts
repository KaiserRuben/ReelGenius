import { getTasks, PlatformType } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

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

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get('status') || undefined;
    const platform = searchParams.get('platform') as PlatformType | undefined;
    const limit = parseInt(searchParams.get('limit') || '10');
    const skip = parseInt(searchParams.get('skip') || '0');
    const include_details = searchParams.get('include_details') === 'true';
    const include_media_info = searchParams.get('include_media_info') === 'true';
    
    const tasksData = await getTasks(
      status,
      platform,
      limit,
      skip,
      include_details,
      include_media_info
    );
    
    return NextResponse.json(tasksData);
  } catch (error) {
    console.error('Tasks fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch tasks', details: (error as Error).message },
      { status: 500 }
    );
  }
}

// Handle POST requests for video generation
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { content, platform, voice_gender, config_overrides } = body;
    
    if (!content || content.trim().length < 10) {
      return NextResponse.json(
        { error: 'Content must be at least 10 characters long' },
        { status: 400 }
      );
    }
    
    console.log(`Attempting to send request to backend at ${API_URL}/generate`);

    const requestPayload = JSON.stringify({
      content,
      platform,
      voice_gender,
      config_overrides
    });

    // Format for debugging
    console.log('Request payload:', {
      content: content.substring(0, 50) + (content.length > 50 ? '...' : ''),
      platform,
      voice_gender,
      config_overrides
    });

    let response = null;
    let lastError = null;

    // Try all backend URLs until one works
    for (const backendUrl of backendUrls) {
      try {
        console.log(`Trying backend URL: ${backendUrl}/generate`);
        
        // Server-side proxy to the backend API
        response = await fetch(`${backendUrl}/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: requestPayload,
        });
        
        console.log(`Response received from ${backendUrl} with status: ${response.status}`);
        
        // If we got a successful response, update the API_URL for future calls
        if (response.ok) {
          API_URL = backendUrl;
          console.log(`Success! Using ${API_URL} for future requests.`);
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
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Video generation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate video', details: (error as Error).message },
      { status: 500 }
    );
  }
}