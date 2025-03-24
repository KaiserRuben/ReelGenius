import {NextRequest, NextResponse} from 'next/server';

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
    console.log('Debug mode enabled in generate route');
    console.log('Backend URLs to try:', backendUrls);
}

// Initialize API_URL with the first option
let API_URL = backendUrls[0];

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const {content, platform, voice_gender, config_overrides} = body;

        if (!content || content.trim().length < 10) {
            return NextResponse.json(
                {error: 'Content must be at least 10 characters long'},
                {status: 400}
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

        // Parse response to ensure it's a proper StandardResponse
        const data = await response.json();
        
        // Verify response format
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid response format from backend');
        }
        
        // Ensure it follows StandardResponse format
        if (!('status' in data)) {
            // Convert to standard format if not already
            return NextResponse.json({
                status: response.ok ? 'success' : 'error',
                data: response.ok ? data : null,
                error: !response.ok ? 'Backend error' : undefined,
                message: response.ok ? 'Video generation request successful' : 'Video generation failed',
                meta: { 
                    raw_response: data,
                    timestamp: new Date().toISOString()
                }
            });
        }
        
        // If already in standard format, pass through
        return NextResponse.json(data);
    } catch (error) {
        console.error('Video generation error:', error);
        // Return in standard format
        return NextResponse.json({
            status: 'error',
            error: 'Failed to generate video',
            message: (error as Error).message,
            meta: { 
                timestamp: new Date().toISOString(),
                details: error instanceof Error ? error.stack : String(error)
            }
        }, {status: 500});
    }
}