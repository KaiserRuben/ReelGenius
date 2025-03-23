import { getVideoUrl } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { taskId: string } }
) {
  try {
    const taskId = params.taskId;
    
    const videoUrl = getVideoUrl(taskId);
    
    const response = await fetch(videoUrl);
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