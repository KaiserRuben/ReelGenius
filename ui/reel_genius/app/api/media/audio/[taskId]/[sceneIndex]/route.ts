import { getSceneAudioUrl } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { taskId: string; sceneIndex: string } }
) {
  try {
    const { taskId, sceneIndex } = params;
    
    const audioUrl = getSceneAudioUrl(taskId, sceneIndex === 'hook' ? 'hook' : parseInt(sceneIndex));
    
    const response = await fetch(audioUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch audio: ${response.status}`);
    }
    
    const contentType = response.headers.get('content-type') || 'audio/mpeg';
    const contentLength = response.headers.get('content-length');
    
    const buffer = await response.arrayBuffer();
    
    const headers = new Headers();
    headers.set('Content-Type', contentType);
    if (contentLength) headers.set('Content-Length', contentLength);
    headers.set('Cache-Control', 'public, max-age=3600');
    
    return new NextResponse(buffer, {
      status: 200,
      headers
    });
  } catch (error) {
    console.error('Audio fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch audio', details: (error as Error).message },
      { status: 500 }
    );
  }
}