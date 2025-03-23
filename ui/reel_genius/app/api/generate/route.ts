import { generateVideo } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { content, platform, config_overrides } = body;
    
    if (!content || content.trim().length < 10) {
      return NextResponse.json(
        { error: 'Content must be at least 10 characters long' },
        { status: 400 }
      );
    }
    
    const response = await generateVideo({
      content,
      platform,
      config_overrides
    });
    
    return NextResponse.json(response);
  } catch (error) {
    console.error('Video generation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate video', details: (error as Error).message },
      { status: 500 }
    );
  }
}