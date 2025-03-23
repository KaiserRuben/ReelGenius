import { getTasks } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get('status') || undefined;
    const platform = searchParams.get('platform') as any || undefined;
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