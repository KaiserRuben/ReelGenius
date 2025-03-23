import { getTaskStatus } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { taskId: string } }
) {
  try {
    const taskId = params.taskId;
    
    if (!taskId) {
      return NextResponse.json(
        { error: 'Task ID is required' },
        { status: 400 }
      );
    }
    
    const statusData = await getTaskStatus(taskId);
    return NextResponse.json(statusData);
  } catch (error) {
    console.error('Task status fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch task status', details: (error as Error).message },
      { status: 500 }
    );
  }
}