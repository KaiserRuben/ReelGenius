import { deleteTask } from '@/lib/api';
import { NextRequest, NextResponse } from 'next/server';

export async function DELETE(
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
    
    const result = await deleteTask(taskId);
    return NextResponse.json(result);
  } catch (error) {
    console.error('Task deletion error:', error);
    return NextResponse.json(
      { error: 'Failed to delete task', details: (error as Error).message },
      { status: 500 }
    );
  }
}