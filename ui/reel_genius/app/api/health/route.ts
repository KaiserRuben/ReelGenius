import { getApiHealth } from '@/lib/api';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const healthData = await getApiHealth();
    return NextResponse.json(healthData);
  } catch (error) {
    console.error('API health check error:', error);
    return NextResponse.json(
      { error: 'Failed to check API health', details: (error as Error).message },
      { status: 500 }
    );
  }
}