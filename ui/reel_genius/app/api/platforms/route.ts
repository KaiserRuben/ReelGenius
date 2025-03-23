import { getPlatforms } from '@/lib/api';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const platformData = await getPlatforms();
    return NextResponse.json(platformData);
  } catch (error) {
    console.error('API platforms fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch platforms', details: (error as Error).message },
      { status: 500 }
    );
  }
}