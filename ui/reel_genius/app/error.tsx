'use client';

import { useEffect } from 'react';
import Link from 'next/link';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] text-center px-4">
      <h1 className="text-4xl font-bold mb-6">Something went wrong</h1>
      <p className="text-muted-foreground mb-4 max-w-md">
        We&#39;re sorry, but an unexpected error occurred. Our team has been notified.
      </p>
      
      <div className="p-4 mb-6 bg-destructive/10 text-destructive rounded-md max-w-md text-left">
        <details>
          <summary className="cursor-pointer font-medium">Error details</summary>
          <p className="mt-2 text-sm whitespace-pre-wrap">
            {error.message || 'Unknown error'}
          </p>
        </details>
      </div>
      
      <div className="flex gap-4">
        <button
          onClick={reset}
          className="button-primary"
        >
          Try again
        </button>
        <Link href="/" className="button-secondary">
          Go Home
        </Link>
      </div>
    </div>
  );
}