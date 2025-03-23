import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  images: {
    domains: ['localhost', 'app'],
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
      {
        protocol: 'http',
        hostname: 'app',
        port: '8000',
        pathname: '/**',
      },
    ],
  },
  // Removed circular redirect that was causing ERR_TOO_MANY_REDIRECTS
};

export default nextConfig;
