import type { NextConfig } from "next";

// Get the API URL based on the environment (production vs development)
const isDocker = process.env.DOCKER_ENV === 'true';
const apiUrl = isDocker ? 'http://app:8000' : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: apiUrl,
    DOCKER_ENV: isDocker ? 'true' : 'false',
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
