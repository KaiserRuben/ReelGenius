# ReelGenius UI

A modern Next.js frontend for the ReelGenius AI video generation platform.

## Features

- Create AI-powered videos from text content
- Support for multiple platforms (TikTok, YouTube Shorts, Instagram Reels)
- Advanced customization options for video creation
- Real-time progress tracking
- Video preview and download capabilities

## Tech Stack

- **Framework**: Next.js 15.2.3
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Custom components built from scratch
- **API Integration**: Server components + Route handlers

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- ReelGenius backend API running

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ui/reel_genius
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   # Create a .env.local file with:
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

### Development

Run the development server:
```bash
npm run dev
```

The application will be available at http://localhost:3000.

### Building for Production

Build the application:
```bash
npm run build
```

Start the production server:
```bash
npm run start
```

## Docker Deployment

The UI can be run as a Docker container alongside the ReelGenius backend:

```bash
# Build the Docker image
docker build -t reel-genius-ui .

# Run the container
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://localhost:8000 reel-genius-ui
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | URL of the ReelGenius backend API | `http://localhost:8000` |

## API Endpoints

The UI proxies requests to the ReelGenius backend API. The following endpoints are available:

- `GET /api/health` - Get API health status
- `GET /api/platforms` - Get supported platforms and configurations
- `POST /api/generate` - Generate a video from content
- `GET /api/status/[taskId]` - Get the status of a generation task
- `GET /api/video/[taskId]` - Get the generated video
- `GET /api/tasks` - List all generation tasks

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request