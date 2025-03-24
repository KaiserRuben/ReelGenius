# ReelGenius

ReelGenius is an AI-powered video generation platform that transforms text content into engaging short-form videos optimized for social media platforms like TikTok, YouTube Shorts, and Instagram Reels.

## Key Features

- **Multi-Platform Support**: Generate videos optimized for TikTok, YouTube Shorts, Instagram Reels, or general purposes
- **AI-Driven Generation**: Uses AI for script development, image generation, and voice synthesis
- **Custom Visual Styles**: Adjust colors, animations, transitions, and effects
- **Voice Customization**: Control voice style, speaking rate, and emphasis
- **Asynchronous Processing**: Task queue system for efficient video generation
- **Modern React UI**: Clean, responsive interface for video creation and management
- **Semantic Caching**: Intelligent caching with semantic similarity for image generation to reduce API costs
- **Algorithm Optimization**: Advanced prompting system optimized for 2025 social media algorithms

## System Architecture

ReelGenius consists of three main components:

1. **Video Generation API**: Python FastAPI backend for video creation
2. **Task Processing System**: Celery worker for asynchronous video generation
3. **Web UI**: Next.js frontend for user interaction

## Requirements

- Python 3.10+
- Node.js 18+
- PostgreSQL
- Redis
- FFmpeg with AV1 support
- Docker and Docker Compose (for containerized deployment)

## API Keys Required

ReelGenius requires API keys for the following services:

- **DeepSeek API** (`DEEPSEEK_API_KEY`): For LLM capabilities
- **ElevenLabs API** (`ELEVENLABS_API_KEY`): For text-to-speech
- **Stability AI API** (`STABILITY_API_KEY`): For image generation

## Getting Started

### Quick Start with Docker

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ReelGenius
   ```

2. Create a `.env` file with your API keys:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   STABILITY_API_KEY=your_stability_api_key
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the UI at http://localhost:3000 and the API at http://localhost:8000

### Manual Setup

#### Backend (Python API)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export DEEPSEEK_API_KEY=your_deepseek_api_key
   export ELEVENLABS_API_KEY=your_elevenlabs_api_key
   export STABILITY_API_KEY=your_stability_api_key
   ```

4. Start the API server:
   ```bash
   uvicorn video_generator.api:app --host 0.0.0.0 --port 8000
   ```

5. Start the worker (in a separate terminal):
   ```bash
   celery -A video_generator.tasks worker --loglevel=info
   ```

#### Frontend (Next.js UI)

1. Navigate to the UI directory:
   ```bash
   cd ui/reel_genius
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment:
   ```bash
   echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

## API Documentation

The API is available at http://localhost:8000 when running the server. Documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate a video from text content |
| `/generate-from-file` | POST | Generate a video from an uploaded file |
| `/status/{task_id}` | GET | Get the status of a generation task |
| `/video/{task_id}` | GET | Get the generated video |
| `/metadata/{task_id}` | GET | Get video metadata |
| `/platforms` | GET | List supported platforms and configurations |
| `/tasks` | GET | List all generation tasks |
| `/health` | GET | Check API health status |
| `/cache/stats` | GET | Get semantic cache statistics |

## Configuration

ReelGenius can be configured through environment variables and the configuration files in the `video_generator/config` directory.

### Database Schema

ReelGenius uses PostgreSQL for data persistence with the following structure:

#### Primary Tables
- **users**: User accounts, authentication, and preferences
- **tasks**: Video generation tasks and their status
- **video_templates**: Templates for different video formats by platform
- **media_assets**: Images, audio files, and videos associated with tasks
- **cache_entries**: Semantic cache for generated content
- **prompt_templates**: Templates for LLM prompts by type
- **platform_accounts**: Social media account connections
- **usage_statistics**: Usage tracking for analytics and billing

#### Relationships
- A user can have many tasks, platform accounts, and usage statistics
- A task belongs to a user (optional) and can use a video template
- A task can have many media assets
- Templates can be public or associated with specific users

The database is automatically initialized when the containers start, and includes indexes for optimized queries.

### Health Check System

ReelGenius implements a robust health check system for reliable container orchestration:

- **Backend API Health**: Endpoint at `/health` provides detailed system status
- **UI Health System**: Multi-layered approach with fallback mechanisms:
  - **App Router**: `/api/health` - Comprehensive health status with backend connectivity  
  - **Pages Router**: `/api/healthcheck` - Lightweight, reliable health endpoint
  - **Static File**: `/healthcheck.txt` - Ultimate fallback option

The UI container uses a sophisticated health check script (`healthcheck.js`) that tries all three methods in sequence, ensuring maximum reliability even during startup or when one system is temporarily unavailable.

> **Note**: Following Next.js best practices, we ensure no path conflicts between App Router and Pages Router by using different paths for each (`/api/health` and `/api/healthcheck` respectively).

### Platform-Specific Configuration

Each platform (TikTok, YouTube Shorts, etc.) has specific settings that control video dimensions, duration, and styling. These can be found in the `config.py` file.

### Advanced Configuration Options

The following configuration options can be set in the UI or through the API:

- **Visual Settings**: Color scheme, text animation, motion effects, transition style
- **Audio Settings**: Voice style, speaking rate
- **Image Settings**: Style (photorealistic, 3D render, cartoon, etc.)
- **AI Model Settings**: Meta-prompting, chain of thought, few-shot examples

### Semantic Caching System

ReelGenius implements a semantic caching system that dramatically reduces API costs for image generation:

- **Similarity-Based Caching**: Uses embeddings to find similar previous prompts, avoiding redundant API calls
- **Cost Reduction**: Tracks and reports money saved by reusing semantically similar content
- **Configurable Thresholds**: Adjustable similarity thresholds to control cache hit rates
- **Redis Integration**: Scalable performance with Redis backend (with in-memory fallback)
- **Detailed Statistics**: Access cache performance metrics via the `/cache/stats` endpoint
- **30-Day Persistence**: Cache entries expire after 30 days to maintain freshness

The semantic cache stores embeddings of image generation prompts and retrieves cached results when similar prompts are encountered, significantly reducing the number of API calls to expensive image generation services.

### Algorithm Optimization Framework

ReelGenius features a sophisticated algorithm optimization system for maximizing engagement:

- **Algorithmic Psychology**: Prompting templates designed around psychological engagement triggers
- **Retention Architecture**: Strategic open-loop design prevents viewer dropoff at key exit points (15%, 35%, 55%, 85%)
- **Cognitive Pattern Interrupts**: Hook designs engineered to create immediate psychological commitment
- **Visual Optimization**: Image generation prompts specifically designed for algorithm-favored attributes
- **Metadata Engineering**: Reverse-engineered metadata structures for maximum algorithmic distribution
- **Identity Reinforcement**: Content framing that creates powerful sharing impulses through viewer identity alignment
- **Exit-Prevention**: Specialized techniques to make scrolling away feel psychologically incomplete

The algorithm optimization framework constantly evolves based on the latest research in algorithmic psychology and platform performance metrics.

## Development

### Project Structure

```
ReelGenius/
├── video_generator/       # Backend Python code
│   ├── api.py             # FastAPI endpoints
│   ├── tasks.py           # Celery task definitions
│   ├── config/            # Configuration
│   ├── models/            # LLM, TTS, and image generation models
│   ├── semantic_cache/    # Similarity-based caching system
│   └── pipeline/          # Video generation pipeline
├── ui/
│   └── reel_genius/       # Next.js frontend
│       ├── app/           # Next.js app directory
│       ├── components/    # React components
│       └── lib/           # Utility functions
├── data/                  # Data storage (mounted volume)
└── output/                # Generated videos (mounted volume)
```

### Testing

Run backend tests:
```bash
pytest
```

Run frontend tests:
```bash
cd ui/reel_genius
npm test
```

## Acknowledgements

- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [DeepSeek](https://deepseek.ai/) for LLM capabilities
- [ElevenLabs](https://elevenlabs.io/) for text-to-speech
- [Stability AI](https://stability.ai/) for image generation