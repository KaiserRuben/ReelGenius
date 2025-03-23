# ReelGenius

ReelGenius is an AI-powered video generation platform that transforms text content into engaging short-form videos optimized for social media platforms like TikTok, YouTube Shorts, and Instagram Reels.

## Key Features

- **Multi-Platform Support**: Generate videos optimized for TikTok, YouTube Shorts, Instagram Reels, or general purposes
- **AI-Driven Generation**: Uses AI for script development, image generation, and voice synthesis
- **Custom Visual Styles**: Adjust colors, animations, transitions, and effects
- **Voice Customization**: Control voice style, speaking rate, and emphasis
- **Asynchronous Processing**: Task queue system for efficient video generation
- **Modern React UI**: Clean, responsive interface for video creation and management

## System Architecture

ReelGenius consists of three main components:

1. **Video Generation API**: Python FastAPI backend for video creation
2. **Task Processing System**: Celery worker for asynchronous video generation
3. **Web UI**: Next.js frontend for user interaction

## Requirements

- Python 3.10+
- Node.js 18+
- MongoDB
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

## Configuration

ReelGenius can be configured through environment variables and the configuration files in the `video_generator/config` directory.

### Platform-Specific Configuration

Each platform (TikTok, YouTube Shorts, etc.) has specific settings that control video dimensions, duration, and styling. These can be found in the `config.py` file.

### Advanced Configuration Options

The following configuration options can be set in the UI or through the API:

- **Visual Settings**: Color scheme, text animation, motion effects, transition style
- **Audio Settings**: Voice style, speaking rate
- **Image Settings**: Style (photorealistic, 3D render, cartoon, etc.)
- **AI Model Settings**: Meta-prompting, chain of thought, few-shot examples

## Development

### Project Structure

```
ReelGenius/
├── video_generator/       # Backend Python code
│   ├── api.py             # FastAPI endpoints
│   ├── tasks.py           # Celery task definitions
│   ├── config/            # Configuration
│   ├── models/            # LLM, TTS, and image generation models
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

## License

[MIT License](LICENSE)

## Acknowledgements

- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [DeepSeek](https://deepseek.ai/) for LLM capabilities
- [ElevenLabs](https://elevenlabs.io/) for text-to-speech
- [Stability AI](https://stability.ai/) for image generation