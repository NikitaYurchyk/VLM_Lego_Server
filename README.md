# VLM Agent Server

FastAPI server with vision-language models for Lego assembly assistance.

## Prerequisites

- Python 3.8+
- LM Studio (for local model support) - Download from [lmstudio.ai](https://lmstudio.ai)

## Setup

1. Create Python virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure `.env` file with required environment variables:

Required variables:
```
# AI API Keys (at least one is required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenRouter (alternative LLM provider)
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# LM Studio (local model provider)
DEFAULT_PROVIDER=lm_studio
DEFAULT_MODEL=lm-studio/gemma-3-27b-tools-mlx

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info
ACCESS_LOG=true

# LEGO Dataset Configuration
LEGO_SET_NAME=lego-60399-green-race-car-readscr
LEGO_SET_DISPLAY_NAME=Green Race Car
LEGO_BASE_URL=https://legoaudioinstructions.com/wp-content/themes/mtt-wordpress-theme/assets/manual/manual-images
LEGO_OUTPUT_FOLDER=json_file_updated
MAX_CONCURRENT_DOWNLOADS=5

# Site Information
YOUR_SITE_URL=http://localhost:8000
YOUR_SITE_NAME=AR LEGO Assembly
```

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## Features

- Chat interface for assembly instructions
- Image upload and analysis capabilities
- Lego instruction database (ChromaDB)
- REST API endpoints for AI-powered assistance