# Local Development Guide

This guide covers setting up the AI-Human Inventory Game for local development.

## Prerequisites

- Python 3.8 or higher
- pip or uv package manager
- An OpenAI API key

## Quick Start (Recommended)

### 1. Install Dependencies

From the `examples/fullstack_demo/` directory:

```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-key-here
USE_LOCAL_STORAGE=true
```

### 3. Run the Application

```bash
python main.py
```

This will:
- Start the FastAPI server on `http://localhost:8000`
- Automatically open your browser to the application
- Load environment variables from `.env`

The application is now running with local JSON storage. Game data will be saved to `data/game_runs.json`.

## Development with Auto-Reload

The application automatically enables hot-reload in development mode. Any changes to `backend/` or `frontend/` files will trigger a restart:

```bash
# Auto-reload is enabled by default in development
FULLSTACK_DEMO_RELOAD=1 python main.py
```

To disable auto-reload:

```bash
FULLSTACK_DEMO_RELOAD=0 python main.py
```

## Debugging

### View Application Logs

Logs are printed to the console with the following format:

```
[REQ] POST /api/create-run | Active: 1 | Total: 125
[RES] POST /api/create-run | Status: 200 | Time: 0.45s
```

### Enable Debug Logging

Add to your `.env`:

```bash
DEBUG=true
```

### Common Issues

**Issue**: `openai.AuthenticationError: Incorrect API key`
- **Solution**: Verify your `OPENAI_API_KEY` in `.env` is correct

**Issue**: Port 8000 already in use
- **Solution**: Change the port in `.env`:
  ```bash
  PORT=8001
  ```

**Issue**: Game runs not saving
- **Solution**: Ensure `data/` directory exists and is writable:
  ```bash
  mkdir -p data
  chmod 755 data
  ```

## Project Structure

```
fullstack_demo/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── simulation_current.py  # Game simulation logic
│   ├── supabase_client.py  # Supabase integration (optional)
│   └── token_verifier.py   # JWT token verification (optional)
├── frontend/
│   ├── index.html          # Main interface
│   ├── modeA.html          # Mode 1 interface
│   └── modeB.html          # Mode 2 interface
├── data/
│   └── game_runs.json      # Local storage (auto-created)
├── main.py                 # Entry point
├── .env.example            # Configuration template
└── requirements.txt        # Dependencies
```

## Testing the Application

### Test Local Storage Mode

1. Start the app with local storage:
   ```bash
   USE_LOCAL_STORAGE=true python main.py
   ```

2. Open `http://localhost:8000` in your browser

3. Start a new game run

4. Complete a game and verify data is saved in `data/game_runs.json`

### Test Different Game Modes

- **Mode 1 (Daily Feedback)**: Chat with the agent and make decisions
- **Mode 2 (Periodic Guidance)**: Agent runs automatically with guidance at intervals

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model to use |
| `USE_LOCAL_STORAGE` | No | `true` | Use local JSON storage instead of Supabase |
| `PORT` | No | `8000` | Server port |
| `HOST` | No | `0.0.0.0` | Server host |
| `FULLSTACK_DEMO_RELOAD` | No | `1` | Enable auto-reload in development |
| `DEBUG` | No | `false` | Enable debug logging |

## Next Steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for deploying to production
- See [../README.md](../README.md) for general usage information
