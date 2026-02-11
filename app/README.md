# AI-Human Inventory Game

A fullstack web application for exploring human-AI collaboration in inventory management. This application combines an interactive web interface with LLM agents to create an engaging learning experience for inventory control and supply chain optimization.

This implementation uses FastAPI for the backend and supports both Supabase (for multi-user deployments) and local JSON storage (for local testing and deployment without external dependencies).

## Project Layout

```
app/
├── backend/
│   ├── app.py                 # FastAPI application and endpoints
│   ├── simulation_current.py  # Game simulation and environment logic
│   ├── storage.py             # Storage abstraction (JSON / Supabase)
│   ├── config.py              # Configuration management
│   ├── supabase_client.py     # Supabase integration (optional)
│   └── token_verifier.py      # JWT authentication (optional)
├── frontend/
│   ├── index.html             # Main interface
│   ├── modeA.html             # Mode 1 (Daily Feedback)
│   ├── modeB.html             # Mode 2 (Periodic Guidance)
│   └── modeC.html             # Mode 3 (Alternative)
├── data/
│   └── game_runs.json         # Local storage (auto-created)
├── docs/
│   ├── DEVELOPMENT.md         # Local development guide
│   ├── DEPLOYMENT.md          # Production deployment guide
│   └── CONFIGURATION.md       # Configuration reference
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Configuration template
└── README.md                 # This file
```

**Key Files:**
- `backend/storage.py` – Abstraction layer supporting both local JSON and Supabase storage
- `backend/config.py` – Centralized configuration management
- `backend/app.py` – FastAPI endpoints for game sessions and interactions
- `backend/simulation_current.py` – Game logic wrapping the vending machine environment
- `frontend/` – Interactive HTML interfaces for different game modes

## Setup Options

### Option 1: Local JSON Storage (Recommended for Getting Started)

For quick local testing and deployment without external dependencies, use JSON file storage:

1. Set the environment variable:
   ```bash
   export USE_LOCAL_STORAGE=true
   ```

2. Game data will be saved to `data/game_runs.json` automatically.

3. Continue to the **Installation & Running** section below.

### Option 2: Supabase Setup (For Multi-User Deployments)

For production deployments with multi-user support and cloud database:

1. Create a Supabase project and note the **Project URL**, **anon/public key**, and **service role key**.
2. Create a table `game_runs`:

   ```sql
   create table if not exists public.game_runs (
     id uuid primary key default uuid_generate_v4(),
     run_id uuid,
     user_id uuid not null,
     mode text not null,
     guidance_frequency int,
     final_reward numeric,
     log_text text,
     created_at timestamp with time zone default timezone('utc', now())
   );
   ```

3. Enable Row Level Security and add a policy allowing inserts via the service role key (or leave
   RLS disabled while experimenting).

4. In Supabase Authentication settings, ensure email/password signups are enabled.

5. Set the Supabase environment variables (see **Backend Configuration** below).

## Documentation

For detailed guides and references, see:

- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Local development setup, debugging, and project structure
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment guides (Render, Docker, local server, Supabase)
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Complete environment variable reference and configuration options

## Installation & Running

### Prerequisites

- Python 3.8+
- An OpenAI API key (for the inventory agent)

### Quick Start (Local Storage)

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies from the root directory:
   ```bash
   uv sync
   ```

3. Create a `.env` file in the current directory:
   ```
   USE_LOCAL_STORAGE=true
   OPENAI_API_KEY=your-openai-api-key-here
   ```

4. Run the application:
   ```bash
   uv run main.py
   ```

5. The app will automatically open in your browser at `http://localhost:8000`

6. Game data is saved locally in `data/game_runs.json`

### Supabase Configuration (Optional)

If using Supabase instead of local storage, set these environment variables in your `.env` file:

- `USE_LOCAL_STORAGE=false` (or omit this to disable local storage)
- `SUPABASE_URL` – Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` – service role key
- `SUPABASE_ANON_KEY` – public anon key (used for frontend config)
- `OPENAI_API_KEY` – OpenAI API key

### Environment Variables Reference

```bash
# Storage choice (optional, defaults to Supabase if variables are set)
USE_LOCAL_STORAGE=true

# Required for all setups
OPENAI_API_KEY=your-api-key-here

# Required only for Supabase setup
SUPABASE_URL=your-project-url
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key
```

## Deployment

### Local Development Deployment

No additional configuration needed. The quick start above handles everything.

### Render Deployment

To deploy to Render.com:

1. Fork or push this repository to GitHub

2. Create a new Web Service on Render:
   - Connect your GitHub repository
   - Set the **Start Command** to:
     ```bash
     cd examples/fullstack_demo && pip install -r requirements.txt && python backend/app.py
     ```

3. Add environment variables in Render dashboard:
   ```
   USE_LOCAL_STORAGE=true
   OPENAI_API_KEY=your-openai-api-key
   ```

   (If using Supabase, also add SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and SUPABASE_ANON_KEY)

4. Deploy and your application will be accessible at your Render URL

5. Data will be persisted in the container's ephemeral filesystem for local storage, or in Supabase if configured

### Docker Deployment

To run the application in Docker:

1. Create a `Dockerfile` in the `examples/fullstack_demo/` directory:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8000
   CMD ["python", "backend/app.py"]
   ```

2. Build and run:
   ```bash
   docker build -t ai-inventory-game .
   docker run -p 8000:8000 -e OPENAI_API_KEY=your-key -e USE_LOCAL_STORAGE=true ai-inventory-game
   ```

## Frontend Configuration

The backend serves configuration to the frontend based on your setup:
- For **local storage**: No authentication required, simple session-based gameplay
- For **Supabase**: Backend injects Supabase URL/anon key via config endpoint for browser-side auth

## Usage Flow

1. **Local Storage Mode**: Simply start the game directly
   - No login required for local testing
   - Choose Mode 1 (Daily Feedback) or Mode 2 (Periodic Guidance)
   - Press **Start Run** to begin the inventory management simulation

2. **Supabase Mode**: Sign up or log in using email/password
   - The frontend uses supabase-js for authentication
   - Choose your preferred game mode
   - Press **Start Run** to begin

3. **During Gameplay**:
   - The transcript panel mirrors backend events (observations, agent proposals, demand actions, etc.)
   - **Mode 1**: Use the chat box to exchange messages with the agent adviser, then submit final decisions
   - **Mode 2**: Agent runs automatically until guidance checkpoints; provide guidance when prompted

4. **Game Results**: Upon completion, results are saved to local storage or Supabase

## Notes & Tips

- Ensure your OpenAI API key is set; the demo uses `gpt-4o-mini`
- For **Supabase setup**: JWT verification pulls JWKS from `SUPABASE_URL/auth/v1/keys`. Ensure backend has outbound network access
- For **local storage**: Data persists in `data/game_runs.json` in the working directory
- Inspect backend logs for demand and agent outputs when debugging prompts
- To run Mode 1 through multiple dialogue turns, keep sending chat messages until satisfied, then submit the final action

## Citation

If you use this application in your research or teaching, please cite our paper:

```bibtex
@article{baek2024ai,
    title={AI Agents for Inventory Control: Human-LLM-OR Complementarity},
    author={Baek, Jackie and Fu, Yaopeng and Ma, Will and Peng, Tianyi},
    year={2024}
}
```

**Full Citation**: Baek, J., Fu, Y., Ma, W., & Peng, T. (2024). AI Agents for Inventory Control: Human-LLM-OR Complementarity.

