# AI-Human Inventory Game

A fullstack web application for exploring human-AI collaboration in inventory management. This application combines an interactive web interface with LLM agents to create an engaging learning experience for inventory control and supply chain optimization.

This implementation uses FastAPI for the backend and supports both Supabase (for multi-user deployments) and local JSON storage (for local testing and deployment without external dependencies).

## Project Layout

- `backend/simulation.py` – reusable helpers that wrap the vending machine environment, capturing
  transcript events and supporting both Mode 1 (Daily Feedback) and Mode 2 (Periodic Guidance).
- `backend/app.py` – FastAPI application exposing endpoints for creating runs, exchanging
  messages/guidance, and recording final actions. Supports persistence to either Supabase or local JSON storage.
- `backend/token_verifier.py` – fetches Supabase JWKS and validates JWTs from the frontend (optional, only required for Supabase setup).
- `backend/supabase_client.py` – minimal supabase-py wrapper for inserting completed runs (optional, only required for Supabase setup).
- `frontend/` – static HTML interfaces with login (or simple session), mode selection, transcript view, and chat/guidance controls.

The application loads `examples/demand.csv` for demand data by default.

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

## Installation & Running

### Prerequisites

- Python 3.8+
- An OpenAI API key (for the inventory agent)

### Quick Start (Local Storage)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the `backend/` directory:
   ```
   USE_LOCAL_STORAGE=true
   OPENAI_API_KEY=your-openai-api-key-here
   ```

3. Run the application:
   ```bash
   cd examples/fullstack_demo
   python backend/app.py
   ```

4. The app will automatically open in your browser at `http://localhost:8000`

5. Game data is saved locally in `data/game_runs.json`

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

