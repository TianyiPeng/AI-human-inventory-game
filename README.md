<div align="center">

# AI-Human Inventory Game

A fullstack web application for exploring human-AI collaboration in inventory management.

Combines interactive gameplay with LLM agents to create an engaging learning experience for inventory control and supply chain optimization.

</div>

## Overview

The **AI-Human Inventory Game** is a modern web application that teaches inventory management principles through interactive play. Users manage a vending machine inventory while an AI agent provides recommendations based on Operations Research and machine learning techniques.

**Key Features:**
- 🤖 AI agent powered by OpenAI LLMs
- 👥 Two gameplay modes:
  - **Mode 1 (Daily Feedback)**: Chat with AI and make daily decisions
  - **Mode 2 (Periodic Guidance)**: Agent runs automatically with periodic guidance
- 💾 Local JSON storage (no external dependencies) or cloud-based Supabase
- 🚀 Easy deployment (local, Render.com, Docker)
- 📚 Educational tool for teaching supply chain optimization
- 🎓 Perfect for classroom use and demonstrations

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation with uv (30 seconds)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy environment template
cp app/.env.example app/.env

# Edit .env and add your OpenAI API key
# nano app/.env
# OPENAI_API_KEY=sk-your-key-here

# Run the app
cd app && uv run main.py
```

The app will open automatically in your browser at `http://localhost:8000`.

Game data is saved locally in `app/data/game_runs.json`.

## Project Structure

```
├── app/                             # Main application
│   ├── backend/                     # FastAPI application
│   │   ├── app.py                  # Main endpoints
│   │   ├── simulation_current.py    # Game logic & agent integration
│   │   ├── storage.py              # Storage abstraction (JSON/Supabase)
│   │   ├── config.py               # Configuration management
│   │   └── ...
│   ├── core/                        # Core business logic
│   │   ├── or_csv_demo.py          # OR agent implementation
│   │   └── or_to_llm_csv_demo.py   # Hybrid LLM-OR agent
│   ├── frontend/                    # Web interfaces (HTML/JS)
│   ├── data/                        # Local storage (auto-created)
│   ├── docs/                        # Detailed guides
│   ├── main.py                     # Entry point
│   ├── requirements.txt            # Dependencies
│   ├── .env.example                # Configuration template
│   └── README.md                   # Application documentation
├── examples/
│   └── initial_synthetic_demand_files/  # Sample demand data
├── textarena/                       # Game environment framework
├── pyproject.toml                  # Python project configuration
└── README.md                        # This file
```

## Documentation

**Get Started Quickly:**
- [Quick Start Guide](app/README.md) - Overview and setup options
- [Development Guide](app/docs/DEVELOPMENT.md) - Local development and debugging

**Detailed References:**
- [Deployment Guide](app/docs/DEPLOYMENT.md) - Production deployment (Render, Docker, local)
- [Configuration Guide](app/docs/CONFIGURATION.md) - All environment variables and options

## Deployment Options

### 🏠 Local Deployment
Perfect for classrooms and personal use:
```bash
cd app
uv run main.py
```

### ☁️ Render Cloud Hosting
Deploy to Render.com with one click - see [Deployment Guide](app/docs/DEPLOYMENT.md)

### 🐳 Docker
Run in a containerized environment:
```bash
docker build -t ai-inventory-game app
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-inventory-game
```

### 💎 Supabase
Multi-user support with cloud database - see [Deployment Guide](app/docs/DEPLOYMENT.md)

## Configuration

The application uses environment variables for configuration. Copy and customize:

```bash
cp app/.env.example app/.env
```

**Essential Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `USE_LOCAL_STORAGE` - Use local JSON storage (default: true)
- `OPENAI_MODEL` - Model to use (default: gpt-4o-mini)
- `PORT` - Server port (default: 8000)

See [Configuration Guide](app/docs/CONFIGURATION.md) for all options.

## Gameplay Modes

### Mode 1: Daily Feedback
- You make inventory decisions each day
- Chat with the AI agent for advice and recommendations
- Submit your final order decision
- Receive feedback on your choices

### Mode 2: Periodic Guidance
- Agent makes decisions autonomously
- You provide guidance at key decision points
- Agent adapts recommendations based on your feedback
- Less hands-on, great for observing AI decision-making

## Citation

If you use this application in research or teaching, please cite:

```bibtex
@article{baek2024ai,
    title={AI Agents for Inventory Control: Human-LLM-OR Complementarity},
    author={Baek, Jackie and Fu, Yaopeng and Ma, Will and Peng, Tianyi},
    year={2024}
}
```

**Full Citation:** Baek, J., Fu, Y., Ma, W., & Peng, T. (2024). AI Agents for Inventory Control: Human-LLM-OR Complementarity.

## Development

The application is built with:
- **Backend**: FastAPI (Python)
- **Frontend**: HTML5 + JavaScript
- **Game Framework**: TextArena
- **LLM**: OpenAI GPT models
- **Storage**: JSON (local) or Supabase (cloud)

### For Local Development
See [Development Guide](app/docs/DEVELOPMENT.md) for setup and debugging.

## Troubleshooting

**"Port 8000 already in use"**
- Change the port in `.env`: `PORT=8001`

**"OpenAI API key error"**
- Verify your key at https://platform.openai.com/api-keys
- Check `.env` file is in the correct location

**"Game data not saving"**
- Ensure `app/data/` directory exists and is writable
- Check `USE_LOCAL_STORAGE=true` in `.env`

For more help, see:
- [DEVELOPMENT.md](app/docs/DEVELOPMENT.md) - Debugging guide
- [DEPLOYMENT.md](app/docs/DEPLOYMENT.md) - Common deployment issues

## Contributing

To contribute improvements:
1. Make changes in a feature branch
2. Test locally with `python main.py`
3. Create a pull request with your improvements

## License

MIT License - See [LICENSE](LICENSE) file

## Support

- 📖 [Quick Start](app/README.md)
- 🛠️ [Development Guide](app/docs/DEVELOPMENT.md)
- 🚀 [Deployment Guide](app/docs/DEPLOYMENT.md)
- ⚙️ [Configuration Reference](app/docs/CONFIGURATION.md)

---

Made for teaching and exploring human-AI collaboration in inventory management.
