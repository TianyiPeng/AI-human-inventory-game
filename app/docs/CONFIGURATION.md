# Configuration Guide

The AI-Human Inventory Game uses environment variables for configuration. This guide explains all available options.

## Configuration Files

Configuration can be provided through:

1. **`.env` file** (recommended for development)
   - Copy from `.env.example` and modify values
   - Loaded automatically by `main.py`

2. **Environment variables** (recommended for production)
   - Set directly in your shell or deployment environment
   - Takes precedence over `.env` file

3. **System environment**
   - Loaded from current working directory

## Configuration Options

### Required Configuration

#### `OPENAI_API_KEY`
- **Type**: String
- **Required**: Yes
- **Description**: Your OpenAI API key for LLM agent functionality
- **Example**: `OPENAI_API_KEY=sk-proj-xxxxx`
- **Get it**: https://platform.openai.com/api-keys

### Storage Configuration

#### `USE_LOCAL_STORAGE`
- **Type**: Boolean
- **Required**: No
- **Default**: `true`
- **Valid values**: `true`, `false`, `1`, `0`, `yes`, `no`
- **Description**: Use local JSON file storage instead of Supabase
  - `true`: Store game data in `data/game_runs.json`
  - `false`: Store game data in Supabase (requires Supabase env vars)
- **Example**: `USE_LOCAL_STORAGE=true`

#### `DATA_DIR`
- **Type**: Path
- **Required**: No
- **Default**: `./data`
- **Description**: Directory for storing local game data and files
- **Example**: `DATA_DIR=/var/lib/ai-inventory-game/data`
- **Note**: Only used when `USE_LOCAL_STORAGE=true`

### Supabase Configuration

Required only when `USE_LOCAL_STORAGE=false`.

#### `SUPABASE_URL`
- **Type**: String
- **Required**: Conditional (if not using local storage)
- **Description**: Your Supabase project URL
- **Format**: `https://your-project.supabase.co`
- **Example**: `SUPABASE_URL=https://myproject.supabase.co`
- **Get it**: Supabase Dashboard → Settings → API → Project URL

#### `SUPABASE_SERVICE_ROLE_KEY`
- **Type**: String
- **Required**: Conditional (if not using local storage)
- **Description**: Supabase service role key for backend operations
- **Example**: `SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
- **Get it**: Supabase Dashboard → Settings → API → Service Role Secret
- **Security**: Keep this secret! Never commit to version control.

#### `SUPABASE_ANON_KEY`
- **Type**: String
- **Required**: No (only needed if using Supabase from frontend)
- **Description**: Supabase anonymous key for frontend authentication
- **Example**: `SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
- **Get it**: Supabase Dashboard → Settings → API → Anon Public Key

### OpenAI Configuration

#### `OPENAI_MODEL`
- **Type**: String
- **Required**: No
- **Default**: `gpt-4o-mini`
- **Valid values**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`
- **Description**: OpenAI model to use for the inventory agent
- **Example**: `OPENAI_MODEL=gpt-4o`
- **Cost note**: Smaller models (e.g., `gpt-4o-mini`) are more cost-effective for testing

### Server Configuration

#### `PORT`
- **Type**: Integer
- **Required**: No
- **Default**: `8000`
- **Valid range**: `1024` to `65535`
- **Description**: Port number for the web server
- **Example**: `PORT=8080`
- **Note**: Ports below 1024 require root privileges

#### `HOST`
- **Type**: String
- **Required**: No
- **Default**: `0.0.0.0`
- **Valid values**: IP address or hostname
- **Description**: Host/interface to bind to
  - `0.0.0.0`: Accept connections from any interface
  - `localhost` or `127.0.0.1`: Only local connections
  - Specific IP: Bind to specific interface
- **Example**: `HOST=0.0.0.0`

#### `FULLSTACK_DEMO_RELOAD`
- **Type**: Boolean
- **Required**: No
- **Default**: `1` (enabled)
- **Valid values**: `1`, `0`, `true`, `false`
- **Description**: Enable auto-reload on file changes (development mode)
  - `1`: Enable reload (development)
  - `0`: Disable reload (production)
- **Example**: `FULLSTACK_DEMO_RELOAD=0`

#### `DEBUG`
- **Type**: Boolean
- **Required**: No
- **Default**: `false`
- **Valid values**: `true`, `false`, `1`, `0`
- **Description**: Enable debug logging
- **Example**: `DEBUG=true`
- **Note**: Produces verbose logs, not recommended for production

### Application Configuration

#### `DEMAND_DATA_PATH`
- **Type**: Path
- **Required**: No
- **Default**: `examples/initial_synthetic_demand_files/demand.csv`
- **Description**: Path to CSV file with demand data
- **Example**: `DEMAND_DATA_PATH=data/custom_demand.csv`
- **Note**: Relative paths are relative to project root

## Example Configurations

### Local Development Setup

```bash
# .env or environment variables
USE_LOCAL_STORAGE=true
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
PORT=8000
HOST=localhost
FULLSTACK_DEMO_RELOAD=1
DEBUG=true
```

### Classroom/Institution Deployment

```bash
# .env or environment variables
USE_LOCAL_STORAGE=true
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
PORT=8000
HOST=0.0.0.0
FULLSTACK_DEMO_RELOAD=0
DEBUG=false
DATA_DIR=/var/lib/ai-inventory-game/data
```

### Cloud Deployment with Supabase

```bash
# .env or environment variables
USE_LOCAL_STORAGE=false
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
SUPABASE_URL=https://myproject.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
SUPABASE_ANON_KEY=eyJhbGci...
PORT=8000
HOST=0.0.0.0
FULLSTACK_DEMO_RELOAD=0
DEBUG=false
```

### Budget-Conscious Setup

```bash
# Use smallest model and local storage to minimize costs
USE_LOCAL_STORAGE=true
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

## Configuration Priority

Environment variables are loaded in this order (first found wins):

1. `.env` file in `examples/fullstack_demo/` directory
2. `.env` file in project root (`OR_Agent/`)
3. System environment variables
4. Default values

Example:
```bash
# Shell environment variable takes precedence
PORT=8080 python main.py  # Will use port 8080
```

## Validation

The application validates configuration on startup:

- OpenAI API key is checked to be non-empty
- Supabase configuration is validated if local storage is disabled
- Port number is checked to be valid (1024-65535)

Invalid configuration will cause the application to exit with an error message.

## Security Best Practices

1. **Never commit `.env` to version control**
   - Use `.env.example` as a template instead
   - Add `.env` to `.gitignore`

2. **Protect API keys**
   - Use environment variables in production
   - Rotate keys regularly
   - Never share keys in logs or error messages

3. **Use HTTPS in production**
   - Deploy behind a reverse proxy (nginx, Cloudflare)
   - Enable SSL/TLS certificates

4. **Restrict data access**
   - Use Supabase Row Level Security (RLS)
   - Limit who can access game data

5. **Monitor and log**
   - Enable debug logging only when needed
   - Review logs regularly for issues
   - Set up error monitoring (Sentry, etc.)

## Troubleshooting

### "OPENAI_API_KEY not set"
- Check if key is in `.env` or environment
- Verify the key is valid at https://platform.openai.com/api-keys

### "Port already in use"
- Change `PORT` to a different number
- Or kill the process using that port

### "Connection refused" to Supabase
- Verify `SUPABASE_URL` is correct
- Check network connectivity
- Ensure firewall allows outbound HTTPS

### "403 Forbidden" errors
- Verify `SUPABASE_SERVICE_ROLE_KEY` is correct
- Check Supabase Row Level Security policies

## Reference Table

| Variable | Type | Required | Default | Notes |
|----------|------|----------|---------|-------|
| `OPENAI_API_KEY` | String | Yes | - | Keep secret |
| `OPENAI_MODEL` | String | No | `gpt-4o-mini` | Affects costs |
| `USE_LOCAL_STORAGE` | Boolean | No | `true` | Simple setup |
| `DATA_DIR` | Path | No | `./data` | Local storage only |
| `SUPABASE_URL` | String | Conditional | - | Required if not using local |
| `SUPABASE_SERVICE_ROLE_KEY` | String | Conditional | - | Keep secret |
| `SUPABASE_ANON_KEY` | String | No | - | For frontend auth |
| `PORT` | Integer | No | `8000` | 1024-65535 |
| `HOST` | String | No | `0.0.0.0` | Bind address |
| `FULLSTACK_DEMO_RELOAD` | Boolean | No | `1` | Dev mode only |
| `DEBUG` | Boolean | No | `false` | Verbose logging |
| `DEMAND_DATA_PATH` | Path | No | `examples/...` | CSV file path |
