# Deployment Guide

This guide covers deploying the AI-Human Inventory Game to production environments.

## Deployment Options

### Option 1: Local Server (Recommended for Classrooms)

Deploy on a local machine or institutional server with local JSON storage.

**Advantages:**
- No external dependencies
- Full control over data
- Easy to set up for educational use
- No monthly subscription costs

**Steps:**

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` from template:
   ```bash
   cp .env.example .env
   ```

3. Configure for production:
   ```bash
   # .env
   USE_LOCAL_STORAGE=true
   OPENAI_API_KEY=your-key-here
   PORT=8000
   HOST=0.0.0.0
   FULLSTACK_DEMO_RELOAD=0
   ```

4. Run with a process manager (e.g., systemd, supervisor):
   ```bash
   python main.py
   ```

5. Set up reverse proxy (optional, for HTTPS):
   - Use nginx or Apache to proxy requests to localhost:8000
   - Enable SSL/TLS for security

### Option 2: Render Deployment (Cloud Hosting)

Deploy to Render.com for cloud hosting with persistent storage options.

**Advantages:**
- No server management
- Built-in SSL/TLS
- Persistent storage available
- Easy scaling

**Steps:**

1. Push code to GitHub

2. Go to [Render Dashboard](https://dashboard.render.com/)

3. Click "New +" → "Web Service"

4. Connect your GitHub repository

5. Configure the service:
   - **Name**: `ai-inventory-game` (or your preference)
   - **Environment**: Python 3.11
   - **Build Command**:
     ```bash
     cd examples/fullstack_demo && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     cd examples/fullstack_demo && python main.py
     ```

6. Add environment variables in dashboard:
   ```
   USE_LOCAL_STORAGE=true
   OPENAI_API_KEY=sk-...
   FULLSTACK_DEMO_RELOAD=0
   ```

7. Click "Create Web Service"

8. Wait for deployment (2-3 minutes)

9. Access your app at the provided Render URL

### Option 3: Docker Deployment

Deploy using Docker for consistency across environments.

**Advantages:**
- Reproducible environments
- Easy to scale
- Works on any platform with Docker

**Steps:**

1. Create a `Dockerfile` in `examples/fullstack_demo/`:

   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Copy application files
   COPY . .

   # Install dependencies
   RUN pip install --no-cache-dir -r requirements.txt

   # Expose port
   EXPOSE 8000

   # Run the application
   CMD ["python", "main.py"]
   ```

2. Build the Docker image:
   ```bash
   docker build -t ai-inventory-game:latest .
   ```

3. Run the container:
   ```bash
   docker run \
     -p 8000:8000 \
     -e OPENAI_API_KEY=sk-your-key \
     -e USE_LOCAL_STORAGE=true \
     -v game_data:/app/data \
     ai-inventory-game:latest
   ```

4. For production, use docker-compose:

   Create `docker-compose.yml`:

   ```yaml
   version: '3.8'

   services:
     app:
       build: .
       ports:
         - "8000:8000"
       environment:
         OPENAI_API_KEY: ${OPENAI_API_KEY}
         USE_LOCAL_STORAGE: "true"
         FULLSTACK_DEMO_RELOAD: "0"
       volumes:
         - game_data:/app/data
       restart: unless-stopped

   volumes:
     game_data:
   ```

   Run with:
   ```bash
   docker-compose up -d
   ```

### Option 4: Supabase + Cloud Deployment (Advanced)

For production systems with multi-user support and cloud database.

**Prerequisites:**
- Supabase account (free tier available)
- Cloud deployment platform (Render, Heroku, AWS, etc.)

**Supabase Setup:**

1. Create a Supabase project at [supabase.com](https://supabase.com/)

2. Create the game_runs table in SQL editor:
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

3. Enable Row Level Security (RLS)

4. Create policy for service role:
   ```sql
   create policy "Allow service role to insert"
   on public.game_runs
   for insert
   with check (true);
   ```

5. Enable Auth in Supabase settings

**Deploy with Supabase:**

1. Update `.env`:
   ```bash
   USE_LOCAL_STORAGE=false
   OPENAI_API_KEY=sk-...
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-service-key
   SUPABASE_ANON_KEY=your-anon-key
   ```

2. Deploy using your preferred platform (Render, Docker, etc.)

## Production Checklist

- [ ] Use a production-grade OpenAI model or provide model selection
- [ ] Set `FULLSTACK_DEMO_RELOAD=0` to disable auto-reload
- [ ] Configure HTTPS/SSL
- [ ] Set appropriate rate limits
- [ ] Monitor logs for errors
- [ ] Backup game data regularly (for local storage mode)
- [ ] Set up error monitoring (optional: Sentry, etc.)
- [ ] Configure CORS if needed

## Monitoring and Maintenance

### View Application Logs

**Render:**
```
Dashboard → Your Service → Logs tab
```

**Docker:**
```bash
docker logs ai-inventory-game
```

**Local Server:**
```bash
# Check stdout/stderr from the process
tail -f logs/app.log  # if configured
```

### Performance

The application tracks request statistics:
- Total requests
- Active concurrent connections
- Request timing
- Status codes

View these in application logs.

### Troubleshooting

**Application won't start:**
- Check OpenAI API key is valid
- Verify all required environment variables are set
- Check port is not already in use

**Slow response times:**
- Check OpenAI API status
- Monitor concurrent connections
- Consider increasing server resources

**Data not saving:**
- Verify `data/` directory permissions
- For Supabase, check network connectivity
- Review error logs for specific issues

## Scaling Considerations

For large deployments:

1. **Multiple instances**: Use load balancer (nginx, Render, etc.)
2. **Persistent storage**: Consider external database instead of JSON files
3. **Caching**: Add Redis for session caching
4. **Monitoring**: Set up comprehensive logging
5. **Analytics**: Track user engagement metrics

## Support

For issues or questions:
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for local setup help
- Review application logs for error details
- See main [README.md](../README.md) for usage information
