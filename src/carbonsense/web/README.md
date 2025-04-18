# CarbonSense Web Dashboard

This is the web interface for the CarbonSense carbon footprint analysis system, built with FastAPI, modern HTML/CSS, and JavaScript.

## Features

- Interactive carbon footprint dashboard
- Circular progress indicators for various carbon metrics
- Natural language query system for carbon footprint information
- Activity tracking and carbon impact calculation
- Weekly carbon footprint trends visualization
- User goals and achievements

## Setup and Installation

1. Make sure you have already installed the CarbonSense package:

```bash
# From the project root
pip install -e .
```

2. All required dependencies should be installed automatically, including:
   - FastAPI
   - Uvicorn (ASGI server)
   - Jinja2 (templating)
   - Other CarbonSense dependencies

## Running the Server

From the project root directory, run:

```bash
python -m src.carbonsense.web.run_server
```

This will start the server at http://localhost:8000.

## Development

- **Frontend**: The frontend uses vanilla JavaScript with Chart.js for visualizations
- **Templates**: HTML templates are in the `templates/` directory
- **Static files**: CSS and JavaScript are in the `static/` directory
- **API routes**: All API endpoints are defined in `app.py`

## Integration with CarbonSense Backend

The web dashboard connects to the core CarbonSense functionality:

- Carbon calculation from the core modules
- Retrieval-Augmented Generation (RAG) for answering carbon footprint queries
- Activity tracking and carbon impact calculation

## Environment Variables

The web dashboard uses the same environment variables as the rest of the CarbonSense system. Make sure your `.env` file is properly configured in the project root.

## Production Deployment

For production deployment:

1. Set `reload=False` in `run_server.py`
2. Use a production ASGI server (Uvicorn with Gunicorn)
3. Consider using a reverse proxy like Nginx

Example production deployment command:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.carbonsense.web.app:app
``` 