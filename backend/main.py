"""
FastAPI Backend for AgentOps AI Platform

This is the main entry point for the REST API that exposes the multi-agent system
to external clients (web UI, CLI, other services).

Architecture:
- FastAPI application with modular routers
- Health check endpoint for monitoring
- Separate routers for different concerns (run, history, etc.)
- Future: WebSocket support for streaming agent responses

Run locally:
    uvicorn backend.main:app --reload --port 8000

Production:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routers import run, history

# =============================================================================
# Lifespan Context Manager
# =============================================================================
# This handles startup and shutdown events for the FastAPI app.
# Use this for:
# - Database connection pooling
# - Cache initialization
# - Background task setup
# - Resource cleanup on shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle (startup/shutdown).

    Startup tasks (future):
    - Initialize database connection pool
    - Load configuration from environment
    - Validate API keys (OpenAI, Langfuse, LangSmith)
    - Pre-warm LLM clients if needed

    Shutdown tasks (future):
    - Close database connections
    - Flush observability buffers
    - Cancel pending tasks
    """
    # Startup
    print("🚀 FastAPI backend starting up...")
    # TODO: Add startup logic here (database, cache, etc.)

    yield  # Application runs here

    # Shutdown
    print("🛑 FastAPI backend shutting down...")
    # TODO: Add cleanup logic here


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="AgentOps AI Platform API",
    description="REST API for multi-agent task execution with LangGraph",
    version="0.1.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc documentation at /redoc
    lifespan=lifespan,
)

# =============================================================================
# CORS Middleware
# =============================================================================
# Allow cross-origin requests from frontend applications.
# In production, restrict `allow_origins` to specific domains.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Demo: allow all origins for local frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# =============================================================================
# Health Check Endpoint
# =============================================================================
# This endpoint is used by:
# - Load balancers to check if the service is alive
# - Monitoring systems (Prometheus, Datadog, etc.)
# - Kubernetes liveness/readiness probes
# - CI/CD pipelines to verify deployment


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancing.

    Returns:
        dict: Health status and system information

    Example response:
        {
            "status": "healthy",
            "version": "0.1.0",
            "service": "agentops-ai-backend"
        }

    Future enhancements:
    - Check database connectivity
    - Check LLM API availability (OpenAI, Gemini)
    - Check observability backend status (Langfuse, LangSmith)
    - Return detailed status codes (200 = healthy, 503 = degraded)
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "agentops-ai-backend",
    }


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        dict: Welcome message and links to documentation
    """
    return {
        "message": "Welcome to AgentOps AI Platform API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "version": "0.1.0",
    }


# =============================================================================
# Router Registration
# =============================================================================
# Routers organize endpoints by domain (run, history, etc.).
# This keeps main.py clean and makes the codebase easier to navigate.

app.include_router(run.router, tags=["Run"])
app.include_router(history.router, tags=["History"])


# =============================================================================
# Error Handlers
# =============================================================================
# Custom error handlers for graceful error responses.
# Future: Add handlers for specific exceptions (ValidationError, etc.)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # This allows running the app directly with: python backend/main.py
    # For development only. Use uvicorn in production.
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
