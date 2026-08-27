# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the framework project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
# If there is a docs directory required for packaging or monitor, copy it
COPY docs/ ./docs/

# Install IntaGrin and its dependencies globally inside the container
RUN pip install --no-cache-dir .

# Expose port 8000 for the API Server and 3000 for the Monitor (if needed)
EXPOSE 8000 3000

# Copy your actual agent files (like ai.yaml) into the container
# This assumes the user will mount or build their own agent project over /app/agent
# We create a placeholder directory
WORKDIR /app/agent

# Set a health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Define the command to run the production ASGI server with Gunicorn & Uvicorn workers
CMD ["gunicorn", "intagrin.server.api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
