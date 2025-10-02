# -------- Stage 1: Builder (install dependencies, build with Poetry) --------
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build tools needed for compiling dependencies (if any)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl gcc libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Poetry installation directory and update PATH
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Poetry via official installer in builder stage
RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.1.2

# Copy project dependency files first (to leverage Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Configure Poetry to install only prod deps (no venv, no cache)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_NO_ANSI=1

RUN poetry install --only main --no-root

# Copy the app source code (after dependencies so that code changes don't bust dependency layers)
COPY app ./app

# -------- Stage 2: Runtime (slim, minimal) --------
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy over the installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY app ./app

# Expose port
EXPOSE 8000

# Entrypoint / command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
