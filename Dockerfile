# Use Python 3.10 (stable with Ray and JAX)
FROM python:3.10-slim

# Set environment variables (avoid warnings and optimize JAX)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        python3-dev \
        libopenmpi-dev \
        && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Ray RLlib and dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install "ray[rllib]" "ray[default]" "ray[serve]" && \
    pip install gymnasium[atari,accept-rom-license] && \
    pip install pygame

# Install JAX + Brax (CPU version)
RUN pip install --upgrade jax jaxlib brax

# Install JupyterLab
RUN pip install jupyterlab

# Expose port for JupyterLab
EXPOSE 8888

# Launch JupyterLab when container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
