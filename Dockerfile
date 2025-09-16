# Use Ubuntu latest as base
FROM ubuntu:latest

# Set environment variables (avoid warnings and optimize JAX)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-venv \
        libopenmpi-dev \
        && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip point to Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Install Ray RLlib and dependencies
RUN pip install "ray[rllib]" "ray[default]" "ray[serve]" && \
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

