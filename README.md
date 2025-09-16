# Ray + JAX + Brax + JupyterLab Docker Image

This repository provides a minimal Docker setup for running **Ray RLlib**, **Gymnasium**, **JAX**, **Brax**, and **JupyterLab** in a Python 3.10 environment.  
It is designed for reinforcement learning (RL) experiments and development, with JupyterLab as the default entrypoint.

---

## Features

- **Python 3.10** (stable with Ray and JAX)
- **Ray** (`rllib`, `serve`, and core features)
- **Gymnasium** with Atari support (ROM license accepted automatically)
- **JAX** + **Brax** (CPU version by default)
- **JupyterLab** pre-installed and auto-started when container runs
- Minimal `slim` base image for efficiency

---

## Build

You can build the image locally using Docker:

```bash
docker build -t python-jupyter-lab .
```

---

## Run

Start a container with JupyterLab exposed on port `8888`:

```bash
docker run -it --rm -p 8888:8888 python-jupyter-lab
```

Then open [http://localhost:8888](http://localhost:8888) in your browser.  
By default, authentication token is disabled (`--NotebookApp.token=''`), so no login is required.

### Custom Port

If you prefer a different outside port (e.g. `9999`):

```bash
docker run -it --rm -p 9999:8888 python-jupyter-lab
```

JupyterLab inside the container still listens on port `8888`, but you access it via `http://localhost:9999`.

### Mount Local Directory

To make your local files available inside JupyterLab:

```bash
docker run -it --rm -p 8888:8888 -v $(pwd):/app python-jupyter-lab
```

---

## GitHub Actions Workflow

This repository also contains a GitHub Actions workflow that:

1. Builds the Docker image from the provided `Dockerfile`.
2. Tags the image.
3. Pushes it automatically to [Docker Hub](https://hub.docker.com/) when changes are pushed to the repository.

Make sure to configure the following GitHub secrets in your repository settings:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## Notes

- This image installs the **CPU version of JAX/Brax**.  
  For GPU-enabled systems, update the `pip install jaxlib` line in the Dockerfile to the appropriate CUDA wheel (see [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)).
- Security: the container disables Jupyter token authentication for convenience.  
  If you run this in a shared or public environment, set a password or re-enable token auth.

---

## License

MIT
