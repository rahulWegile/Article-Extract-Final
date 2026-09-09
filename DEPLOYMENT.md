# GPU Cloud Deployment

One-command deployment of the full stack (Postgres + GPU-accelerated
FastAPI backend + Nginx-served React frontend) to a cloud GPU VM
(RunPod, AWS EC2 G4dn/G5, Vast.ai, etc.).

## 1. Provision the VM

Pick an instance with an NVIDIA GPU (T4/G4dn is enough; A10G/L4 or
better if you want closer to the ~5-10s/page figure). Ubuntu 22.04 is
assumed below.

## 2. Install the NVIDIA driver + Docker + NVIDIA Container Toolkit

Skip the driver step if your provider's GPU image already has it
(RunPod/Vast.ai templates usually do — check with `nvidia-smi`).

```bash
# NVIDIA driver (skip if `nvidia-smi` already works)
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot   # then reconnect and confirm: nvidia-smi

# Docker Engine + Compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit (lets containers see the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Sanity check -- should print the same GPU nvidia-smi showed above
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

## 3. Clone the repo and add your models

```bash
git clone <your-repo-url> newspaper-archive
cd newspaper-archive
```

The model weights are gitignored (too large for the repo) and are
volume-mounted at runtime, not baked into the image — copy them onto
the VM into the same layout as local dev:

```
models/
  doclayout_yolo.pt
  urdu_line_detector/yolov8m_UrduDoc.pt
  utrnet/UTRNet-Large.pth
  tessdata/            (populated automatically at image build time)
```

`scp`, `rsync`, or a cloud storage bucket all work — however they
get there, `models/doclayout_yolo.pt` etc. must exist before you run
a document through the pipeline (the app will raise a clear
`FileNotFoundError`/`RuntimeError` naming the missing path otherwise).

## 4. Configure environment variables

```bash
cp .env.example .env
nano .env   # set OPENAI_API_KEY / GEMINI_API_KEY, DB_PASSWORD, etc.
```

Leave `OCR_DEVICE` alone — `docker-compose.yml` forces it to `cuda`
for the backend container regardless of what's in `.env`, so the same
`.env` still works unmodified for someone running the pipeline
directly on a CPU-only laptop.

## 5. Build and start everything

```bash
docker compose up -d --build
```

First boot will take a few minutes (CUDA base image pull, pip
install, tessdata download, Postgres schema init). Watch it with:

```bash
docker compose logs -f
```

## 6. Verify

```bash
curl http://localhost/health          # -> {"status":"ok","db":"ok"}
docker exec -it $(docker compose ps -q backend) python3.12 -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then open `http://<server-ip>` in a browser — that single URL serves
the frontend and, via Nginx's reverse proxy, the backend API too, so
there's nothing else to open in your cloud provider's firewall/security
group except port 80 (and 22 for SSH).

## 7. Common operations

```bash
docker compose logs -f backend        # tail backend logs
docker compose restart backend        # after editing .env
docker compose down                   # stop everything (keeps volumes)
docker compose down -v                # also wipes the Postgres volume
docker compose up -d --build backend  # rebuild just one service
```

`output/`, `uploads/`, `cache/`, and `models/` are bind-mounted from
the VM's filesystem (see `docker-compose.yml`), and Postgres data
lives in the named volume `postgres_data` — all of it survives
`docker compose down` / container restarts / image rebuilds. Only
`docker compose down -v` or manually deleting those host directories
loses data.

## Notes on the pieces

- **Why the CUDA base image, not `pytorch/pytorch:2.4.0-...`**:
  `requirements.txt` already pins `torch==2.12.0`, and PyPI's Linux
  wheel for it bundles its own CUDA runtime — a `pytorch/pytorch`
  base would just add a second, conflicting torch install. The plain
  `nvidia/cuda` base provides the CUDA/cuDNN userspace libraries other
  native deps expect without fighting the pinned version.
- **Why the frontend build has no hardcoded API URL**: the frontend
  is built with `VITE_API_URL=""`, which makes every API call go out
  relative to whatever host it's loaded from. Nginx then reverse-
  proxies those paths to the backend container on the same origin —
  that's what gives you one URL with zero CORS configuration, and why
  the frontend image doesn't need to know the server's IP/domain at
  build time.
- **Why `/upload`, `/search`, and `/health` are proxied alongside
  `/documents` and `/output`**: the app's actual routes
  (`backend/main.py`, `backend/routes/*.py`) are mounted at the root
  path, not under `/api/` — `/api/` is proxied too (see
  `frontend/nginx.conf`) so it's ready if routes move under that
  prefix later, but today it's `/upload`, `/search/*`, `/documents/*`,
  `/output/*`, and `/health` that actually need to reach the backend.
