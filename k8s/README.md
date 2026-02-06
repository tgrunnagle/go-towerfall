# Kubernetes Deployment for go-towerfall

This directory contains Kubernetes manifests for deploying the go-towerfall application (frontend, backend, and bot2) to a Kubernetes cluster with Cloudflare Tunnel for internet exposure.

## Prerequisites

### Required Tools

- Docker Desktop with WSL2 backend (Windows) or Docker (Linux/Mac)
- kubectl
- kind (Kubernetes in Docker) - install via `task k8s:install-kind`
- Cloudflare account with a configured tunnel (for external access)
- envsubst (run `task k8s:install-envsubst` on Windows)

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CF_TUNNEL_ID` | Cloudflare tunnel ID | `abc123-...` |
| `CF_ACCOUNT_TAG` | Cloudflare account tag | `abc123...` |
| `CF_TUNNEL_SECRET` | Tunnel secret (base64) | `secret...` |
| `API_DOMAIN` | Domain for backend API | `api.example.com` |
| `FRONTEND_DOMAIN` | Domain for frontend | `game.example.com` |
| `IMAGE_REGISTRY` | Container registry prefix | `ghcr.io/user` or `towerfall` |

## Quick Start: Local Kind Cluster

This section covers deploying to a local kind cluster for development and testing.

### 1. Install Prerequisites

```bash
# Install kind via task (requires Go)
task k8s:install-kind

# Or install kubectl if not already installed
# Windows: winget install Kubernetes.kubectl
# Mac: brew install kubectl
# Linux: See https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
```

### 2. Create Kind Cluster

```bash
# Create a new kind cluster named "towerfall"
kind create cluster --name towerfall

# Verify the cluster is running
kubectl cluster-info --context kind-towerfall

# Set as current context (if not already)
kubectl config use-context kind-towerfall
```

### 3. Build and Load Docker Images

Kind runs its own container registry isolated from your local Docker daemon. Images must be explicitly loaded into the kind cluster after building.

```bash
# Navigate to project root
cd /path/to/go-towerfall

# Build all images
docker build -t towerfall/backend:latest ./backend
docker build -t towerfall/frontend:latest ./frontend
docker build -t towerfall/bot2:latest ./bot2

# Load images into kind cluster
# This copies images from your local Docker daemon into kind's internal registry
kind load docker-image towerfall/backend:latest --name towerfall
kind load docker-image towerfall/frontend:latest --name towerfall
kind load docker-image towerfall/bot2:latest --name towerfall

# Verify images are loaded
docker exec -it towerfall-control-plane crictl images | grep towerfall
```

**Important notes about kind image loading:**

| Consideration | Details |
|---------------|---------|
| **Images must be reloaded after rebuild** | After `docker build`, always run `kind load` again |
| **Image tags must match** | Use the same tag in `docker build`, `kind load`, and deployment manifests |
| **imagePullPolicy** | Deployments use `IfNotPresent` to prefer local images over pulling |
| **Large images take time** | The bot2 image (with CUDA) may take several minutes to load |
| **Cluster name matters** | Use `--name towerfall` to target the correct cluster |

**Quick reload script** (after code changes):
```bash
# One-liner to rebuild and reload a single service
docker build -t towerfall/backend:latest ./backend && kind load docker-image towerfall/backend:latest --name towerfall

# Rebuild all and reload
for svc in backend frontend bot2; do
  docker build -t towerfall/$svc:latest ./$svc && kind load docker-image towerfall/$svc:latest --name towerfall
done
```

### 4. Configure Environment

```bash
cd k8s

# Copy example environment file
cp .env.example .env

# Edit .env with your values
# For local testing without Cloudflare, you can use placeholder values
```

### 5. Deploy to Cluster

```bash
# Generate manifests and apply to cluster
./install.sh --apply

# Or on Windows PowerShell (after running: task k8s:install-envsubst)
.\install.ps1 -Apply
```

### 6. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n towerfall

# Check services
kubectl get services -n towerfall

# Watch pod status
kubectl get pods -n towerfall -w
```

### 7. Access Services Locally

For local development without Cloudflare, use port-forwarding:

```bash
# Forward backend API (port 4000)
kubectl port-forward -n towerfall svc/backend-service 4000:4000

# In another terminal, forward frontend (port 4001)
kubectl port-forward -n towerfall svc/frontend-service 4001:4001
```

Then access:
- Frontend: http://localhost:4001
- Backend API: http://localhost:4000/api/maps

## Kind Cluster Management

### Common Commands

```bash
# List clusters
kind get clusters

# Delete cluster
kind delete cluster --name towerfall

# Get cluster info
kubectl cluster-info --context kind-towerfall

# View nodes
kubectl get nodes
```

### Rebuilding and Redeploying

After code changes:

```bash
# Rebuild specific service
docker build -t towerfall/backend:latest ./backend
kind load docker-image towerfall/backend:latest --name towerfall
kubectl rollout restart deployment/backend -n towerfall

# Rebuild all services
docker build -t towerfall/backend:latest ./backend
docker build -t towerfall/frontend:latest ./frontend
docker build -t towerfall/bot2:latest ./bot2
kind load docker-image towerfall/backend:latest --name towerfall
kind load docker-image towerfall/frontend:latest --name towerfall
kind load docker-image towerfall/bot2:latest --name towerfall
kubectl rollout restart -n towerfall deployment/backend deployment/frontend deployment/bot2
```

### Viewing Logs

```bash
# View backend logs
kubectl logs -n towerfall deployment/backend

# Follow logs in real-time
kubectl logs -n towerfall deployment/backend -f

# View logs from all pods of a deployment
kubectl logs -n towerfall -l app=backend --all-containers
```

## Windows/WSL Setup

For Windows users who prefer WSL:

```bash
# 1. Install WSL2 (from PowerShell as Admin)
wsl --install

# 2. Install Docker Desktop and enable WSL2 backend

# 3. Inside WSL, install kind (alternative to task)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# 4. Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# 5. Create and use cluster
kind create cluster --name towerfall
kubectl cluster-info --context kind-towerfall
```

## Deployment with Cloudflare Tunnel

For production deployment with external access via Cloudflare:

### 1. Create Cloudflare Tunnel

1. Log in to Cloudflare Dashboard
2. Go to Zero Trust > Access > Tunnels
3. Create a new tunnel named "towerfall"
4. Copy the tunnel credentials (ID, account tag, secret)

### 2. Configure DNS

Add DNS records pointing to your tunnel:
- `api.yourdomain.com` → Tunnel
- `game.yourdomain.com` → Tunnel

### 3. Update Environment

Edit `.env` with your Cloudflare credentials:

```bash
CF_TUNNEL_ID=your-tunnel-id
CF_ACCOUNT_TAG=your-account-tag
CF_TUNNEL_SECRET=your-tunnel-secret-base64
API_DOMAIN=api.yourdomain.com
FRONTEND_DOMAIN=game.yourdomain.com
IMAGE_REGISTRY=towerfall
```

### 4. Deploy

```bash
./install.sh --apply
```

The cloudflared pods will connect to Cloudflare and route traffic to your services.

## Troubleshooting

### Pods in ImagePullBackOff

Images not loaded into kind:
```bash
kind load docker-image towerfall/<service>:latest --name towerfall
```

### Pods in CrashLoopBackOff

Check logs for errors:
```bash
kubectl logs -n towerfall deployment/<service> --previous
kubectl describe pod -n towerfall -l app=<service>
```

### Cloudflared not connecting

1. Check tunnel credentials:
   ```bash
   kubectl get secret cloudflare-tunnel-credentials -n towerfall -o yaml
   ```

2. Check cloudflared logs:
   ```bash
   kubectl logs -n towerfall deployment/cloudflared
   ```

3. Verify tunnel is active in Cloudflare dashboard

### Frontend not loading

1. Verify ConfigMap mounted:
   ```bash
   kubectl describe pod -n towerfall -l app=frontend
   ```

2. Check nginx configuration:
   ```bash
   kubectl exec -n towerfall deployment/frontend -- cat /etc/nginx/conf.d/default.conf
   ```

### Backend health check failing

1. Check backend logs:
   ```bash
   kubectl logs -n towerfall deployment/backend
   ```

2. Test endpoint from within cluster:
   ```bash
   kubectl run -n towerfall curl --rm -it --image=curlimages/curl -- curl http://backend-service:4000/api/maps
   ```

### Kind cluster issues

```bash
# Reset cluster
kind delete cluster --name towerfall
kind create cluster --name towerfall

# Check Docker is running
docker ps

# Verify kind node is healthy
docker exec -it towerfall-control-plane kubectl get nodes
```

## Architecture

```
                    Internet
                        |
                        v
                  Cloudflare
                        |
                        v
              Cloudflare Tunnel
                        |
                        v
    +-------------------------------------------+
    |            Kubernetes Cluster             |
    |                                           |
    |   +---------------+   +---------------+   |
    |   | cloudflared   |   | cloudflared   |   |
    |   | (replica 1)   |   | (replica 2)   |   |
    |   +-------+-------+   +-------+-------+   |
    |           |                   |           |
    |           +-------+   +-------+           |
    |                   |   |                   |
    |                   v   v                   |
    |   +---------------+   +---------------+   |
    |   | backend-svc   |   | frontend-svc  |   |
    |   | :4000         |   | :4001         |   |
    |   +-------+-------+   +-------+-------+   |
    |           |                   |           |
    |           v                   v           |
    |   +---------------+   +---------------+   |
    |   | backend       |   | frontend      |   |
    |   | (replica 1)   |   | (replica 2)   |   |
    |   +---------------+   +---------------+   |
    |                                           |
    |   +---------------+                       |
    |   | bot2          |                       |
    |   | (ML training) |                       |
    |   +---------------+                       |
    +-------------------------------------------+
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ClusterIP services only | Cloudflared handles external exposure; no need for LoadBalancer/NodePort |
| Runtime config.js injection | Avoids rebuilding frontend image for URL changes |
| Separate hostnames (api/game) | Clean routing; WebSocket support on api hostname |
| GPU optional for bot2 | Works on CPU, GPU accelerates training when available |
| 2 cloudflared replicas | High availability for tunnel |
| envsubst for templating | Simple, standard tool for environment variable substitution in manifests |
| .env.example + .env pattern | Provides documentation while keeping secrets out of git |
| kind for local dev | Lightweight, Docker-based Kubernetes for fast iteration |
