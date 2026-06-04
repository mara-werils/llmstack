# Kubernetes Deployment

## Quick start

```bash
kubectl apply -f deployment.yaml
kubectl apply -f secrets.yaml
```

## Components

- **Deployment**: 2 replicas with HPA (auto-scales to 10)
- **Service**: ClusterIP on port 80
- **HPA**: CPU 70% / Memory 80% thresholds
- **PDB**: Minimum 1 pod available during disruptions

## Configuration

Edit `deployment.yaml` ConfigMap for runtime settings.
Edit `secrets.yaml` for API keys and credentials.

## Health checks

- Liveness: `/healthz` (15s interval)
- Readiness: `/healthz` (10s interval)
- Startup: `/healthz` (5s interval, 60s budget)
