"""Secure preset — hardened production-ready configuration.

Enables authentication, rate limiting, and conservative defaults
for deploying LLMStack in shared or production environments.
"""

SECURE_PRESET_YAML = """# LLMStack Secure Preset — Production-Hardened Configuration
# Locked down defaults suitable for shared and production environments.
version: "1"

models:
  chat:
    name: llama3.2
    backend: auto
    context_length: 8192
  embeddings:
    name: bge-m3
    backend: auto

services:
  vectors:
    provider: qdrant
  cache:
    provider: redis
    max_memory: 512mb

gateway:
  port: 8000
  auth: api_key
  api_keys: []                        # Add your keys here
  rate_limit: 30/min                  # Conservative rate limit
  cors:
    - "http://localhost:3000"         # Restrict CORS origins
  request_timeout: 60                 # Shorter timeout

observe:
  metrics: true
  quality_tracking: true
  alert_threshold: 0.4
  drift_threshold: -0.1
  trace_store_size: 10000

docker:
  network: llmstack_net
  gpu: auto
"""
