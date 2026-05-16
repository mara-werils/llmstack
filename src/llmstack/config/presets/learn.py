"""Learn preset — self-improving AI that adapts to your preferences.

Enables the adaptive learning pipeline with feedback collection,
automatic training triggers, quality monitoring, and preference
learning. Best for users who want their local AI to improve over time.
"""

LEARN_PRESET_YAML = """# LLMStack Learning Preset — Self-Improving Local AI
# Your AI gets measurably better the more you use it.
version: "1"

models:
  chat:
    name: llama3.2
    backend: auto
    context_length: 8192
  embeddings:
    name: bge-m3
    backend: auto

# Adaptive Learning Pipeline
learn:
  enabled: true

  feedback:
    implicit_signals: true        # Track copy, regenerate, abandon
    prompt_interval: 5            # Ask for feedback every 5 interactions
    interactive_feedback: true    # Enable in chat/ask modes

  training:
    min_feedback: 25              # Train after 25 feedback items
    min_interval_hours: 1         # Don't train more than once per hour
    max_wait_hours: 24            # Force train after 24h if feedback exists
    strategy: mixed               # SFT + DPO combined
    base_model: unsloth/llama-3.2-1b-instruct-bnb-4bit
    max_examples: 5000
    auto_activate: true           # Only if quality improves
    min_improvement: 0.01

  quality:
    enabled: true
    auto_rollback: true           # Rollback on severe regression
    min_samples: 10
    metrics: [overall, coherence, relevance]
    severe_threshold: 0.15

  preferences:
    enabled: true
    inject_into_prompts: true     # Auto-adapt to your style
    min_signals: 5

gateway:
  port: 8000

observe:
  quality_tracking: true
"""
