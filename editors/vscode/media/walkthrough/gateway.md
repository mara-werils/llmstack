## Start your local gateway

LLMStack talks to a gateway running on **your** machine — code never leaves it.

```bash
# Install the CLI (one-liner)
curl -LsSf https://raw.githubusercontent.com/mara-werils/llmstack/main/install.sh | sh

# Start it (zero config, no API key needed)
llmstack quickstart
llmstack up
```

The status-bar item (bottom right) turns into **✓ LLMStack** once the gateway is
reachable.
