# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in LLMStack, please report it
responsibly. **Do not open a public issue.**

1. Email **security@llmstack.dev** with a description of the vulnerability.
2. Include steps to reproduce, affected versions, and any proof-of-concept.
3. We will acknowledge receipt within 48 hours and aim to provide an initial
   assessment within 5 business days.
4. Once a fix is ready we will coordinate disclosure with you before releasing.

## Security Features

LLMStack includes several built-in security controls:

- **API key authentication** -- the gateway enforces API key checks by default
  (`gateway.auth: api_key` in `llmstack.yaml`).
- **Rate limiting** -- configurable per-client rate limits prevent abuse
  (`gateway.rate_limit`).
- **Request size limits** -- the `RequestSizeMiddleware` rejects oversized
  payloads before they reach route handlers (default 10 MB).
- **CORS configuration** -- restrict allowed origins via `gateway.cors`.
- **PII detection guardrails** -- optional content filters can redact
  personally identifiable information before it reaches the LLM.
- **Prompt injection detection** -- optional guardrails flag common prompt
  injection patterns.
- **No secrets in config** -- API keys can be supplied via environment
  variables (`api_key_env` field) instead of being stored in YAML files.

## Best Practices

- Never commit `llmstack.yaml` files that contain plain-text API keys.
  Use `api_key_env` references instead.
- Run `llmstack env-check` to scan for leaked secrets in `.env` files.
- Run `llmstack security <directory>` for an AI-powered security audit
  of your codebase.
- Keep LLMStack and its dependencies up to date.
- When exposing the gateway to the internet, always enable API key
  authentication and set a restrictive `cors` origin list.
