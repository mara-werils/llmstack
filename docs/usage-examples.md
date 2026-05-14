# Usage Examples

## llmstack review

Review the last commit:
```bash
llmstack review
```

Review staged changes:
```bash
llmstack review --staged
```

Review last 3 commits:
```bash
llmstack review --commits 3
```

Review a GitHub PR:
```bash
llmstack review --pr https://github.com/owner/repo/pull/42
```

Only show CRITICAL issues:
```bash
llmstack review --severity CRITICAL
```

Export review as JSON:
```bash
llmstack review --output json --output-file review.json
```

## llmstack fix

Fix an issue in a file:
```bash
llmstack fix "null pointer dereference in main loop" --file src/main.py
```

Preview the patch without applying:
```bash
llmstack fix "missing error handling" --file api.py --dry-run
```

## llmstack docs

Generate docstrings for a file:
```bash
llmstack docs src/mymodule.py --write
```

Generate a README for the project:
```bash
llmstack docs --type readme --write
```

## llmstack test

Generate tests for a file:
```bash
llmstack test src/utils.py
```

Save tests to disk:
```bash
llmstack test src/utils.py --write
```

## llmstack security

Audit the current directory:
```bash
llmstack security
```

Audit a specific file and save report:
```bash
llmstack security src/api.py --output markdown --output-file security-report.md
```

Only show HIGH and CRITICAL:
```bash
llmstack security --severity HIGH
```

## llmstack diff

Explain the last commit:
```bash
llmstack diff
```

Explain staged changes:
```bash
llmstack diff --staged
```

Explain a specific file's changes:
```bash
llmstack diff --file src/main.py
```

## llmstack watch

Watch current directory for changes:
```bash
llmstack watch
```

Watch specific patterns with custom debounce:
```bash
llmstack watch src/ --patterns "*.py,*.yaml" --debounce 3.0
```

## llmstack commit

Generate a commit message for staged changes:
```bash
llmstack commit
```

Stage everything and generate + push:
```bash
llmstack commit --all --push
```

## llmstack export-conv

Export conversation history as Markdown:
```bash
llmstack export-conv
```

Export as JSON:
```bash
llmstack export-conv --format json --output history.json
```
