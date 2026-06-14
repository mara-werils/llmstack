# Homebrew distribution

llmstack ships a Homebrew formula so macOS and Linux users can install the CLI
with a single command and get updates through `brew upgrade`.

## Install

```bash
brew install mara-werils/llmstack/llmstack
```

This is shorthand for tapping `mara-werils/homebrew-llmstack` and installing the
`llmstack` formula. The formula installs the CLI and its dependencies into an
isolated Homebrew-managed virtualenv and links the `llmstack` entry point.

## How the formula stays current

`llmstack.rb` in this directory is the source of truth. On every `v*` tag, the
`update-homebrew-tap` job in `.github/workflows/release.yml`:

1. Downloads the GitHub release tarball for the tag.
2. Computes its `sha256`.
3. Rewrites the `url` and `sha256` lines.
4. Pushes the rendered formula to the `mara-werils/homebrew-llmstack` tap repo
   (`Formula/llmstack.rb`).

So the `sha256` checked in here is a placeholder — the published tap always
carries the real digest for the tagged release.

## One-time setup (maintainers)

1. Create a public repo named **`homebrew-llmstack`** under the same owner.
2. Add a repository secret **`HOMEBREW_TAP_TOKEN`** — a PAT with `contents:write`
   on that tap repo. Without it, the release job logs a skip and does nothing.

## Manual install from source (no tap)

```bash
brew install --build-from-source ./packaging/homebrew/llmstack.rb
```
