"""llmstack chat — interactive terminal chat with your local LLM."""

from __future__ import annotations

import json
import sys

import httpx

from llmstack.cli.console import console
from llmstack.config.loader import load_config


def chat(model: str | None = None) -> None:
    """Start an interactive chat session with the running LLM."""
    config = load_config()
    base_url = f"http://localhost:{config.gateway.port}"
    model_name = model or config.models.chat.name

    # Get API key
    api_key = config.gateway.api_keys[0] if config.gateway.api_keys else ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Verify gateway is reachable
    try:
        resp = httpx.get(f"{base_url}/healthz", headers=headers, timeout=5)
        if resp.status_code != 200:
            console.print("[error]Gateway is not healthy. Run 'llmstack up' first.[/]")
            sys.exit(1)
    except httpx.ConnectError:
        console.print("[error]Cannot connect to gateway. Run 'llmstack up' first.[/]")
        sys.exit(1)

    # Initialize learning feedback collector
    from llmstack.learn.collector import FeedbackCollector

    collector = FeedbackCollector()

    console.print(f"\n[bold]LLMStack Chat[/] — model: [cyan]{model_name}[/]")
    console.print("[dim]Type 'exit' or Ctrl+C to quit. '/clear' to reset. '/fb' for feedback.[/]\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = console.input("[bold green]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            collector.close()
            console.print("\n[dim]Goodbye![/]")
            break

        text = user_input.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            collector.close()
            console.print("[dim]Goodbye![/]")
            break
        if text == "/clear":
            messages.clear()
            console.print("[dim]Conversation cleared.[/]\n")
            continue

        # Feedback commands
        if text.startswith("/fb"):
            fb_input = text[3:].strip() if len(text) > 3 else ""
            if not fb_input:
                console.print("[dim]Feedback: y=good, n=bad, c:text=correction, s=skip[/]")
                try:
                    fb_input = console.input("[dim]→ [/]").strip()
                except (KeyboardInterrupt, EOFError):
                    continue
            result = collector.parse_feedback_input(fb_input)
            if result:
                console.print(f"[dim]✓ Recorded {result.feedback_type.value}[/]")
            else:
                console.print("[dim]Skipped.[/]")
            continue

        if text == "/learn":
            stats = collector.get_stats()
            console.print(f"[dim]Learning: {stats['total_stored']} total, {stats['pending']} pending[/]")
            continue

        messages.append({"role": "user", "content": text})

        # Stream response
        console.print("[bold cyan]Assistant:[/] ", end="")
        assistant_text = ""

        try:
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={"model": model_name, "messages": messages, "stream": True},
                timeout=httpx.Timeout(300, connect=10),
            ) as resp:
                if resp.status_code != 200:
                    console.print(f"[error]Error {resp.status_code}[/]")
                    messages.pop()
                    continue

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            print(token, end="", flush=True)
                            assistant_text += token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        except httpx.ConnectError:
            console.print("[error]Connection lost.[/]")
            messages.pop()
            continue
        except KeyboardInterrupt:
            pass

        print()  # newline after streamed response
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            collector.record_interaction(
                query=text, response=assistant_text,
                model=model_name, command="chat",
            )

        # Periodic feedback prompt
        if collector.should_prompt():
            console.print("[dim]Was this helpful? (y/n/c:correction/s=skip):[/] ", end="")
            try:
                fb_input = console.input("").strip()
                result = collector.parse_feedback_input(fb_input)
                if result:
                    console.print(f"[dim]✓ {result.feedback_type.value}[/]")
            except (KeyboardInterrupt, EOFError):
                pass

        print()
