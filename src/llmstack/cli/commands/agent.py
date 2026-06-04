"""CLI command: llmstack agent — run AI agents with tool use."""

from __future__ import annotations

import asyncio
import sys

from llmstack.cli.console import console


def agent(
    task: str,
    model: str | None = None,
    ollama_url: str = "http://localhost:11434",
    max_steps: int = 25,
    tools: str | None = None,
    working_dir: str = ".",
    verbose: bool = False,
) -> None:
    """Run an AI agent that uses tools to complete a task."""
    asyncio.run(
        _agent_async(
            task=task,
            model=model,
            ollama_url=ollama_url,
            max_steps=max_steps,
            tools_filter=tools,
            working_dir=working_dir,
            verbose=verbose,
        )
    )


async def _agent_async(
    task: str,
    model: str | None,
    ollama_url: str,
    max_steps: int,
    tools_filter: str | None,
    working_dir: str,
    verbose: bool,
) -> None:
    from pathlib import Path

    import httpx
    from rich.markdown import Markdown
    from rich.panel import Panel

    from llmstack.agent.loop import AgentConfig, AgentLoop
    from llmstack.agent.tools import create_default_registry

    model_name = model or "llama3.2"

    # Check Ollama connectivity
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(ollama_url)
            resp.raise_for_status()
    except Exception:
        console.print(
            Panel(
                f"[error]Cannot connect to Ollama at {ollama_url}[/]\n\n"
                "Make sure Ollama is running:\n"
                "  [info]ollama serve[/]\n"
                "  [info]ollama pull {model_name}[/]",
                title="Connection Error",
                border_style="red",
            )
        )
        sys.exit(1)

    # Set up tools
    cwd = str(Path(working_dir).resolve())
    registry = create_default_registry(cwd)

    # Filter tools if specified
    if tools_filter:
        allowed = set(tools_filter.split(","))
        for name in list(registry.names()):
            if name not in allowed:
                registry._tools.pop(name, None)

    config = AgentConfig(
        model=model_name,
        api_url=ollama_url,
        max_steps=max_steps,
        temperature=0.1,
    )

    agent_loop = AgentLoop(config=config, tools=registry)

    # Display header
    tool_names = ", ".join(registry.names())
    console.print(
        Panel(
            f"[bold]Task:[/] {task}\n"
            f"[bold]Model:[/] {model_name}\n"
            f"[bold]Tools:[/] {tool_names}\n"
            f"[bold]Max steps:[/] {max_steps}",
            title="LLMStack Agent",
            border_style="cyan",
        )
    )
    console.print()

    # Run agent and display events
    final_answer = ""
    async for event in agent_loop.run(task):
        if event.type == "tool_call":
            args_str = ""
            for k, v in event.tool_args.items():
                val = str(v)
                if len(val) > 80:
                    val = val[:80] + "..."
                args_str += f"\n    {k}: {val}"

            console.print(
                f"  [info]Step {event.step}[/] [bold]Tool:[/] {event.tool_name}{args_str}"
            )

        elif event.type == "tool_result":
            output = event.content
            if len(output) > 500:
                output = output[:500] + f"\n... ({len(event.content)} chars total)"

            style = "green" if "Error:" not in output else "red"
            lines = output.split("\n")
            if len(lines) > 10:
                display = "\n".join(lines[:10]) + f"\n... ({len(lines)} lines total)"
            else:
                display = output

            console.print(f"  [{style}]{display}[/{style}]")
            console.print()

        elif event.type == "message":
            final_answer = event.content
            # Don't print yet — show at the end

        elif event.type == "error":
            console.print(f"  [error]Error: {event.content}[/]")

        elif event.type == "done":
            pass

    # Show final answer
    if final_answer:
        console.print()
        console.print(
            Panel(
                Markdown(final_answer),
                title="Agent Response",
                border_style="green",
            )
        )

    # Show summary
    console.print(f"\n[info]Completed in {agent_loop.steps_taken} steps[/]")
