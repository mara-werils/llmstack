"""llmstack complexity — Analyze code complexity and maintainability."""

from __future__ import annotations

import json
from pathlib import Path

from llmstack.cli.console import console


def complexity(
    target: str | None = None,
    threshold: int = 10,
    sort_by: str = "complexity",
    output: str | None = None,
    show_all: bool = False,
) -> None:
    """Analyze code complexity."""
    from rich.table import Table
    from rich.panel import Panel
    from llmstack.analyze.complexity import analyze_directory, analyze_python_file

    target_path = Path(target) if target else Path.cwd()

    if target_path.is_file():
        from llmstack.analyze.complexity import analyze_python_file

        metrics = analyze_python_file(target_path)
        if not metrics:
            console.print("[error]Could not analyze file.[/]")
            return
        file_results = [metrics]
    else:
        file_results = analyze_directory(target_path, threshold)

    if not file_results:
        console.print("[warning]No Python files found to analyze.[/]")
        return

    console.print()
    console.print(f"[bold]llmstack complexity[/]  threshold=[dim]{threshold}[/]")
    console.print()

    # Collect all functions above threshold (or all if show_all)
    all_functions = []
    for fm in file_results:
        for func in fm.function_metrics:
            if show_all or func.cyclomatic >= threshold:
                all_functions.append(func)

    # Sort
    sort_keys = {
        "complexity": lambda f: -f.cyclomatic,
        "cognitive": lambda f: -f.cognitive,
        "lines": lambda f: -f.lines_of_code,
        "name": lambda f: f.name,
    }
    all_functions.sort(key=sort_keys.get(sort_by, sort_keys["complexity"]))

    # Function table
    if all_functions:
        grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "bold red"}

        table = Table(
            title=f"Complex Functions (CC ≥ {threshold})" if not show_all else "All Functions",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Grade", width=5)
        table.add_column("Function", style="bold")
        table.add_column("CC", justify="right", width=4)
        table.add_column("Cog", justify="right", width=4)
        table.add_column("LOC", justify="right", width=5)
        table.add_column("Params", justify="right", width=6)
        table.add_column("Depth", justify="right", width=5)
        table.add_column("File:Line")

        for func in all_functions[:50]:
            color = grade_colors.get(func.grade, "white")
            rel_file = func.file
            try:
                rel_file = str(Path(func.file).relative_to(target_path))
            except ValueError:
                pass

            table.add_row(
                f"[{color}]{func.grade}[/]",
                func.name,
                str(func.cyclomatic),
                str(func.cognitive),
                str(func.lines_of_code),
                str(func.parameters),
                str(func.nested_depth),
                f"{rel_file}:{func.line}",
            )

        console.print(table)

    # File summary
    file_table = Table(
        title="File Summary",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    file_table.add_column("Grade", width=5)
    file_table.add_column("File", style="bold")
    file_table.add_column("LOC", justify="right", width=6)
    file_table.add_column("Functions", justify="right", width=9)
    file_table.add_column("Avg CC", justify="right", width=6)
    file_table.add_column("Max CC", justify="right", width=6)
    file_table.add_column("MI", justify="right", width=6)

    grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "bold red"}

    for fm in sorted(file_results, key=lambda x: -x.max_complexity)[:30]:
        color = grade_colors.get(fm.grade, "white")
        rel_file = fm.file
        try:
            rel_file = str(Path(fm.file).relative_to(target_path))
        except ValueError:
            pass

        file_table.add_row(
            f"[{color}]{fm.grade}[/]",
            rel_file[:50],
            str(fm.code_lines),
            str(fm.functions),
            str(fm.avg_complexity),
            str(fm.max_complexity),
            str(fm.maintainability_index),
        )

    console.print()
    console.print(file_table)

    # Overall summary
    total_functions = sum(fm.functions for fm in file_results)
    total_loc = sum(fm.code_lines for fm in file_results)
    total_classes = sum(fm.classes for fm in file_results)
    avg_mi = sum(fm.maintainability_index for fm in file_results) / max(1, len(file_results))
    complex_count = sum(1 for f in all_functions if f.cyclomatic >= threshold)

    overall_grade = (
        "A"
        if avg_mi >= 80
        else "B"
        if avg_mi >= 60
        else "C"
        if avg_mi >= 40
        else "D"
        if avg_mi >= 20
        else "F"
    )
    grade_color = grade_colors.get(overall_grade, "white")

    console.print()
    console.print(
        Panel(
            f"[bold]Overall Grade:[/] [{grade_color}]{overall_grade}[/]\n"
            f"[bold]Files:[/] {len(file_results)}  |  [bold]Classes:[/] {total_classes}  |  [bold]Functions:[/] {total_functions}\n"
            f"[bold]Lines of Code:[/] {total_loc:,}\n"
            f"[bold]Avg Maintainability:[/] {avg_mi:.1f}/100\n"
            f"[bold]Complex functions (CC≥{threshold}):[/] {complex_count}",
            title="Complexity Summary",
            border_style=grade_color,
        )
    )

    if output:
        data = {
            "files": [
                {
                    "file": fm.file,
                    "grade": fm.grade,
                    "code_lines": fm.code_lines,
                    "functions": fm.functions,
                    "avg_complexity": fm.avg_complexity,
                    "max_complexity": fm.max_complexity,
                    "maintainability_index": fm.maintainability_index,
                }
                for fm in file_results
            ],
            "summary": {
                "grade": overall_grade,
                "total_files": len(file_results),
                "total_loc": total_loc,
                "total_functions": total_functions,
                "avg_maintainability": round(avg_mi, 1),
                "complex_functions": complex_count,
            },
        }
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]Report saved to {output}[/]")
