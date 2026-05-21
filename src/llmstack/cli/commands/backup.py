"""Backup and restore — export/import all LLMStack configuration and data."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from llmstack.cli.console import console

DEFAULT_DATA_DIR = Path.home() / ".llmstack"
BACKUP_MANIFEST = "backup_manifest.json"


def _collect_files(data_dir: Path) -> list[Path]:
    """Collect all files to backup from the data directory."""
    files = []
    for pattern in ["*.yaml", "*.yml", "*.json", "*.db", "*.jsonl"]:
        files.extend(data_dir.glob(f"**/{pattern}"))
    return sorted(files)


def backup(
    output: str | None = None,
    data_dir: str | None = None,
    include_models: bool = False,
) -> None:
    """Create a backup of all LLMStack configuration and data.

    Args:
        output: Output path for the backup archive (.tar.gz)
        data_dir: Custom data directory (default: ~/.llmstack)
        include_models: Whether to include model files (can be large)
    """
    src = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    if not src.exists():
        console.print("[yellow]No LLMStack data directory found. Nothing to backup.[/]")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output) if output else Path(f"llmstack_backup_{timestamp}")

    # Create temp directory for backup contents
    backup_dir = output_path.with_suffix("")
    backup_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_files(src)
    if not files:
        console.print("[yellow]No configuration files found to backup.[/]")
        return

    console.print(f"[bold]Creating backup from {src}...[/]")

    copied = []
    for f in files:
        rel = f.relative_to(src)
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied.append(str(rel))

    # Also backup llmstack.yaml from current directory if it exists
    local_config = Path("llmstack.yaml")
    if local_config.exists():
        shutil.copy2(local_config, backup_dir / "llmstack.yaml")
        copied.append("llmstack.yaml")

    # Write manifest
    manifest = {
        "version": "1.0",
        "timestamp": timestamp,
        "source": str(src),
        "files": copied,
        "total_files": len(copied),
        "include_models": include_models,
    }
    (backup_dir / BACKUP_MANIFEST).write_text(json.dumps(manifest, indent=2))

    # Create archive
    archive_path = shutil.make_archive(str(backup_dir), "gztar", str(backup_dir.parent), backup_dir.name)

    # Clean up temp dir
    shutil.rmtree(backup_dir)

    console.print(f"[green]Backup created: {archive_path}[/]")
    console.print(f"  Files: {len(copied)}")
    console.print(f"  Size: {Path(archive_path).stat().st_size / 1024:.1f} KB")


def restore(
    archive: str,
    data_dir: str | None = None,
    force: bool = False,
) -> None:
    """Restore LLMStack configuration and data from a backup archive.

    Args:
        archive: Path to the backup archive (.tar.gz)
        data_dir: Target data directory (default: ~/.llmstack)
        force: Overwrite existing files without confirmation
    """
    archive_path = Path(archive)
    if not archive_path.exists():
        console.print(f"[red]Backup archive not found: {archive}[/]")
        return

    dest = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    console.print(f"[bold]Restoring backup from {archive}...[/]")

    # Extract to temp directory
    import tarfile
    temp_dir = Path(f"/tmp/llmstack_restore_{int(time.time())}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(temp_dir)

    # Find the backup root (first directory in archive)
    extracted_dirs = list(temp_dir.iterdir())
    if not extracted_dirs:
        console.print("[red]Empty backup archive.[/]")
        shutil.rmtree(temp_dir)
        return

    backup_root = extracted_dirs[0] if extracted_dirs[0].is_dir() else temp_dir

    # Read manifest
    manifest_path = backup_root / BACKUP_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        console.print(f"  Backup date: {manifest.get('timestamp', 'unknown')}")
        console.print(f"  Files: {manifest.get('total_files', 'unknown')}")
    else:
        manifest = {}

    # Restore files
    dest.mkdir(parents=True, exist_ok=True)
    restored = 0
    for item in backup_root.rglob("*"):
        if item.is_file() and item.name != BACKUP_MANIFEST:
            rel = item.relative_to(backup_root)
            target = dest / rel
            if target.exists() and not force:
                console.print(f"  [yellow]Skipping existing: {rel}[/]")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            restored += 1

    # Clean up
    shutil.rmtree(temp_dir)

    console.print(f"[green]Restored {restored} files to {dest}[/]")


def list_backups(directory: str = ".") -> None:
    """List available backup files in a directory."""
    search_dir = Path(directory)
    backups = sorted(search_dir.glob("llmstack_backup_*.tar.gz"), reverse=True)

    if not backups:
        console.print("[yellow]No backup files found.[/]")
        return

    console.print(f"[bold]Found {len(backups)} backup(s):[/]")
    for b in backups:
        size_kb = b.stat().st_size / 1024
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(b.stat().st_mtime))
        console.print(f"  {b.name}  ({size_kb:.1f} KB, {mtime})")
