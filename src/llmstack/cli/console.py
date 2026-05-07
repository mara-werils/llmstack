"""Rich console singleton and display helpers."""

from rich.console import Console
from rich.theme import Theme

theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
})

console = Console(theme=theme)
