"""Export the OpenAPI spec from the LLMStack gateway."""

from __future__ import annotations

import json
from pathlib import Path

from llmstack.cli.console import console, success, info


def openapi_export(output: str = "", pretty: bool = True) -> None:
    """Export the OpenAPI JSON spec from the gateway."""
    from llmstack.gateway.main import create_app

    app = create_app()
    spec = app.openapi()

    if output:
        path = Path(output)
        indent = 2 if pretty else None
        path.write_text(json.dumps(spec, indent=indent))
        success(f"OpenAPI spec exported to {path}")
        info(f"Endpoints: {len(spec.get('paths', {}))}")
    else:
        console.print_json(json.dumps(spec))
