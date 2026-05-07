"""Plugin specification.

To create an llmstack plugin:

1. Create a class that extends ServiceBase
2. Set `name` and `category` class attributes
3. Implement: container_spec(), health_url()
4. Optionally implement: post_start(), openai_base_url()
5. Register via entry_points in pyproject.toml:

    [project.entry-points."llmstack.services"]
    my_service = "my_package:MyService"

6. Publish to PyPI. Users install with:
    pip install llmstack-plugin-myservice
"""

from llmstack.services.base import ServiceBase

__all__ = ["ServiceBase"]
