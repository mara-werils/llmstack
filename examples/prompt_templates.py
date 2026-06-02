"""Example: Using prompt templates for common development tasks.

Usage:
    python examples/prompt_templates.py
"""

from llmstack.prompts.templates import TemplateManager


def main():
    mgr = TemplateManager()

    # List built-in templates
    print("Built-in templates:")
    for t in mgr.list_all():
        if t.is_builtin:
            print(f"  {t.name:20s} [{t.category}] {t.description}")

    # Render a built-in template
    print("\n--- Rendered 'code-review' template ---")
    rendered = mgr.render(
        "code-review",
        language="python",
        code="def add(a, b): return a + b",
        focus="correctness and edge cases",
    )
    print(rendered)

    # Render debug template
    print("\n--- Rendered 'debug' template ---")
    rendered = mgr.render(
        "debug",
        error="TypeError: cannot unpack non-sequence NoneType",
        language="python",
        code="name, age = get_user_info(user_id)",
    )
    print(rendered)

    # Create a custom template
    print("\n--- Creating custom template ---")
    custom = mgr.save(
        name="pr-description",
        template="""Write a pull request description for these changes:

## What changed
{{changes}}

## Why
{{reason}}

## Testing
{{testing|Manual testing}}

Use markdown format with bullet points.""",
        description="Generate PR descriptions",
        category="git",
    )
    print(f"Created: {custom.name} (variables: {custom.variables})")

    # Use custom template
    rendered = mgr.render(
        "pr-description",
        changes="Added user authentication with JWT tokens",
        reason="Security requirement for API access control",
        testing="Unit tests + integration tests with test JWT tokens",
    )
    print("\n--- Rendered custom template ---")
    print(rendered)


if __name__ == "__main__":
    main()
