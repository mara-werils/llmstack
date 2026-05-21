"""Structured output validation — JSON schema validation for LLM responses.

Validates that LLM outputs conform to a provided JSON schema,
with automatic retry and repair capabilities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result of a structured output validation."""

    valid: bool = False
    data: Any = None
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""
    extracted_json: str = ""

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "data": self.data,
            "errors": self.errors,
        }


def extract_json(text: str) -> str | None:
    """Extract JSON from LLM output, handling markdown code blocks."""
    # Try direct JSON parse first
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    # Try markdown code blocks
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching end
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


def validate_type(value: Any, schema: dict) -> list[str]:
    """Validate a value against a JSON schema type definition."""
    errors = []
    schema_type = schema.get("type")

    if schema_type is None:
        return errors

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    expected = type_map.get(schema_type)
    if expected and not isinstance(value, expected):
        errors.append(f"Expected {schema_type}, got {type(value).__name__}")
        return errors

    # String validations
    if schema_type == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"String too short: {len(value)} < {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"String too long: {len(value)} > {schema['maxLength']}")
        if "pattern" in schema:
            if not re.match(schema["pattern"], value):
                errors.append(f"String doesn't match pattern: {schema['pattern']}")
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"Value '{value}' not in enum: {schema['enum']}")

    # Number validations
    if schema_type in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"Value {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"Value {value} > maximum {schema['maximum']}")

    # Array validations
    if schema_type == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"Array too short: {len(value)} < {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"Array too long: {len(value)} > {schema['maxItems']}")
        if "items" in schema:
            for i, item in enumerate(value):
                item_errors = validate_type(item, schema["items"])
                for e in item_errors:
                    errors.append(f"items[{i}]: {e}")

    # Object validations
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for req_field in required:
            if req_field not in value:
                errors.append(f"Missing required field: {req_field}")

        for prop_name, prop_schema in properties.items():
            if prop_name in value:
                prop_errors = validate_type(value[prop_name], prop_schema)
                for e in prop_errors:
                    errors.append(f"{prop_name}: {e}")

    return errors


def validate_output(
    output: str,
    schema: dict,
) -> ValidationResult:
    """Validate LLM output against a JSON schema.

    Handles markdown code blocks, extracts JSON, parses,
    and validates against the schema.
    """
    result = ValidationResult(raw_output=output)

    # Extract JSON
    json_str = extract_json(output)
    if json_str is None:
        result.errors.append("No JSON found in output")
        return result

    result.extracted_json = json_str

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON: {e}")
        return result

    result.data = data

    # Validate against schema
    errors = validate_type(data, schema)
    result.errors = errors
    result.valid = len(errors) == 0

    return result


def build_schema_prompt(schema: dict) -> str:
    """Build an instruction prompt for structured output generation."""
    schema_str = json.dumps(schema, indent=2)
    return (
        "You MUST respond with valid JSON that conforms to this schema:\n\n"
        f"```json\n{schema_str}\n```\n\n"
        "Important:\n"
        "- Respond ONLY with the JSON object, no other text\n"
        "- All required fields must be present\n"
        "- Field types must match the schema exactly"
    )
