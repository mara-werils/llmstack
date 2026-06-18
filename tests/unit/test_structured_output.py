"""Tests for structured output validation."""

from llmstack.gateway.structured_output import (
    ValidationResult,
    build_schema_prompt,
    extract_json,
    validate_output,
    validate_type,
)


class TestExtractJson:
    def test_direct_json(self):
        assert extract_json('{"key": "value"}') == '{"key": "value"}'

    def test_direct_array(self):
        assert extract_json("[1, 2, 3]") == "[1, 2, 3]"

    def test_markdown_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        assert extract_json(text) == '{"key": "value"}'

    def test_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        assert extract_json(text) == '{"key": "value"}'

    def test_embedded_json(self):
        text = 'The answer is {"result": 42} as expected.'
        assert extract_json(text) == '{"result": 42}'

    def test_no_json(self):
        assert extract_json("Just plain text, no JSON here.") is None


class TestValidateType:
    def test_string_valid(self):
        assert validate_type("hello", {"type": "string"}) == []

    def test_string_invalid(self):
        errors = validate_type(42, {"type": "string"})
        assert len(errors) == 1

    def test_string_min_length(self):
        errors = validate_type("ab", {"type": "string", "minLength": 3})
        assert len(errors) == 1

    def test_string_enum(self):
        errors = validate_type("c", {"type": "string", "enum": ["a", "b"]})
        assert len(errors) == 1

    def test_integer_valid(self):
        assert validate_type(42, {"type": "integer"}) == []

    def test_integer_minimum(self):
        errors = validate_type(3, {"type": "integer", "minimum": 5})
        assert len(errors) == 1

    def test_array_valid(self):
        assert validate_type([1, 2], {"type": "array"}) == []

    def test_array_min_items(self):
        errors = validate_type([], {"type": "array", "minItems": 1})
        assert len(errors) == 1

    def test_object_required_missing(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        errors = validate_type({}, schema)
        assert any("Missing required" in e for e in errors)

    def test_object_property_type(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        errors = validate_type({"age": "not_a_number"}, schema)
        assert len(errors) == 1

    def test_nested_array_items(self):
        schema = {"type": "array", "items": {"type": "string"}}
        errors = validate_type(["a", 1, "b"], schema)
        assert len(errors) == 1  # items[1] is invalid

    def test_no_type_specified_returns_no_errors(self):
        assert validate_type("anything", {}) == []

    def test_string_max_length(self):
        errors = validate_type("hello world", {"type": "string", "maxLength": 5})
        assert len(errors) == 1

    def test_string_pattern_mismatch(self):
        errors = validate_type("abc", {"type": "string", "pattern": r"^\d+$"})
        assert len(errors) == 1

    def test_integer_maximum(self):
        errors = validate_type(99, {"type": "integer", "maximum": 10})
        assert len(errors) == 1

    def test_array_max_items(self):
        errors = validate_type([1, 2, 3], {"type": "array", "maxItems": 2})
        assert len(errors) == 1


class TestValidateOutput:
    def test_valid_output(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        result = validate_output('{"name": "Alice"}', schema)
        assert result.valid is True
        assert result.data == {"name": "Alice"}

    def test_invalid_json(self):
        result = validate_output("not json", {"type": "object"})
        assert result.valid is False
        assert len(result.errors) > 0

    def test_schema_violation(self):
        schema = {"type": "object", "required": ["x"]}
        result = validate_output("{}", schema)
        assert result.valid is False

    def test_markdown_extraction(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        result = validate_output('```json\n{"n": 42}\n```', schema)
        assert result.valid is True
        assert result.data["n"] == 42

    def test_malformed_json_content(self):
        result = validate_output("{not: valid, json}", {"type": "object"})
        assert result.valid is False
        assert any("Invalid JSON" in e for e in result.errors)

    def test_to_dict(self):
        result = ValidationResult(valid=True, data={"a": 1}, errors=[])
        assert result.to_dict() == {"valid": True, "data": {"a": 1}, "errors": []}


class TestBuildSchemaPrompt:
    def test_produces_prompt(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        prompt = build_schema_prompt(schema)
        assert "JSON" in prompt
        assert "schema" in prompt
        assert '"name"' in prompt
