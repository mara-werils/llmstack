"""Guardrails — input/output content safety filtering for LLM requests.

Provides configurable content moderation rules that can block or flag
requests and responses based on pattern matching, keyword detection,
PII detection, and custom rules.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


class GuardrailAction(str, Enum):
    BLOCK = "block"     # Block the request/response entirely
    FLAG = "flag"       # Allow but flag for review
    REDACT = "redact"   # Redact matched content


class GuardrailTarget(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


@dataclass
class GuardrailRule:
    """A single content filtering rule."""

    id: str = ""
    name: str = ""
    description: str = ""
    pattern: str = ""              # regex pattern
    keywords: list[str] = field(default_factory=list)
    action: GuardrailAction = GuardrailAction.BLOCK
    target: GuardrailTarget = GuardrailTarget.BOTH
    enabled: bool = True
    priority: int = 0              # higher = checked first
    category: str = "custom"
    message: str = ""              # custom block message

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""

    rule_id: str
    rule_name: str
    action: GuardrailAction
    target: str
    matched_text: str
    category: str
    timestamp: float = 0.0
    request_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "action": self.action.value,
            "target": self.target,
            "matched_text": self.matched_text[:100],
            "category": self.category,
            "timestamp": self.timestamp,
        }


# Built-in PII patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "api_key": r"\b(?:sk-|pk-|key-|api[_-]?key[=:]\s*)[a-zA-Z0-9_-]{20,}\b",
}

# Built-in harmful content keywords
HARMFUL_KEYWORDS = [
    "ignore previous instructions",
    "disregard all previous",
    "forget your instructions",
    "you are now",
    "pretend you are",
    "jailbreak",
    "DAN mode",
]


class GuardrailEngine:
    """Content safety engine with configurable rules and PII detection."""

    def __init__(self):
        self._lock = Lock()
        self._rules: dict[str, GuardrailRule] = {}
        self._violations: list[GuardrailViolation] = []
        self._stats = {"checked": 0, "blocked": 0, "flagged": 0, "redacted": 0}

    def add_rule(self, rule: GuardrailRule) -> None:
        """Add or update a guardrail rule."""
        with self._lock:
            self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def get_rules(self) -> list[GuardrailRule]:
        """Get all rules sorted by priority."""
        with self._lock:
            return sorted(
                self._rules.values(),
                key=lambda r: r.priority,
                reverse=True,
            )

    def load_defaults(self) -> None:
        """Load built-in PII and prompt injection rules."""
        # PII detection rules
        for name, pattern in PII_PATTERNS.items():
            self.add_rule(GuardrailRule(
                name=f"pii-{name}",
                description=f"Detect {name.replace('_', ' ')} in content",
                pattern=pattern,
                action=GuardrailAction.REDACT,
                target=GuardrailTarget.OUTPUT,
                category="pii",
                priority=10,
            ))

        # Prompt injection detection
        self.add_rule(GuardrailRule(
            name="prompt-injection",
            description="Detect common prompt injection attempts",
            keywords=HARMFUL_KEYWORDS,
            action=GuardrailAction.BLOCK,
            target=GuardrailTarget.INPUT,
            category="injection",
            priority=100,
            message="Request blocked: potential prompt injection detected",
        ))

    def check(
        self,
        content: str,
        target: GuardrailTarget,
        request_id: str = "",
    ) -> tuple[str, list[GuardrailViolation]]:
        """Check content against all rules.

        Returns ``(processed_content, violations)``.
        If any rule blocks, raises GuardrailBlockError.
        If any rule redacts, content is modified.
        """
        violations: list[GuardrailViolation] = []
        processed = content

        with self._lock:
            self._stats["checked"] += 1
            rules = sorted(
                self._rules.values(),
                key=lambda r: r.priority,
                reverse=True,
            )

        for rule in rules:
            if not rule.enabled:
                continue
            if rule.target not in (target, GuardrailTarget.BOTH):
                continue

            matches = self._find_matches(processed, rule)
            if not matches:
                continue

            for match_text in matches:
                violation = GuardrailViolation(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    action=rule.action,
                    target=target.value,
                    matched_text=match_text,
                    category=rule.category,
                    request_id=request_id,
                )
                violations.append(violation)

            with self._lock:
                self._violations.extend(violations)

            if rule.action == GuardrailAction.BLOCK:
                with self._lock:
                    self._stats["blocked"] += 1
                raise GuardrailBlockError(
                    rule.message or f"Content blocked by rule: {rule.name}",
                    violations=violations,
                )
            elif rule.action == GuardrailAction.REDACT:
                processed = self._redact(processed, rule)
                with self._lock:
                    self._stats["redacted"] += 1
            elif rule.action == GuardrailAction.FLAG:
                with self._lock:
                    self._stats["flagged"] += 1

        return processed, violations

    def check_input(self, messages: list[dict], request_id: str = "") -> list[dict]:
        """Check all user messages in a chat payload."""
        processed = []
        for msg in messages:
            if msg.get("role") == "user":
                content, _ = self.check(
                    msg.get("content", ""),
                    GuardrailTarget.INPUT,
                    request_id=request_id,
                )
                processed.append({**msg, "content": content})
            else:
                processed.append(msg)
        return processed

    def check_output(self, content: str, request_id: str = "") -> str:
        """Check assistant response content."""
        processed, _ = self.check(
            content, GuardrailTarget.OUTPUT, request_id=request_id,
        )
        return processed

    def get_violations(self, limit: int = 100) -> list[GuardrailViolation]:
        """Get recent violations."""
        with self._lock:
            return list(reversed(self._violations[-limit:]))

    def get_stats(self) -> dict:
        """Get guardrail statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_rules": len(self._rules),
                "active_rules": sum(1 for r in self._rules.values() if r.enabled),
                "total_violations": len(self._violations),
            }

    @staticmethod
    def _find_matches(content: str, rule: GuardrailRule) -> list[str]:
        """Find content that matches a rule."""
        matches = []
        content_lower = content.lower()

        # Check keywords
        for kw in rule.keywords:
            if kw.lower() in content_lower:
                matches.append(kw)

        # Check regex pattern
        if rule.pattern:
            try:
                for m in re.finditer(rule.pattern, content, re.IGNORECASE):
                    matches.append(m.group())
            except re.error:
                pass

        return matches

    @staticmethod
    def _redact(content: str, rule: GuardrailRule) -> str:
        """Redact matched content."""
        result = content
        if rule.pattern:
            try:
                result = re.sub(rule.pattern, "[REDACTED]", result, flags=re.IGNORECASE)
            except re.error:
                pass
        for kw in rule.keywords:
            result = re.sub(re.escape(kw), "[REDACTED]", result, flags=re.IGNORECASE)
        return result


class GuardrailBlockError(Exception):
    """Raised when content is blocked by a guardrail rule."""

    def __init__(self, message: str, violations: list[GuardrailViolation] | None = None):
        super().__init__(message)
        self.violations = violations or []
