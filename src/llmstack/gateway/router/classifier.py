"""Query complexity classifier for smart model routing.

Scores queries on a 0-1 scale using heuristic analysis of message content,
structure, and intent. No ML models required — all pattern-based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryProfile:
    """Result of query complexity classification."""

    score: float  # 0.0 (trivial) to 1.0 (expert)
    tier: str  # "simple" | "medium" | "complex"
    factors: dict  # {"token_count": 0.3, "task_markers": 0.7, ...}
    suggested_model: str | None = None  # Hint based on tier

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))
        if self.tier not in ("simple", "medium", "complex"):
            raise ValueError(f"Invalid tier: {self.tier}")


# ---------------------------------------------------------------------------
# Marker word sets
# ---------------------------------------------------------------------------

_SIMPLE_MARKERS = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank",
        "bye",
        "goodbye",
        "translate",
        "summarize",
        "summary",
        "define",
        "list",
        "tldr",
        "repeat",
        "count",
        "spell",
        "name",
        "greet",
        "yes",
        "no",
        "ok",
        "okay",
        "sure",
        "please",
    }
)

_COMPLEX_MARKERS = frozenset(
    {
        "analyze",
        "analyse",
        "compare",
        "contrast",
        "implement",
        "debug",
        "refactor",
        "optimise",
        "optimize",
        "evaluate",
        "explain how",
        "explain why",
        "explain the",
        "explain step by step",
        "step-by-step",
        "step by step",
        "critically",
        "critique",
        "assess",
        "investigate",
        "troubleshoot",
        "diagnose",
        "decompose",
        "synthesize",
        "trade-off",
        "tradeoff",
        "trade off",
        "implications",
        "advantages and disadvantages",
        "pros and cons",
        "including",
        "handling",
        "considering",
    }
)

_EXPERT_MARKERS = frozenset(
    {
        "write a full",
        "design a",
        "architect",
        "build a complete",
        "create a comprehensive",
        "develop a system",
        "distributed",
        "consensus",
        "fault-tolerant",
        "fault tolerant",
        "fault tolerance",
        "microservice",
        "end-to-end",
        "end to end",
        "production-ready",
        "production ready",
        "scalable",
        "high-availability",
        "high availability",
        "implement a",
        "design a system",
    }
)

_CODE_KEYWORDS = frozenset(
    {
        "def ",
        "class ",
        "import ",
        "function ",
        "const ",
        "let ",
        "var ",
        "return ",
        "async ",
        "await ",
        "for ",
        "while ",
        "if ",
        "struct ",
        "impl ",
        "fn ",
        "pub ",
        "enum ",
        "SELECT ",
        "INSERT ",
        "CREATE TABLE",
        "ALTER TABLE",
        "```",
        ">>>",
        "$ ",
        "#!/",
    }
)

_PROGRAMMING_TERMS = frozenset(
    {
        "algorithm",
        "recursion",
        "binary tree",
        "linked list",
        "hash map",
        "api",
        "endpoint",
        "database",
        "sql",
        "regex",
        "lambda",
        "closure",
        "decorator",
        "middleware",
        "callback",
        "polymorphism",
        "inheritance",
        "interface",
        "generic",
        "concurrency",
        "thread",
        "mutex",
        "semaphore",
        "deadlock",
        "docker",
        "kubernetes",
        "terraform",
        "cicd",
        "ci/cd",
        "webpack",
        "typescript",
        "python",
        "javascript",
        "rust",
        "golang",
        "neural network",
        "backpropagation",
        "gradient",
        "machine learning",
        "deep learning",
        "transformer",
        "embedding",
        "tokenizer",
        "graphql",
        "rest",
        "microservice",
        "architecture",
        "consensus",
        "replication",
        "fault tolerance",
        "leader election",
        "sorting",
        "caching",
        "pipeline",
        "compiler",
        "parser",
        "function",
        "class",
        "module",
        "package",
    }
)

# Simple question patterns
_SIMPLE_QUESTION_RE = re.compile(
    r"^(what is|what\'s|who is|who\'s|when did|where is|where\'s|how many|how old)\b",
    re.IGNORECASE,
)

# Complex question patterns
_COMPLEX_QUESTION_RE = re.compile(
    r"(how would you|how could|how might|how should|what approach|explain how|explain why|explain the|describe how|given .+ (and|with) .+ (how|what|why))",
    re.IGNORECASE,
)

# Multi-constraint pattern
_MULTI_CONSTRAINT_RE = re.compile(
    r"(given|assuming|considering|with the constraint|subject to|while ensuring|but also|and also|at the same time)",
    re.IGNORECASE,
)

# Enumeration / multi-part request
_MULTI_PART_RE = re.compile(
    r"(\d+\.\s|\b(first|second|third|then|next|finally|also|additionally)\b)",
    re.IGNORECASE,
)

# Non-ASCII / multi-language detection
_NON_LATIN_RE = re.compile(
    r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u0600-\u06FF]"
)


class QueryClassifier:
    """Classifies query complexity to route to the optimal model.

    All classification is heuristic-based — fast, deterministic, and
    requires no external models or API calls.
    """

    def classify(self, messages: list[dict]) -> QueryProfile:
        """Analyse a conversation and return a QueryProfile with complexity score."""
        if not messages:
            return QueryProfile(score=0.0, tier="simple", factors={})

        factors: dict[str, float] = {}

        # Separate system prompt and user/assistant messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1].get("content", "") if user_msgs else ""

        # --- Factor 1: Token count (approximate by whitespace split) ---
        token_count = len(last_user.split())
        if token_count <= 5:
            factors["token_count"] = 0.05
        elif token_count <= 15:
            factors["token_count"] = 0.2
        elif token_count <= 30:
            factors["token_count"] = 0.45
        elif token_count <= 60:
            factors["token_count"] = 0.6
        elif token_count <= 150:
            factors["token_count"] = 0.75
        else:
            factors["token_count"] = 0.9

        # --- Factor 2: Task markers ---
        lower_content = last_user.lower()
        marker_score = 0.3  # neutral default

        # Check expert markers first (multi-word phrases)
        expert_hits = sum(1 for m in _EXPERT_MARKERS if m in lower_content)
        if expert_hits:
            marker_score = min(1.0, 0.85 + expert_hits * 0.05)

        # Check complex markers (multi-word phrases first, then single words)
        complex_hits = sum(1 for m in _COMPLEX_MARKERS if m in lower_content)
        if complex_hits and marker_score < 0.7:
            marker_score = min(0.9, 0.55 + complex_hits * 0.12)

        # Check simple markers
        words = set(re.findall(r"\b\w+\b", lower_content))
        simple_hits = len(words & _SIMPLE_MARKERS)
        if simple_hits and complex_hits == 0 and expert_hits == 0:
            marker_score = max(0.05, 0.2 - simple_hits * 0.03)

        factors["task_markers"] = marker_score

        # --- Factor 3: Code detection ---
        code_score = 0.0
        code_keyword_hits = sum(1 for kw in _CODE_KEYWORDS if kw in last_user)
        prog_term_hits = sum(1 for t in _PROGRAMMING_TERMS if t in lower_content)
        code_block_count = last_user.count("```")

        if code_block_count >= 2:
            code_score = 0.7
        elif code_keyword_hits >= 3:
            code_score = 0.6
        elif code_keyword_hits >= 1:
            code_score = 0.4
        elif prog_term_hits >= 2:
            code_score = 0.35

        if prog_term_hits >= 3:
            code_score = min(1.0, code_score + 0.2)
        elif prog_term_hits >= 1 and code_score < 0.25:
            code_score = 0.25

        factors["code_detection"] = code_score

        # --- Factor 4: Conversation depth ---
        num_turns = len(user_msgs)
        if num_turns <= 1:
            factors["conversation_depth"] = 0.1
        elif num_turns <= 3:
            factors["conversation_depth"] = 0.3
        elif num_turns <= 6:
            factors["conversation_depth"] = 0.5
        else:
            factors["conversation_depth"] = 0.7

        # --- Factor 5: System prompt complexity ---
        system_text = " ".join(m.get("content", "") for m in system_msgs)
        sys_tokens = len(system_text.split())
        if sys_tokens == 0:
            factors["system_prompt"] = 0.0
        elif sys_tokens <= 50:
            factors["system_prompt"] = 0.15
        elif sys_tokens <= 200:
            factors["system_prompt"] = 0.35
        else:
            factors["system_prompt"] = 0.6

        # --- Factor 6: Language detection ---
        non_latin_chars = len(_NON_LATIN_RE.findall(last_user))
        has_latin = bool(re.search(r"[a-zA-Z]", last_user))
        if non_latin_chars > 0 and has_latin:
            factors["language_mix"] = 0.3  # multilingual
        elif non_latin_chars > 0:
            factors["language_mix"] = 0.15  # non-Latin only
        else:
            factors["language_mix"] = 0.0

        # --- Factor 7: Question complexity ---
        q_score = 0.25  # neutral
        if _SIMPLE_QUESTION_RE.search(last_user):
            q_score = 0.1
        if _COMPLEX_QUESTION_RE.search(last_user):
            q_score = 0.7
        constraint_hits = len(_MULTI_CONSTRAINT_RE.findall(last_user))
        if constraint_hits >= 2:
            q_score = min(1.0, q_score + 0.25)
        elif constraint_hits >= 1:
            q_score = min(1.0, q_score + 0.1)
        multi_part_hits = len(_MULTI_PART_RE.findall(last_user))
        if multi_part_hits >= 3:
            q_score = min(1.0, q_score + 0.15)

        factors["question_complexity"] = q_score

        # --- Weighted combination ---
        weights = {
            "token_count": 0.10,
            "task_markers": 0.35,
            "code_detection": 0.15,
            "conversation_depth": 0.05,
            "system_prompt": 0.05,
            "language_mix": 0.05,
            "question_complexity": 0.25,
        }

        score = sum(factors.get(k, 0.0) * w for k, w in weights.items())

        # Expert-marker floor: if the classifier found expert-level markers,
        # the query is genuinely hard regardless of other factors.
        if factors["task_markers"] >= 0.85:
            score = max(score, 0.72)
        elif factors["task_markers"] >= 0.67:
            score = max(score, 0.40)

        score = max(0.0, min(1.0, score))

        # Determine tier
        if score < 0.35:
            tier = "simple"
        elif score < 0.7:
            tier = "medium"
        else:
            tier = "complex"

        return QueryProfile(score=round(score, 4), tier=tier, factors=factors)
