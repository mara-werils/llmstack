"""Extra coverage for synthetic data augmentation — targets uncovered branches.

Complements tests/unit/learn/test_synthetic.py by exercising the property
accessors, the empty-strategy path, and each private strategy method directly
(including their early-return / no-op branches).
"""

from __future__ import annotations

from llmstack.learn.dataset import TrainingExample
from llmstack.learn.synthetic import AugmentationConfig, SyntheticAugmenter


def _ex(user: str, assistant: str = "ok", **metadata) -> TrainingExample:
    return TrainingExample(
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        metadata=metadata,
    )


def _single_message_ex(content: str = "hello") -> TrainingExample:
    return TrainingExample(messages=[{"role": "user", "content": content}])


# ---------------------------------------------------------------------------
# enabled_strategies / strategy_count properties (lines 57-66, 71)
# ---------------------------------------------------------------------------


def test_enabled_strategies_all_on():
    aug = SyntheticAugmenter(AugmentationConfig())
    assert aug.enabled_strategies == [
        "paraphrase",
        "word_swap",
        "format_variation",
        "instruction_variation",
    ]
    assert aug.strategy_count == 4


def test_enabled_strategies_all_off():
    cfg = AugmentationConfig(
        paraphrase=False,
        word_swap=False,
        format_variation=False,
        instruction_variation=False,
    )
    aug = SyntheticAugmenter(cfg)
    assert aug.enabled_strategies == []
    assert aug.strategy_count == 0


def test_enabled_strategies_subset():
    cfg = AugmentationConfig(
        paraphrase=False,
        word_swap=True,
        format_variation=False,
        instruction_variation=True,
    )
    aug = SyntheticAugmenter(cfg)
    assert aug.enabled_strategies == ["word_swap", "instruction_variation"]
    assert aug.strategy_count == 2


# ---------------------------------------------------------------------------
# augment: target_count is None default (line 86)
# ---------------------------------------------------------------------------


def test_augment_default_target_count_uses_max_factor():
    aug = SyntheticAugmenter(AugmentationConfig(seed=1, max_factor=2.0))
    examples = [_ex(f"How do I do thing {i}?", "Use the foo function") for i in range(5)]
    result = aug.augment(examples)  # target_count omitted -> None branch
    # capped at len * max_factor = 10
    assert len(result) <= 10
    assert len(result) >= len(examples)


# ---------------------------------------------------------------------------
# _create_variant: no strategies enabled -> None (line 122)
# ---------------------------------------------------------------------------


def test_create_variant_no_strategies_returns_none():
    cfg = AugmentationConfig(
        paraphrase=False,
        word_swap=False,
        format_variation=False,
        instruction_variation=False,
    )
    aug = SyntheticAugmenter(cfg)
    assert aug._create_variant(_ex("anything")) is None


# ---------------------------------------------------------------------------
# _paraphrase (lines 130, 156-158, 161)
# ---------------------------------------------------------------------------


def test_paraphrase_too_few_messages_returns_none():
    aug = SyntheticAugmenter(AugmentationConfig())
    assert aug._paraphrase(_single_message_ex()) is None


def test_paraphrase_pattern_match():
    aug = SyntheticAugmenter(AugmentationConfig(seed=42))
    variant = aug._paraphrase(_ex("How do I sort a list?", "Use sorted()."))
    assert variant is not None
    assert variant.messages[0]["content"] != "How do I sort a list?"
    assert variant.metadata["augmented"] is True
    assert variant.metadata["strategy"] == "paraphrase"


def test_paraphrase_no_pattern_uses_prefix_suffix():
    # Query matches none of the regex patterns -> else branch (156-158).
    # Seed chosen so the random prefix/suffix actually changes the query.
    for seed in range(50):
        aug = SyntheticAugmenter(AugmentationConfig(seed=seed))
        variant = aug._paraphrase(_ex("Sorting algorithms overview", "details"))
        if variant is not None:
            assert variant.messages[0]["content"] != "Sorting algorithms overview"
            assert variant.metadata["strategy"] == "paraphrase"
            break
    else:  # pragma: no cover
        raise AssertionError("expected at least one seed to produce a variant")


def test_paraphrase_no_change_returns_none():
    # No regex match, and find a seed where prefix+suffix are both empty -> None (161).
    for seed in range(50):
        aug = SyntheticAugmenter(AugmentationConfig(seed=seed))
        variant = aug._paraphrase(_ex("Sorting algorithms overview", "details"))
        if variant is None:
            break
    else:  # pragma: no cover
        raise AssertionError("expected at least one seed to produce no change")


# ---------------------------------------------------------------------------
# _word_swap (lines 174, 204)
# ---------------------------------------------------------------------------


def test_word_swap_too_few_messages_returns_none():
    aug = SyntheticAugmenter(AugmentationConfig())
    assert aug._word_swap(_single_message_ex("fix the error")) is None


def test_word_swap_replaces_known_word():
    aug = SyntheticAugmenter(AugmentationConfig(seed=42))
    variant = aug._word_swap(_ex("How do I fix this error?", "answer"))
    assert variant is not None
    assert variant.messages[0]["content"] != "How do I fix this error?"
    assert variant.metadata["strategy"] == "word_swap"


def test_word_swap_no_synonym_returns_none():
    aug = SyntheticAugmenter(AugmentationConfig(seed=42))
    assert aug._word_swap(_ex("Tell me about quantum physics", "answer")) is None


# ---------------------------------------------------------------------------
# _format_variation (lines 217, 224, 231, 234-236, 249)
# ---------------------------------------------------------------------------


def test_format_variation_too_few_messages_returns_none():
    aug = SyntheticAugmenter(AugmentationConfig())
    assert aug._format_variation(_single_message_ex()) is None


def test_format_variation_short_response_returns_none():
    aug = SyntheticAugmenter(AugmentationConfig())
    assert aug._format_variation(_ex("q", "short")) is None  # len < 50


def test_format_variation_wraps_code_block():
    # choice == 0 path: response has "def " and no fences -> wrap in ```python (231).
    response = "def add(a, b):\n    return a + b  # this is a sufficiently long response body"
    assert len(response) >= 50 and "def " in response and "```" not in response
    for seed in range(50):
        aug = SyntheticAugmenter(AugmentationConfig(seed=seed))
        variant = aug._format_variation(_ex("write add", response))
        if variant is not None and variant.messages[1]["content"].startswith("```python"):
            assert variant.metadata["strategy"] == "format_variation"
            break
    else:  # pragma: no cover
        raise AssertionError("expected a seed producing the code-block variant")


def test_format_variation_bullet_points():
    # choice == 1 path: 3+ sentences, no existing bullets -> bullet list (234-236).
    response = "First do this. Then do that. Finally verify the result. All done here."
    assert len(response) >= 50 and "\n- " not in response and ". " in response
    for seed in range(80):
        aug = SyntheticAugmenter(AugmentationConfig(seed=seed))
        variant = aug._format_variation(_ex("steps", response))
        if variant is not None and variant.messages[1]["content"].startswith("- "):
            assert variant.metadata["strategy"] == "format_variation"
            break
    else:  # pragma: no cover
        raise AssertionError("expected a seed producing the bullet variant")


def test_format_variation_prefix():
    # choice == 2 path: add an explanation prefix (returns a non-empty changed response).
    response = "This is a fairly long plain response with no code and no period chains here"
    for seed in range(80):
        aug = SyntheticAugmenter(AugmentationConfig(seed=seed))
        variant = aug._format_variation(_ex("explain", response))
        if variant is not None and variant.messages[1]["content"] != response:
            assert variant.metadata["strategy"] == "format_variation"
            break
    else:  # pragma: no cover
        raise AssertionError("expected a seed producing the prefix variant")


def test_format_variation_no_change_returns_none():
    # Long response but every branch is a no-op for some seed -> None (line ~247).
    # Plain prose, no "def ", fewer than 3 sentences, and choice==2 with empty prefix.
    response = "x" * 60  # no ". ", no "def ", no fences
    seen_none = False
    for seed in range(80):
        aug = SyntheticAugmenter(AugmentationConfig(seed=seed))
        variant = aug._format_variation(_ex("q", response))
        if variant is None:
            seen_none = True
            break
    assert seen_none


# ---------------------------------------------------------------------------
# _instruction_variation (lines 260, 274)
# ---------------------------------------------------------------------------


def test_instruction_variation_too_few_messages_returns_none():
    aug = SyntheticAugmenter(AugmentationConfig())
    assert aug._instruction_variation(_single_message_ex()) is None


def test_instruction_variation_adds_context():
    aug = SyntheticAugmenter(AugmentationConfig(seed=42))
    variant = aug._instruction_variation(_ex("sort a list quickly", "answer"))
    assert variant is not None
    assert variant.messages[0]["content"] != "sort a list quickly"
    assert variant.metadata["strategy"] == "instruction_variation"


def test_instruction_variation_no_valid_contexts_returns_none():
    # A query starting with "in " disqualifies context[0]; starting with "i" disqualifies
    # context[2]; context[1] is always valid... so to hit the None path (274) we need a
    # query where context[1] is the only one — that is still valid. The only way all three
    # become None is impossible normally, so verify behaviour: query starting with "in i..."
    # Actually context[1] (f"For a project...") is never None, so _instruction_variation
    # always returns a variant when messages >= 2. Confirm that invariant instead.
    aug = SyntheticAugmenter(AugmentationConfig(seed=7))
    variant = aug._instruction_variation(_ex("in python improve speed", "answer"))
    assert variant is not None
    assert variant.metadata["strategy"] == "instruction_variation"
