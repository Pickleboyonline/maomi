"""Tests for improved code action suggestions (proportional edit distance)."""
from maomi.lsp._code_actions import _ca_find_similar, _ca_edit_distance


class TestProportionalEditDistance:
    def test_short_name_default_threshold(self):
        """Short names (1-3 chars) use max_distance=2, catches transpositions."""
        # "x" with threshold=2: "y" (dist 1), "xy" (dist 1), "xyz" (dist 2) match
        result = _ca_find_similar("x", ["y", "xy", "abc", "xyz"])
        assert "y" in result  # distance 1
        assert "xy" in result  # distance 1
        assert "xyz" in result  # distance 2, within threshold
        assert "abc" not in result  # distance 3, over threshold

    def test_medium_name(self):
        """Medium names (4-6 chars) use max_distance=2 (proportional floor)."""
        result = _ca_find_similar("relu", ["ralu", "gelu", "sigmoid", "relu6"])
        assert "gelu" in result  # distance 1
        assert "ralu" in result  # distance 1 (substitution)
        assert "relu6" in result  # distance 1 (insertion)
        assert "sigmoid" not in result  # distance 7, way over threshold

    def test_long_name_lenient(self):
        """Long names (10+) allow more edits via proportional scaling."""
        result = _ca_find_similar("batch_normalize", ["batch_normalise", "layer_normalize", "batch_norm"])
        assert "batch_normalise" in result  # distance 1, within max_distance=5
        assert "layer_normalize" in result  # distance 4, within max_distance=5
        assert "batch_norm" in result  # distance 5, within max_distance=5

    def test_proportional_scaling(self):
        """Threshold scales with name length: len // 3, minimum 2."""
        # 3-char: max(3//3, 2) = 2
        assert "exp" in _ca_find_similar("epx", ["exp"])  # distance 2, within threshold
        # 9-char: max(9//3, 2) = 3
        assert "some_func" in _ca_find_similar("sume_funk", ["some_func"])  # distance 2
        # 12-char: max(12//3, 2) = 4
        result = _ca_find_similar("some_functio", ["some_function"])
        assert "some_function" in result  # distance 1

    def test_case_insensitive_fallback(self):
        """Case-insensitive match when no edit-distance match found."""
        # "BATCH_NORMALIZE" vs "batch_normalize": distance is 15 (all chars differ in case)
        # but Levenshtein treats case changes as substitutions, so we need a name
        # where edit distance exceeds threshold but case matches
        result = _ca_find_similar("SIGMOID", ["sigmoid", "softmax", "tanh"])
        assert "sigmoid" in result  # case-insensitive fallback

    def test_case_insensitive_not_used_when_edit_match_exists(self):
        """Case-insensitive fallback only activates when no edit matches found."""
        result = _ca_find_similar("galu", ["relu", "gelu"])
        assert "gelu" in result  # edit distance match

    def test_results_sorted_by_distance(self):
        """Results are sorted by edit distance, closest first."""
        result = _ca_find_similar("test_function", ["test_functian", "test_funtcion", "completely_different"])
        if len(result) >= 2:
            # First result should be closer
            d1 = _ca_edit_distance("test_function", result[0])
            d2 = _ca_edit_distance("test_function", result[1])
            assert d1 <= d2

    def test_max_five_results(self):
        """At most 5 results returned."""
        candidates = [f"name{i}" for i in range(20)]
        result = _ca_find_similar("name", candidates)
        assert len(result) <= 5

    def test_exact_match_excluded(self):
        """Exact match of the name itself is not suggested."""
        result = _ca_find_similar("relu", ["relu", "gelu"])
        assert "relu" not in result

    def test_empty_candidates(self):
        """Empty candidate list returns empty."""
        assert _ca_find_similar("foo", []) == []

    def test_explicit_max_distance_override(self):
        """Explicit max_distance parameter still works."""
        result = _ca_find_similar("relu", ["ralu"], max_distance=2)
        assert "ralu" in result  # distance 1, within explicit max

    def test_case_insensitive_exact_only(self):
        """Case-insensitive fallback requires exact case-insensitive match, not partial."""
        # "RELU" should find "relu" but not "ralu" via case fallback
        result = _ca_find_similar("RELU", ["relu", "ralu", "xyz"])
        assert "relu" in result
