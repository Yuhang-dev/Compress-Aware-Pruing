import unittest

from casafety.baselines.common import content_sha256, normalize_content
from casafety.baselines.data import dpo_records, select_safe_pairs, sft_records


class BaselineDataTest(unittest.TestCase):
    def test_normalized_hash_collapses_case_unicode_and_whitespace(self) -> None:
        left = "  HELLO\tworld  "
        right = "hello world"
        self.assertEqual(normalize_content(left), right)
        self.assertEqual(content_sha256(left), content_sha256(right))

    def test_exactly_one_safe_filter_dedup_and_eval_exclusion(self) -> None:
        excluded = content_sha256("held out")
        rows = [
            {
                "prompt": "keep one",
                "response_0": "safe one",
                "response_1": "unsafe one",
                "is_response_0_safe": True,
                "is_response_1_safe": False,
            },
            {
                "prompt": "KEEP   ONE",
                "response_0": "duplicate",
                "response_1": "duplicate unsafe",
                "is_response_0_safe": True,
                "is_response_1_safe": False,
            },
            {
                "prompt": "held out",
                "response_0": "safe",
                "response_1": "unsafe",
                "is_response_0_safe": True,
                "is_response_1_safe": False,
            },
            {
                "prompt": "both safe",
                "response_0": "a",
                "response_1": "b",
                "is_response_0_safe": True,
                "is_response_1_safe": True,
            },
            {
                "prompt": "keep two",
                "response_0": "unsafe two",
                "response_1": "safe two",
                "is_response_0_safe": False,
                "is_response_1_safe": True,
            },
        ]
        selection = select_safe_pairs(
            rows,
            excluded_prompt_hashes={excluded},
            limit=2,
            seed=7,
        )
        self.assertEqual(len(selection.pairs), 2)
        self.assertEqual(selection.statistics["evaluation_overlap_rows"], 1)
        self.assertEqual(selection.statistics["duplicate_prompt_rows"], 1)
        self.assertEqual(selection.statistics["invalid_or_same_safety_rows"], 1)
        self.assertEqual(selection.statistics["selected_eval_intersection_count"], 0)
        self.assertNotIn(excluded, {pair.prompt_sha256 for pair in selection.pairs})

    def test_sft_and_dpo_orient_safe_response_correctly(self) -> None:
        rows = [
            {
                "prompt": "p",
                "response_0": "unsafe",
                "response_1": "safe",
                "is_response_0_safe": False,
                "is_response_1_safe": True,
            }
        ]
        pair = select_safe_pairs(
            rows, excluded_prompt_hashes=set(), limit=1, seed=0
        ).pairs
        self.assertEqual(sft_records(pair)[0]["messages"][1]["content"], "safe")
        dpo = dpo_records(pair)[0]
        self.assertEqual(dpo["chosen"][0]["content"], "safe")
        self.assertEqual(dpo["rejected"][0]["content"], "unsafe")


if __name__ == "__main__":
    unittest.main()
