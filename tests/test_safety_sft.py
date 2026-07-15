from __future__ import annotations

import unittest

from casafety.baselines.common import content_sha256
from casafety.baselines.safety_sft import (
    CompletionOnlyCollator,
    SafetyTuningRecord,
    example_sha256,
    prepare_records,
    prompt_from_instruction,
    stratified_train_validation_split,
    tokenize_completion_only,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        prefix = [10, 11, 12]
        if len(messages) == 1:
            return prefix
        response = messages[-1]["content"]
        return [*prefix, *[20 + index for index, _ in enumerate(response)], 2]


class SafetySftTests(unittest.TestCase):
    def test_prompt_normalization_matches_registered_alpaca_shape(self):
        self.assertEqual(
            prompt_from_instruction({"instruction": "Do it", "input": "with care"}),
            "Do it\n\nwith care",
        )
        self.assertEqual(
            prompt_from_instruction({"instruction": "Do it", "input": ""}),
            "Do it",
        )

    def test_prepare_records_labels_safety_and_excludes_evaluation(self):
        safety = {"instruction": "unsafe request", "input": "", "output": "I cannot help."}
        general = {"instruction": "explain rain", "input": "", "output": "Rain forms..."}
        excluded = {"instruction": "held out", "input": "", "output": "answer"}
        rows = [safety, general, excluded]
        records, statistics = prepare_records(
            rows,
            safety_example_hashes={example_sha256(safety)},
            excluded_prompt_hashes={content_sha256("held out")},
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(sum(record.is_safety for record in records), 1)
        self.assertEqual(statistics["evaluation_overlap_rows"], 1)
        self.assertEqual(statistics["selected_eval_intersection_count"], 0)

    def test_split_is_stratified_and_deterministic(self):
        records = [
            SafetyTuningRecord(str(i), "r", str(i), str(i), i < 10, i)
            for i in range(100)
        ]
        train_a, validation_a = stratified_train_validation_split(
            records, validation_size=20, seed=42
        )
        train_b, validation_b = stratified_train_validation_split(
            records, validation_size=20, seed=42
        )
        self.assertEqual([row.source_index for row in train_a], [row.source_index for row in train_b])
        self.assertEqual([row.source_index for row in validation_a], [row.source_index for row in validation_b])
        self.assertEqual(sum(row.is_safety for row in validation_a), 2)
        self.assertEqual(sum(row.is_safety for row in train_a), 8)

    def test_tokenization_masks_every_prompt_token(self):
        records = [SafetyTuningRecord("p", "abcd", "h", "e", True, 0)]
        rows, statistics = tokenize_completion_only(
            FakeTokenizer(), records, max_length=32
        )
        self.assertEqual(rows[0]["labels"][:3], [-100, -100, -100])
        self.assertTrue(all(value != -100 for value in rows[0]["labels"][3:]))
        self.assertEqual(statistics["safety_rows"], 1)
        self.assertTrue(statistics["assistant_only_loss"])


if __name__ == "__main__":
    unittest.main()
