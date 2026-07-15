import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from casafety.baselines.common import (
    MODEL_ID,
    RunState,
    Stage,
    parse_gpu_map,
    verify_wanda_checkpoint,
)


class BaselineCommonTest(unittest.TestCase):
    def test_wanda_checkpoint_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = checkpoint / "config.json"
            config.write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model": MODEL_ID,
                        "pruner": "wanda",
                        "requested_sparsity": 0.5,
                        "sparsity": {"realized_zero_fraction": 0.5},
                        "checkpoint_files_sha256": {"config.json": config_hash},
                    }
                ),
                encoding="utf-8",
            )
            identity = verify_wanda_checkpoint(checkpoint, manifest)
            self.assertEqual(identity["requested_sparsity"], 0.5)
            self.assertEqual(identity["checkpoint_files_sha256"]["config.json"], config_hash)

    def test_resume_state_requires_declared_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            stage = Stage("fit", ("python", "worker.py"), (output,))
            state = RunState(root / "state.json", method="sft")
            state.mark_running(stage)
            output.write_text("{}", encoding="utf-8")
            state.mark_completed(stage, seconds=12.0, gpu_count=1)
            restored = RunState(root / "state.json", method="sft")
            self.assertTrue(restored.stage_complete(stage))
            output.unlink()
            with self.assertRaises(FileNotFoundError):
                restored.stage_complete(stage)

    def test_gpu_mapping_is_explicit_and_validated(self) -> None:
        mapping = parse_gpu_map(("sft", "dpo"), "0", ("dpo=1,2",))
        self.assertEqual(mapping, {"sft": "0", "dpo": "1,2"})
        with self.assertRaises(ValueError):
            parse_gpu_map(("sft",), "0", ("dpo=1",))


if __name__ == "__main__":
    unittest.main()
