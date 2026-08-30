import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_skill


class ValidateSkillTests(unittest.TestCase):
    def test_repository_skill_is_valid(self):
        self.assertEqual(validate_skill.validate_skill(str(ROOT)), (True, "Skill is valid!"))

    def test_missing_skill_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            valid, message = validate_skill.validate_skill(temp_dir)
        self.assertFalse(valid)
        self.assertIn("not found", message)

    def test_unexpected_frontmatter_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "SKILL.md").write_text(
                "---\nname: valid-name\ndescription: valid\nunknown: true\n---\nBody\n",
                encoding="utf-8",
            )
            valid, message = validate_skill.validate_skill(temp_dir)
        self.assertFalse(valid)
        self.assertIn("Unexpected", message)

    def test_todo_outside_code_fence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "SKILL.md").write_text(
                "---\nname: valid-name\ndescription: valid\n---\n[TODO: finish]\n",
                encoding="utf-8",
            )
            valid, message = validate_skill.validate_skill(temp_dir)
        self.assertFalse(valid)
        self.assertIn("TODO", message)

    def test_valid_skill_without_capability_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "SKILL.md").write_text(
                "---\nname: valid-name\ndescription: valid\n---\nBody\n",
                encoding="utf-8",
            )
            valid, message = validate_skill.validate_skill(temp_dir)
        self.assertFalse(valid)
        self.assertIn("Capability manifest invalid", message)

    def test_invalid_capability_manifest_reason_is_prefixed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "references").mkdir()
            (root / "SKILL.md").write_text(
                "---\nname: valid-name\ndescription: valid\n---\nBody\n",
                encoding="utf-8",
            )
            (root / "references" / "capabilities.json").write_text(
                json.dumps({"schemaVersion": 99}), encoding="utf-8"
            )
            valid, message = validate_skill.validate_skill(temp_dir)
        self.assertFalse(valid)
        self.assertTrue(message.startswith("Capability manifest invalid: "), message)


if __name__ == "__main__":
    unittest.main()
