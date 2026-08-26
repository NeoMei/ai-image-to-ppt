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


if __name__ == "__main__":
    unittest.main()
