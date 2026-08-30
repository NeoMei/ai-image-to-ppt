import copy
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from capability_manifest import (
    CapabilityManifestError,
    load_capability_manifest,
    validate_capability_manifest,
)


EXPECTED_MANIFEST = {
    "schemaVersion": 1,
    "skill": "ai-image-to-ppt",
    "contracts": {
        "generationResult": 1,
        "serialStickyRouterReport": 1,
        "hostImageImport": 1,
        "editableInput": 1,
    },
    "routingOrder": [
        {"provider": "openai", "channel": "host", "modelSelection": "host-owned"},
        {"provider": "openai", "channel": "api", "defaultModel": "gpt-image-2"},
        {"provider": "gemini", "channel": "host", "modelSelection": "host-owned"},
        {
            "provider": "gemini",
            "channel": "api",
            "defaultModel": "gemini-3.1-flash-image",
        },
        {"provider": "doubao", "channel": "host", "modelSelection": "host-owned"},
        {
            "provider": "doubao",
            "channel": "api",
            "defaultModel": "doubao-seedream-5-0-260128",
        },
    ],
    "outputs": {
        "normalizedSlide": {"format": "image", "width": 1920, "height": 1080},
        "editableInput": {"format": "png", "width": 1280, "height": 720},
    },
    "scripts": {
        "generationResult": "scripts/generation_result.py",
        "hostRoutingPolicy": "scripts/host_routing_policy.py",
        "importHostImage": "scripts/import_host_image.py",
        "prepareEditableInput": "scripts/prepare_editable_input.py",
        "apiGenerator": "scripts/gen_slide.py",
        "normalizedExport": "scripts/export_images.py",
    },
}


def write_skill_fixture(root: Path) -> None:
    (root / "references").mkdir(parents=True)
    for relative_path in EXPECTED_MANIFEST["scripts"].values():
        script = root / relative_path
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# fixture\n", encoding="utf-8")
    write_manifest(root, EXPECTED_MANIFEST)


def write_manifest(root: Path, manifest: dict) -> None:
    (root / "references").mkdir(parents=True, exist_ok=True)
    (root / "references" / "capabilities.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class CapabilityManifestTests(unittest.TestCase):
    def test_published_capability_manifest_is_valid(self):
        valid, message = validate_capability_manifest(REPOSITORY_ROOT)
        self.assertTrue(valid, message)

    def test_loader_returns_the_published_contract(self):
        self.assertEqual(load_capability_manifest(REPOSITORY_ROOT), EXPECTED_MANIFEST)

    def test_manifest_matches_host_first_sticky_order(self):
        manifest = load_capability_manifest(REPOSITORY_ROOT)
        self.assertEqual(
            [f'{item["provider"]}:{item["channel"]}' for item in manifest["routingOrder"]],
            [
                "openai:host",
                "openai:api",
                "gemini:host",
                "gemini:api",
                "doubao:host",
                "doubao:api",
            ],
        )

    def test_api_models_match_runtime_defaults(self):
        import gen_slide_doubao
        import gen_slide_gemini
        import gen_slide_openai

        manifest = load_capability_manifest(REPOSITORY_ROOT)
        api_models = {
            item["provider"]: item["defaultModel"]
            for item in manifest["routingOrder"]
            if item["channel"] == "api"
        }
        self.assertEqual(
            api_models,
            {
                "openai": gen_slide_openai.DEFAULT_MODEL,
                "gemini": gen_slide_gemini.DEFAULT_MODEL,
                "doubao": gen_slide_doubao.MODEL,
            },
        )

    def test_missing_and_malformed_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("manifest", message.lower())

            (root / "references").mkdir()
            (root / "references" / "capabilities.json").write_text(
                "{not-json", encoding="utf-8"
            )
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("JSON", message)

    def test_unknown_and_missing_keys_are_rejected_at_every_object_level(self):
        mutations = {
            "top-level unknown": lambda m: m.__setitem__("unknown", True),
            "top-level missing": lambda m: m.pop("skill"),
            "contracts unknown": lambda m: m["contracts"].__setitem__("unknown", 1),
            "contracts missing": lambda m: m["contracts"].pop("editableInput"),
            "route unknown": lambda m: m["routingOrder"][0].__setitem__("model", "pinned"),
            "route missing": lambda m: m["routingOrder"][0].pop("modelSelection"),
            "outputs unknown": lambda m: m["outputs"].__setitem__("unknown", {}),
            "outputs missing": lambda m: m["outputs"].pop("editableInput"),
            "output unknown": lambda m: m["outputs"]["editableInput"].__setitem__(
                "unknown", 1
            ),
            "output missing": lambda m: m["outputs"]["editableInput"].pop("height"),
            "scripts unknown": lambda m: m["scripts"].__setitem__("unknown", "scripts/x.py"),
            "scripts missing": lambda m: m["scripts"].pop("apiGenerator"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    mutate(manifest)
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("keys", message)

    def test_wrong_scalar_types_are_rejected_without_bool_as_integer(self):
        mutations = {
            "schema bool": lambda m: m.__setitem__("schemaVersion", True),
            "skill list": lambda m: m.__setitem__("skill", ["ai-image-to-ppt"]),
            "contract string": lambda m: m["contracts"].__setitem__(
                "generationResult", "1"
            ),
            "contract bool": lambda m: m["contracts"].__setitem__(
                "generationResult", True
            ),
            "routing object": lambda m: m.__setitem__("routingOrder", {}),
            "route provider integer": lambda m: m["routingOrder"][0].__setitem__(
                "provider", 1
            ),
            "route channel list": lambda m: m["routingOrder"][0].__setitem__(
                "channel", ["host"]
            ),
            "model integer": lambda m: m["routingOrder"][1].__setitem__(
                "defaultModel", 2
            ),
            "outputs list": lambda m: m.__setitem__("outputs", []),
            "format integer": lambda m: m["outputs"]["editableInput"].__setitem__(
                "format", 1
            ),
            "width string": lambda m: m["outputs"]["editableInput"].__setitem__(
                "width", "1280"
            ),
            "height bool": lambda m: m["outputs"]["editableInput"].__setitem__(
                "height", True
            ),
            "scripts list": lambda m: m.__setitem__("scripts", []),
            "script integer": lambda m: m["scripts"].__setitem__("apiGenerator", 1),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    mutate(manifest)
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("type", message)

    def test_unsupported_schema_and_contract_versions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            for label, mutate in (
                ("schema", lambda m: m.__setitem__("schemaVersion", 2)),
                (
                    "contract",
                    lambda m: m["contracts"].__setitem__("generationResult", 2),
                ),
            ):
                with self.subTest(label=label):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    mutate(manifest)
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("unsupported", message.lower())

    def test_duplicate_and_wrong_order_routes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            for label, mutate in (
                (
                    "duplicate",
                    lambda m: m["routingOrder"].__setitem__(
                        1, copy.deepcopy(m["routingOrder"][0])
                    ),
                ),
                (
                    "wrong order",
                    lambda m: m["routingOrder"].__setitem__(
                        slice(0, 2), [m["routingOrder"][1], m["routingOrder"][0]]
                    ),
                ),
                ("missing route", lambda m: m["routingOrder"].pop()),
            ):
                with self.subTest(label=label):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    mutate(manifest)
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("routing order", message.lower())

    def test_host_routes_are_host_owned_and_api_defaults_are_fixed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            mutations = {
                "host model pinned": lambda m: m["routingOrder"][0].__setitem__(
                    "modelSelection", "gpt-image-2"
                ),
                "openai api drift": lambda m: m["routingOrder"][1].__setitem__(
                    "defaultModel", "gpt-image-1"
                ),
                "gemini api drift": lambda m: m["routingOrder"][3].__setitem__(
                    "defaultModel", "gemini-other"
                ),
                "doubao api drift": lambda m: m["routingOrder"][5].__setitem__(
                    "defaultModel", "doubao-other"
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    mutate(manifest)
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("model", message.lower())

    def test_output_formats_and_dimensions_are_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            mutations = {
                "normalized format": lambda m: m["outputs"]["normalizedSlide"].__setitem__(
                    "format", "png"
                ),
                "normalized width": lambda m: m["outputs"]["normalizedSlide"].__setitem__(
                    "width", 1280
                ),
                "normalized height": lambda m: m["outputs"]["normalizedSlide"].__setitem__(
                    "height", 720
                ),
                "editable format": lambda m: m["outputs"]["editableInput"].__setitem__(
                    "format", "image"
                ),
                "editable width": lambda m: m["outputs"]["editableInput"].__setitem__(
                    "width", 1920
                ),
                "editable height": lambda m: m["outputs"]["editableInput"].__setitem__(
                    "height", 1080
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    mutate(manifest)
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("output", message.lower())

    def test_unsafe_absolute_parent_and_control_character_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            for unsafe_path in (
                "/tmp/outside.py",
                "../outside.py",
                "scripts/../generation_result.py",
                "scripts\\generation_result.py",
                "scripts/generation_result.py\nignored",
                "scripts/\x00generation_result.py",
            ):
                with self.subTest(path=repr(unsafe_path)):
                    manifest = copy.deepcopy(EXPECTED_MANIFEST)
                    manifest["scripts"]["generationResult"] = unsafe_path
                    write_manifest(root, manifest)
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("safe relative script", message)

    def test_c1_unicode_control_character_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            c1_path = "scripts/\u0085generation_result.py"
            (root / c1_path).write_text("# unsafe name\n", encoding="utf-8")
            manifest = copy.deepcopy(EXPECTED_MANIFEST)
            manifest["scripts"]["generationResult"] = c1_path
            write_manifest(root, manifest)

            valid, message = validate_capability_manifest(root)

            self.assertFalse(valid, message)
            self.assertIn("safe relative script", message)

    def test_duplicate_json_members_are_rejected_at_top_and_nested_levels(self):
        valid_json = json.dumps(EXPECTED_MANIFEST)
        duplicates = {
            "top level": valid_json.replace(
                '"schemaVersion": 1',
                '"schemaVersion": 99, "schemaVersion": 1',
                1,
            ),
            "nested contract": valid_json.replace(
                '"generationResult": 1',
                '"generationResult": 2, "generationResult": 1',
                1,
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            manifest_path = root / "references" / "capabilities.json"
            for label, manifest_text in duplicates.items():
                with self.subTest(label=label):
                    manifest_path.write_text(manifest_text, encoding="utf-8")
                    valid, message = validate_capability_manifest(root)
                    self.assertFalse(valid, message)
                    self.assertIn("duplicate JSON member", message)

    def test_json_parser_value_error_is_mapped_to_capability_manifest_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            with mock.patch(
                "capability_manifest.json.loads",
                side_effect=ValueError("integer string conversion limit exceeded"),
            ):
                with self.assertRaisesRegex(
                    CapabilityManifestError, "not valid JSON"
                ):
                    load_capability_manifest(root)

    def test_escaped_unicode_surrogate_path_is_rejected_before_filesystem_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            manifest_text = json.dumps(EXPECTED_MANIFEST).replace(
                "scripts/generation_result.py",
                r"scripts/\ud800bad.py",
                1,
            )
            (root / "references" / "capabilities.json").write_text(
                manifest_text, encoding="utf-8"
            )

            valid, message = validate_capability_manifest(root)

            self.assertFalse(valid, message)
            self.assertIn("safe relative script", message)

    def test_missing_non_regular_and_symlink_scripts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            target = root / EXPECTED_MANIFEST["scripts"]["generationResult"]

            target.unlink()
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("regular file", message)

            target.mkdir()
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("regular file", message)

            target.rmdir()
            real_script = root / "scripts" / "real.py"
            real_script.write_text("# real\n", encoding="utf-8")
            target.symlink_to(real_script)
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("symlink", message.lower())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO support")
    def test_fifo_script_is_rejected_as_non_regular(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_skill_fixture(root)
            target = root / EXPECTED_MANIFEST["scripts"]["generationResult"]
            target.unlink()
            os.mkfifo(target, stat.S_IRUSR | stat.S_IWUSR)
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("regular file", message)

    def test_script_resolving_outside_root_and_symlink_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "skill"
            write_skill_fixture(root)
            outside = base / "outside.py"
            outside.write_text("# outside\n", encoding="utf-8")

            declared = root / "scripts" / "generation_result.py"
            declared.unlink()
            declared.symlink_to(outside)
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("Skill root", message)

            declared.unlink()
            real_scripts = root / "real-scripts"
            real_scripts.mkdir()
            (real_scripts / "generation_result.py").write_text("# real\n", encoding="utf-8")
            (root / "scripts").rename(root / "scripts-original")
            (root / "scripts").symlink_to(real_scripts, target_is_directory=True)
            valid, message = validate_capability_manifest(root)
            self.assertFalse(valid)
            self.assertIn("symlink", message.lower())


if __name__ == "__main__":
    unittest.main()
