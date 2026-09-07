#!/usr/bin/env python3
"""Run the ai-image-to-ppt surface that is supported on Windows."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))

MODULES = (
    "test_capability_manifest",
    "test_cleanup_output_locks",
    "test_documentation",
    "test_export_directory_sync",
    "test_export_images",
    "test_gen_slide",
    "test_generation_result",
    "test_host_routing_e2e",
    "test_host_routing_policy",
    "test_import_host_image.HostImageImportTests.test_local_file_is_validated_and_published_inside_workspace",
    "test_output_lock_registry",
    "test_prepare_editable_input",
    "test_round10_commit_marker_priority",
    "test_round7_recovery_identity",
    "test_round8_recovery_protocol",
    "test_round9_loaded_byte_short_circuit",
    "test_round9_recovery_ownership",
    "test_validate_skill",
    "test_vision_check_gemini",
    "test_windows_image_output",
)


def main() -> int:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
