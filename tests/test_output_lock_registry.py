import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import output_lock


class OutputLockRegistryTests(unittest.TestCase):
    def test_released_unique_locks_do_not_accumulate_in_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = len(output_lock._THREAD_LOCKS)
            for index in range(100):
                target = Path(temp_dir) / f"slide-{index}.jpg"
                with output_lock.output_lock(target):
                    pass
            self.assertEqual(len(output_lock._THREAD_LOCKS), baseline)

    def test_failed_contender_does_not_remove_live_registry_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            identity = output_lock._identity(target, "output")
            result = []

            def contend():
                try:
                    with output_lock.output_lock(target):
                        result.append("acquired")
                except output_lock.OutputLockBusy:
                    result.append("busy")

            with output_lock.output_lock(target):
                thread = threading.Thread(target=contend)
                thread.start()
                thread.join()
                self.assertEqual(result, ["busy"])
                self.assertIn(identity, output_lock._THREAD_LOCKS)

            self.assertNotIn(identity, output_lock._THREAD_LOCKS)


if __name__ == "__main__":
    unittest.main()
