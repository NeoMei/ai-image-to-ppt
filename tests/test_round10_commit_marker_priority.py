import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


class CommitMarkerPriorityTests(unittest.TestCase):
    def test_valid_commit_marker_recovers_when_rollback_journal_is_truncated(self):
        program = textwrap.dedent(
            """
            import os
            import sys

            sys.path.insert(0, sys.argv[1])
            import export_images

            real_link = export_images.os.link

            def crash_after_commit_marker(source_path, target_path, *args, **kwargs):
                result = real_link(source_path, target_path, *args, **kwargs)
                if str(target_path).endswith(export_images.COMMIT_MARKER_SUFFIX):
                    os._exit(77)
                return result

            export_images.os.link = crash_after_commit_marker
            export_images.export_deck([sys.argv[2]], sys.argv[3])
            raise SystemExit(3)
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "slide.png"
            prefix = root / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            Image.new("RGB", (160, 90), "navy").save(source)

            crashed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(ROOT / "scripts"),
                    str(source),
                    str(prefix),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(crashed.returncode, 77, crashed.stderr)
            journal = export_images._journal_path(prefix)
            marker = export_images._commit_marker_path(journal)
            preparation = export_images._preparation_path(journal)
            self.assertTrue(journal.is_file())
            self.assertTrue(marker.is_file())
            self.assertTrue(preparation.is_dir())

            # Preserve the journal inode and its hard-linked preparation writer,
            # but make the rollback JSON unreadable. The durable commit marker is
            # authoritative and still contains the complete recovery state.
            journal.write_bytes(b"{")
            pptx.unlink()

            with redirect_stderr(io.StringIO()):
                result = export_images.export_deck(
                    [str(root / "missing.png")], str(prefix)
                )

            self.assertFalse(result)
            self.assertTrue(pdf.is_file())
            self.assertTrue(pptx.is_file())
            self.assertFalse(journal.exists())
            self.assertFalse(marker.exists())
            self.assertFalse(preparation.exists())


if __name__ == "__main__":
    unittest.main()
