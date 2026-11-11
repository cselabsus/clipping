import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).resolve().parents[1] / "Scripts/prepare_model.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ModelPreparationTests(unittest.TestCase):
    def test_verification_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            path.write_bytes(b"valid")
            expected = {"size": 5, "sha256": hashlib.sha256(b"valid").hexdigest()}
            self.assertTrue(prepare.matches(path, expected))
            path.write_bytes(b"wrong")
            self.assertFalse(prepare.matches(path, expected))

    def test_bad_download_cannot_replace_verified_file_or_mark_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            original = model / "weights"
            original.write_bytes(b"old")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"repository": "test/model", "revision": "pinned", "files": [
                {"name": "weights", "size": 5, "sha256": hashlib.sha256(b"valid").hexdigest()}
            ]}))

            def bad_download(arguments, check):
                Path(arguments[arguments.index("--output") + 1]).write_bytes(b"wrong")

            with patch.object(prepare, "MODEL", model), patch.object(prepare, "MANIFEST", manifest), \
                 patch.object(prepare.subprocess, "run", side_effect=bad_download):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    prepare.prepare()
            self.assertEqual(original.read_bytes(), b"old")
            self.assertFalse((model / "verified.json").exists())
            self.assertEqual(list(root.glob(".download-*")), [])


if __name__ == "__main__":
    unittest.main()
