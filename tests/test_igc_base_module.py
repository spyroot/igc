import unittest
from igc.modules.base.igc_base_module import IgcModule


class TestRestApiEnv(unittest.TestCase):

    def test_read_spec(self):
        """
        Test the extract_action_method method of RestApiEnv.
        """
        models_data = IgcModule.read_model_specs()
        self.assertIn("mirrors", models_data)
        mirrors = models_data["mirrors"]
        for mirror_entries in mirrors.values():
            for entry in mirror_entries:
                self.assertIsInstance(entry, dict)
                if "files" in entry and "local_file" in entry:
                    self.assertIn("files", entry)
                    self.assertIn("local_file", entry)
                    files = entry["files"]
                    local_files = entry["local_file"]
                    self.assertIsInstance(files, list)
                    self.assertIsInstance(local_files, list)
                    self.assertEqual(len(files), len(local_files))

    def test_download(self):
        """
        Test the extract_action_method method of RestApiEnv.
        """
        IgcModule.download()
