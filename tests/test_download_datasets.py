import os
import tempfile
import unittest

from igc.ds.ds_utils import create_tar_gz, unpack_tar_gz


class TestDownloadableDataset(unittest.TestCase):

    def test_create_tar_gz(self):
        """Create tarball test
        :return:
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            tarball_loc = temp_dir
            test_input_dir = os.path.join(temp_dir, "test_dir_input")
            test_output_dir = os.path.join(temp_dir, "test_dir_output")

            files = [
                "file1.txt",
                "file2.txt",
                "file3.txt"
            ]

            output_tar_file = f"{tarball_loc}/test.tar.gz"
            os.makedirs(test_input_dir, exist_ok=True)

            for file_name in files:
                # Create test files
                with open(os.path.join(test_input_dir, file_name), "w") as f:
                    f.write("Test file 1")

            # create_tar_gz function
            tarball, tar_hash_file = create_tar_gz(test_input_dir, output_tar_file)
            self.assertTrue(os.path.exists(tarball))
            self.assertTrue(os.path.exists(tar_hash_file))

            unpack_tar_gz(f"{tarball_loc}/test.tar.gz", test_output_dir)
            input_files = os.listdir(test_input_dir)
            output_files = os.listdir(test_output_dir)
            self.assertListEqual(sorted(output_files), sorted(input_files))

    # def test_init(self):
    #     """
    #     :return:
    #     """
    #     dataset = TestDataset(dataset_root_dir="datasets")
    #     self.assertEqual(dataset._dataset_root_dir, "datasets")
    #     self.assertEqual(dataset.dataset_root_dir, dataset.dataset_root_dir / "post")
    #     self.assertEqual(dataset.post_process_dir(), dataset.dataset_root_dir / "raw")
    #     self.assertEqual(dataset.post_process_dir(), dataset.dataset_root_dir / "pre")
    #
    #     self.assertTrue(dataset.is_tarball())
    #     self.assertTrue(dataset.is_overwrite())
    #
    #     downloaded_files = dataset.downloaded_files()
    #     self.assertIsInstance(downloaded_files, list)
    #     self.assertGreater(len(downloaded_files), 0)
    #     dataset_files = dataset.dataset_files()


if __name__ == '__main__':
    unittest.main()
