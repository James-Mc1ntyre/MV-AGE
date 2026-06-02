import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class TestRepoLayout(unittest.TestCase):
    def test_repo_files_exist(self):
        for filename in [
            "LICENSE",
            "README.md",
            "THIRD_PARTY_NOTICES.md",
            "requirements.txt",
            ".github/workflows/tests.yml",
        ]:
            self.assertTrue((PACKAGE_ROOT / filename).exists(), filename)

    def test_pypi_packaging_files_are_not_required(self):
        self.assertFalse((PACKAGE_ROOT / "pyproject.toml").exists())
        self.assertFalse((PACKAGE_ROOT / "MANIFEST.in").exists())

    def test_requirements_uses_tensorflow_platform_markers(self):
        requirements = (PACKAGE_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn('tensorflow>=2.15,<2.16; sys_platform != "win32"', requirements)
        self.assertIn('tensorflow-intel>=2.15,<2.16; sys_platform == "win32"', requirements)

    def test_runtime_tree_excludes_large_magcn_docs_and_benchmark_data(self):
        self.assertFalse((PACKAGE_ROOT / "mv_age/vendor/MAGCN/docs").exists())
        self.assertFalse((PACKAGE_ROOT / "mv_age/vendor/MAGCN/tf_version/data").exists())


if __name__ == "__main__":
    unittest.main()
