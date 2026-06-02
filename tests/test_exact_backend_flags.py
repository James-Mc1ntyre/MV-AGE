import unittest

from mv_age.ExactBackend import _ensure_flags_parsed


class _FakeFlags:
    def __init__(self):
        self.calls = []
        self._parsed = False

    def is_parsed(self):
        return self._parsed

    def __call__(self, argv, known_only=False):
        self.calls.append((list(argv), known_only))
        self._parsed = True


class TestExactBackendFlags(unittest.TestCase):
    def test_ensure_flags_parsed_uses_safe_single_argv(self):
        flags = _FakeFlags()
        _ensure_flags_parsed(flags)
        self.assertEqual(flags.calls, [(["mv_age"], True)])

    def test_ensure_flags_parsed_skips_when_already_parsed(self):
        flags = _FakeFlags()
        flags._parsed = True
        _ensure_flags_parsed(flags)
        self.assertEqual(flags.calls, [])


if __name__ == "__main__":
    unittest.main()
