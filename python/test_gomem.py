"""Pure-Python checks that need no built libgomem (run: python3 -m unittest discover python)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gomem  # noqa: E402


class ConfigDefaultTest(unittest.TestCase):
    """Unset tuning arguments are omitted so Go's DefaultConfig() applies."""

    def _sent_config(self, **kwargs):
        sent = []
        lib = type("L", (), {"GomemNew": object()})()
        orig = gomem._invoke
        gomem._invoke = lambda lib, fn, payload: (
            sent.append(__import__("json").loads(payload)) or {"handle": 1}
        )
        try:
            gomem.Memory("x.db", lib=lib, **kwargs)
        finally:
            gomem._invoke = orig
        return sent[0]

    def test_unset_settings_are_omitted(self):
        self.assertEqual(
            self._sent_config(),
            {"path": "x.db", "provider": "offline", "log_level": "warn"},
        )

    def test_explicit_zero_is_sent(self):
        cfg = self._sent_config(dim=384, alpha_dense=0.0, embed_length_threshold=0)
        self.assertEqual(cfg["dim"], 384)
        self.assertEqual(cfg["alpha_dense"], 0.0)
        self.assertEqual(cfg["embed_length_threshold"], 0)

    def test_send_dimensions_false_is_sent(self):
        cfg = self._sent_config(send_dimensions=False)
        self.assertIs(cfg["send_dimensions"], False)


class TierDefaultTest(unittest.TestCase):
    """Decision D1: tier is optional; omitted on the wire unless the caller sets it."""

    def test_unspecified_tier_is_omitted(self):
        self.assertNotIn("tier", gomem._item_doc(title="t"))

    def test_explicit_zero_is_sent_as_zero(self):
        doc = gomem._item_doc(title="t", tier=0)
        self.assertIn("tier", doc)
        self.assertEqual(doc["tier"], 0)
        self.assertEqual(gomem._item_doc(title="t", tier=gomem.TIER_COLD)["tier"], 0)

    def test_explicit_tier_is_preserved(self):
        self.assertEqual(gomem._item_doc(title="t", tier=1)["tier"], 1)

    def test_signatures_default_to_none(self):
        import inspect

        for name in ("remember", "aremember"):
            p = inspect.signature(getattr(gomem.Memory, name)).parameters["tier"]
            self.assertIsNone(p.default, name)

    def test_remember_many_omits_unspecified_tier(self):
        sent = []

        class Fake(gomem.Memory):
            def __init__(self):  # no library needed
                self._handle = 1
                self._lib = type("L", (), {"GomemRememberMany": object()})()

        orig = gomem._invoke
        gomem._invoke = lambda lib, fn, handle, payload: (
            sent.append(__import__("json").loads(payload)) or {"ids": ["a", "b"]}
        )
        try:
            Fake().remember_many([{"title": "x"}, {"title": "y", "tier": 0}])
        finally:
            gomem._invoke = orig
        self.assertEqual([d.get("tier") for d in sent[0]], [None, 0])


class CorruptOpenTest(unittest.TestCase):
    """Go's New returns a usable store with ErrCorrupt; the client must keep it."""

    def test_handle_with_error_is_kept(self):
        orig = gomem._invoke
        gomem._invoke = lambda *a, **k: {
            "handle": 7,
            "error": "corrupt: backed up and recreated",
        }
        try:
            mem = gomem.Memory("x.db", lib=type("L", (), {"GomemNew": object()})())
            self.assertEqual(mem._handle, 7)
            self.assertTrue(mem.recovered_from_corrupt)
        finally:
            gomem._invoke = orig

    def test_error_without_handle_raises(self):
        orig = gomem._invoke
        gomem._invoke = lambda *a, **k: {"error": "invalid config"}
        try:
            with self.assertRaises(gomem.GomemError):
                gomem.Memory("x.db", lib=type("L", (), {"GomemNew": object()})())
        finally:
            gomem._invoke = orig

    def test_invoke_error_without_handle_raises(self):
        class Lib:
            def GomemFreeString(self, _ptr):
                pass

        orig_string_at = gomem.ctypes.string_at
        gomem.ctypes.string_at = lambda _ptr: b'{"error":"boom"}'
        try:
            with self.assertRaises(gomem.GomemError) as ctx:
                gomem._invoke(Lib(), lambda: 1)
            self.assertIn("boom", str(ctx.exception))
        finally:
            gomem.ctypes.string_at = orig_string_at

    def test_invoke_keeps_doc_when_handle_and_error(self):
        class Lib:
            def GomemFreeString(self, _ptr):
                pass

        orig_string_at = gomem.ctypes.string_at
        gomem.ctypes.string_at = lambda _ptr: b'{"handle":3,"error":"corrupt"}'
        try:
            doc = gomem._invoke(Lib(), lambda: 1)
            self.assertEqual(doc["handle"], 3)
            self.assertEqual(doc["error"], "corrupt")
        finally:
            gomem.ctypes.string_at = orig_string_at


if __name__ == "__main__":
    unittest.main()
