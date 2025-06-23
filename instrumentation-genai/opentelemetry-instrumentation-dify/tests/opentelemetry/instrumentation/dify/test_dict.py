import unittest
from unittest.mock import Mock
from opentelemetry.instrumentation.dify.config import is_version_supported


class TestDict(unittest.TestCase):

    def test_no_key(self):
        d = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
        d = dict(d)
        v = d.get("key6")
        self.assertIsNone(v)

    def test_key_value(self):
        d = {
            "key1": "value1",
            "key2": "value2",
            "key3": {
                "key1": "value1"
            },
        }
        d = dict(d)
        if v := d.get("key1"):
            self.assertEqual(v, "value1")

        if v := d.get("key6"):
            pass
        else:
            self.assertIsNone(v)