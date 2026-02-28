# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import unittest
from datetime import datetime
from unittest.mock import Mock

from opentelemetry.instrumentation.dify.utils import (
    get_timestamp_from_datetime_attr,
)


class TestGetTimestampFromDatetimeAttr(unittest.TestCase):
    def setUp(self):
        self.test_obj = Mock()

    def test_attr_exists_with_valid_datetime(self):
        dt_value = Mock(spec=datetime)
        dt_value.timestamp.return_value = 1698748000.123456
        dt_value.microsecond = 789000

        setattr(self.test_obj, "created_at", dt_value)

        result = get_timestamp_from_datetime_attr(self.test_obj, "created_at")

        expected_timestamp = int(
            1698748000.123456 * 1_000_000_000 + 789000 * 1_000
        )

        self.assertEqual(result, expected_timestamp)

    def test_attr_does_not_exist(self):
        result = get_timestamp_from_datetime_attr(
            self.test_obj, "non_existent_attr"
        )

        current_time_ns = time.time_ns()
        self.assertTrue(
            abs(result - current_time_ns) < 1_000_000,
            f"Returned timestamp {result} is not close to current time {current_time_ns}",
        )


if __name__ == "__main__":
    unittest.main()
