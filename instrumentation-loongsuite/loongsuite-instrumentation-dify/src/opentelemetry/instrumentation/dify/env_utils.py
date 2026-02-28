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

import socket
from os import environ

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


def get_hostname():
    try:
        hostname = socket.gethostname()
        return hostname
    except socket.error as e:
        print(f"Unable to get hostname: {e}")


def get_ip_address():
    try:
        # 获取本地主机名
        hostname = socket.gethostname()
        # 获取本地IP
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.error as e:
        print(f"Unable to get IP Address: {e}")


def is_capture_content_enabled() -> bool:
    capture_content = environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "true"
    )
    return is_true_value(capture_content)


def convert_to_env_var(env_key: str) -> str:
    return env_key.replace(".", "_").upper()


def is_true_value(value) -> bool:
    return value.lower() in {"1", "y", "yes", "true"}
