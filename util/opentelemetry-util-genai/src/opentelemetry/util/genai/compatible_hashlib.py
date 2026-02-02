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

import hashlib as _hashlib
import sys
from hashlib import *  # noqa: F403  # pylint: disable=wildcard-import,unused-wildcard-import  # pyright: ignore[reportWildcardImportFromLibrary]


def _make_compatible(orig_func):  # type: ignore
    def compatible_func(content: bytes = b"", *, usedforsecurity: bool = True):  # type: ignore
        return orig_func(content)  # type: ignore

    return compatible_func


def _setup_hashlib_compatibility():
    """Setup hashlib compatibility to support Python 3.8"""
    if sys.version_info < (3, 9):
        # List of hash functions that need usedforsecurity parameter support
        functions_to_patch = [
            "md5",
            "sha1",
            "sha224",
            "sha256",
            "sha384",
            "sha512",
        ]

        for func_name in functions_to_patch:
            if hasattr(_hashlib, func_name):
                original_func = getattr(_hashlib, func_name)
                # Apply patch
                compatible_func = _make_compatible(original_func)
                setattr(_hashlib, func_name, compatible_func)

                # Also update the reference in the current module
                globals()[func_name] = compatible_func


_setup_hashlib_compatibility()
