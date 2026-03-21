#!/usr/bin/env python3

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

"""
Simple MCP Server Example

This server provides a simple addition tool to demonstrate MCP server capabilities.
"""

from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Simple Math Server")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of the two numbers
    """
    result = a + b
    print(f"Calculation: {a} + {b} = {result}")
    return result


if __name__ == "__main__":
    print("Starting Simple MCP Server...")
    print("Server provides 'add_numbers' tool")
    print("Press Ctrl+C to stop the server")
    mcp.run()
