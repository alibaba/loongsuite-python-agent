"""Extract test cases from test_message_flow_cases.py and save as YAML cassettes."""

import json
import yaml
from pathlib import Path

# 导入测试用例
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_message_flow_cases import (
    TEST_CASE_1_FOO_SH_COMMAND,
    TEST_CASE_2_ECHO_COMMAND,
    TEST_CASE_3_PRETOOLUSE,
)


def save_test_case_as_cassette(test_case, filename):
    """保存测试用例为 YAML cassette 文件。"""
    cassette_data = {
        "description": test_case["description"],
        "prompt": test_case["prompt"],
        "messages": test_case["messages"],
    }
    
    # 如果有 expected_spans，也保存
    if "expected_spans" in test_case:
        cassette_data["expected_spans"] = test_case["expected_spans"]
    
    cassettes_dir = Path(__file__).parent / "cassettes"
    cassettes_dir.mkdir(exist_ok=True)
    
    output_file = cassettes_dir / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(
            cassette_data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )
    
    print(f"✅ Saved: {output_file}")
    return output_file


def main():
    """Extract and save all test cases."""
    print("Extracting test cases to cassettes...\n")
    
    # 保存三个测试用例
    save_test_case_as_cassette(
        TEST_CASE_1_FOO_SH_COMMAND,
        "test_foo_sh_command.yaml"
    )
    
    save_test_case_as_cassette(
        TEST_CASE_2_ECHO_COMMAND,
        "test_echo_command.yaml"
    )
    
    save_test_case_as_cassette(
        TEST_CASE_3_PRETOOLUSE,
        "test_pretooluse_hook.yaml"
    )
    
    print("\n✅ All test cases extracted successfully!")


if __name__ == "__main__":
    main()
