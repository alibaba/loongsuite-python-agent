# -*- coding: utf-8 -*-
"""顶层测试配置，配置 Python 路径以便测试可以导入包"""

import sys
from pathlib import Path

# 获取项目根目录
TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
SRC_PATH = PROJECT_ROOT / "src"

# 将 src 目录添加到 Python 路径
# 这样测试就可以直接导入 opentelemetry.instrumentation.agentscope
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# 将 tests 目录的父目录添加到 Python 路径
# 这样测试就可以使用 `from tests.shared.version_utils import ...`
# 无论 pytest 从哪个目录运行
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
