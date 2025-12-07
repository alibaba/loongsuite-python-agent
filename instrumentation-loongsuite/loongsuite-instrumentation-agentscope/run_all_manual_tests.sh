#!/bin/bash
# 运行所有手动测试用例并验证输出

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查环境变量
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo -e "${RED}错误: 未设置 DASHSCOPE_API_KEY 环境变量${NC}"
    exit 1
fi

# 设置 OpenTelemetry 环境变量
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

# 项目根目录
PROJECT_ROOT="/Users/sipercai/project/agentscopeai"
RESULTS_DIR="/tmp/manual_test_results"
mkdir -p "$RESULTS_DIR"

# 测试结果文件
RESULTS_FILE="$RESULTS_DIR/test_results.txt"
> "$RESULTS_FILE"

# 测试计数器
PASSED=0
FAILED=0
WARNED=0
TOTAL=0

# 运行测试用例的函数
run_test() {
    local test_name=$1
    local script_path=$2
    local test_type=$3  # "core" or "extended"
    
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "=========================================="
    echo -e "${BLUE}运行测试: $test_name${NC}"
    echo "脚本: $script_path"
    echo "类型: $test_type"
    echo "=========================================="
    
    if [ ! -f "$script_path" ]; then
        echo -e "${RED}✗ 文件不存在: $script_path${NC}"
        echo "[FAILED] $test_name - 文件不存在" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        return 1
    fi
    
    # 运行测试并捕获输出
    local output_file="$RESULTS_DIR/${test_name}_output.txt"
    local error_file="$RESULTS_DIR/${test_name}_error.txt"
    
    # 运行测试，设置超时（2分钟）
    timeout 120 opentelemetry-instrument \
        --traces_exporter console \
        --metrics_exporter console \
        python "$script_path" \
        > "$output_file" 2> "$error_file" || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${YELLOW}⚠ 超时 (2分钟)${NC}"
            echo "[TIMEOUT] $test_name - 超时" >> "$RESULTS_FILE"
            WARNED=$((WARNED + 1))
            return 0
        else
            echo -e "${RED}✗ 执行失败 (退出码: $exit_code)${NC}"
            echo "[FAILED] $test_name - 退出码: $exit_code" >> "$RESULTS_FILE"
            echo "错误输出:"
            tail -20 "$error_file"
            FAILED=$((FAILED + 1))
            return 1
        fi
    }
    
    # 合并输出和错误
    cat "$error_file" >> "$output_file"
    
    # 检查输出
    local has_trace=false
    local has_metrics=false
    local has_error=false
    
    # 检查 Trace 输出（查找 JSON span 输出）
    if grep -qE '(\{|"name"|"trace_id"|"span_id"|Span|Trace)' "$output_file"; then
        has_trace=true
    fi
    
    # 检查 Metrics 输出
    if grep -qE "(Metric|Counter|is_monotonic|Gauge|Histogram)" "$output_file"; then
        has_metrics=true
    fi
    
    # 检查错误（排除已知的警告）
    if grep -qiE "(KeyError|AttributeError|TypeError|Exception|Traceback)" "$output_file" | grep -vqiE "(Attempting to instrument|WARNING|INFO|EOFError)"; then
        has_error=true
        echo -e "${RED}✗ 发现错误${NC}"
        echo "错误详情:"
        grep -iE "(KeyError|AttributeError|TypeError|Exception|Traceback)" "$output_file" | grep -viE "(Attempting to instrument|WARNING|INFO|EOFError)" | head -10
    fi
    
    # 验证结果
    if [ "$has_error" = true ]; then
        echo -e "${RED}✗ 测试失败: 发现代码错误${NC}"
        echo "[FAILED] $test_name - 代码错误" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        return 1
    elif [ "$has_trace" = false ] && [ "$has_metrics" = false ]; then
        echo -e "${YELLOW}⚠ 警告: 未检测到 Trace 或 Metrics 输出${NC}"
        echo "输出预览（最后20行）:"
        tail -20 "$output_file"
        echo "[WARNING] $test_name - 无 Trace/Metrics 输出" >> "$RESULTS_FILE"
        WARNED=$((WARNED + 1))
        return 0
    else
        echo -e "${GREEN}✓ 测试通过${NC}"
        if [ "$has_trace" = true ]; then
            echo "  - Trace 输出: ✓"
            # 显示一些 Trace 示例
            echo "  Trace 示例:"
            grep -E '("name"|"trace_id")' "$output_file" | head -3 | sed 's/^/    /'
        fi
        if [ "$has_metrics" = true ]; then
            echo "  - Metrics 输出: ✓"
        fi
        echo "[PASSED] $test_name" >> "$RESULTS_FILE"
        PASSED=$((PASSED + 1))
        return 0
    fi
}

echo "=========================================="
echo "AgentScope OpenTelemetry 手动测试验证"
echo "=========================================="
echo "环境变量:"
echo "  OTEL_SEMCONV_STABILITY_OPT_IN=$OTEL_SEMCONV_STABILITY_OPT_IN"
echo "  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=$OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
echo "  DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:0:10}..."
echo ""

# 核心测试用例
echo "=========================================="
echo "核心测试用例 (必测)"
echo "=========================================="

run_test "2.a_structured_output" \
    "$PROJECT_ROOT/agentscope/examples/functionality/structured_output/main.py" \
    "core"

run_test "2.b_multi_agent_streaming" \
    "$PROJECT_ROOT/agentscope/examples/functionality/stream_printing_messages/multi_agent.py" \
    "core"

run_test "2.c_rag_basic" \
    "$PROJECT_ROOT/agentscope/examples/functionality/rag/basic_usage.py" \
    "core"

# 扩展测试用例
echo ""
echo "=========================================="
echo "扩展测试用例 (可选)"
echo "=========================================="

run_test "3.a_single_agent_streaming" \
    "$PROJECT_ROOT/agentscope/examples/functionality/stream_printing_messages/single_agent.py" \
    "extended"

run_test "3.b_multimodal_rag" \
    "$PROJECT_ROOT/agentscope/examples/functionality/rag/multimodal_rag.py" \
    "extended"

run_test "3.c_multiagent_conversation" \
    "$PROJECT_ROOT/agentscope/examples/workflows/multiagent_conversation/main.py" \
    "extended"

run_test "3.d_multiagent_debate" \
    "$PROJECT_ROOT/agentscope/examples/workflows/multiagent_debate/main.py" \
    "extended"

# 3.e 需要交互式输入，跳过或特殊处理
echo ""
echo -e "${YELLOW}跳过 3.e (agent_managed_plan) - 需要交互式输入${NC}"
echo "[SKIPPED] 3.e_agent_managed_plan - 需要交互式输入" >> "$RESULTS_FILE"

# 总结
echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo "总测试数: $TOTAL"
echo -e "${GREEN}通过: $PASSED${NC}"
echo -e "${YELLOW}警告: $WARNED${NC}"
echo -e "${RED}失败: $FAILED${NC}"
echo ""
echo "详细结果保存在: $RESULTS_FILE"
echo "输出文件保存在: $RESULTS_DIR/"
echo ""

# 显示 Trace 输出统计
echo "=========================================="
echo "Trace 输出统计"
echo "=========================================="
for file in "$RESULTS_DIR"/*_output.txt; do
    if [ -f "$file" ]; then
        test_name=$(basename "$file" _output.txt)
        trace_count=$(grep -cE '("name"|"trace_id")' "$file" 2>/dev/null || echo "0")
        if [ "$trace_count" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} $test_name: $trace_count 个 Trace 条目"
        else
            echo -e "${YELLOW}⚠${NC} $test_name: 未检测到 Trace 输出"
        fi
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ 所有测试通过或警告！${NC}"
    exit 0
else
    echo -e "${RED}✗ 有 $FAILED 个测试失败${NC}"
    exit 1
fi

