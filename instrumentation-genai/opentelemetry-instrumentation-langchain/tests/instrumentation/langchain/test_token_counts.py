import random
import pytest
from opentelemetry.instrumentation.langchain.internal._tracer import _token_counts
# 新增导入 langchain_core 的 Generation 和 AIMessage
from langchain_core.outputs import Generation, ChatGeneration
from langchain_core.messages import AIMessage

# 定义常量以替代aliyun版本
LLM_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
LLM_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
LLM_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

def random_token_triplet():
    prompt_tokens = random.randint(1, 1000)
    output_tokens = random.randint(1, 1000)
    total_tokens = prompt_tokens + output_tokens
    return prompt_tokens, output_tokens, total_tokens

def case_generation_info_token_usage():
    # generations[0][0].generation_info.token_usage
    prompt_tokens, output_tokens, total_tokens = random_token_triplet()
    # 用 Generation 类
    gen = Generation(
        text="hello",
        generation_info={
            "token_usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        }
    )
    outputs = {
        "generations": [[gen]]
    }
    return outputs, prompt_tokens, output_tokens, total_tokens

def case_message_response_metadata_token_usage():
    # generations[0][0].message.response_metadata.token_usage
    prompt_tokens, output_tokens, total_tokens = random_token_triplet()
    # 用 Generation + AIMessage
    ai_msg = AIMessage(
        content="hi",
        response_metadata={
            "token_usage": {
                "completion_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
    )
    gen = ChatGeneration(
        text="hello",
        message=ai_msg
    )
    outputs = {
        "generations": [[gen]]
    }
    return outputs, prompt_tokens, output_tokens, total_tokens

def case_message_response_metadata_token_usage_uppercase():
    # generations[0][0].message.response_metadata.token_usage (大写key)
    prompt_tokens, output_tokens, total_tokens = random_token_triplet()
    ai_msg = AIMessage(
        content="hi",
        response_metadata={
            "token_usage": {
                "PromptTokens": prompt_tokens,
                "CompletionTokens": output_tokens,
                "TotalTokens": total_tokens
            }
        }
    )
    gen = ChatGeneration(
        text="hello",
        message=ai_msg
    )
    outputs = {
        "generations": [[gen]]
    }
    return outputs, prompt_tokens, output_tokens, total_tokens

def case_llm_output_token_usage_priority():
    # llm_output.token_usage 优先，generations 也有 token_usage，但只取 llm_output
    prompt_tokens, output_tokens, total_tokens = random_token_triplet()
    ai_msg = AIMessage(content="hi")
    gen = ChatGeneration(
        text="hello",
        message=ai_msg,  # 这里必须传 message
        generation_info={
            "token_usage": {
                "PromptTokens": prompt_tokens,
                "CompletionTokens": output_tokens,
                "TotalTokens": total_tokens
            }
        }
    )
    outputs = {
        "generations": [[gen]],
        "llm_output": {
            "token_usage": {
                "completion_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
    }
    return outputs, prompt_tokens, output_tokens, total_tokens

def case_llm_output_empty_should_fallback():
    # llm_output存在但无token_usage，应该跳过，继续找generations
    prompt_tokens, output_tokens, total_tokens = random_token_triplet()
    ai_msg = AIMessage(
        content="hi",
        response_metadata={
            "token_usage": {
                "completion_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
    )
    gen = ChatGeneration(
        text="hello",
        message=ai_msg
    )
    outputs = {
        "generations": [[gen]],
        "llm_output": {}
    }
    return outputs, prompt_tokens, output_tokens, total_tokens

@pytest.mark.parametrize("build_case", [
    case_generation_info_token_usage,
    case_message_response_metadata_token_usage,
    case_message_response_metadata_token_usage_uppercase,
    case_llm_output_token_usage_priority,
    case_llm_output_empty_should_fallback,
])
def test_token_counts_real_formats_minimal(build_case):
    outputs, prompt_tokens, output_tokens, total_tokens = build_case()
    result = dict(_token_counts(outputs))

    assert result[LLM_USAGE_PROMPT_TOKENS] == prompt_tokens
    assert result[LLM_USAGE_COMPLETION_TOKENS] == output_tokens
    assert result[LLM_USAGE_TOTAL_TOKENS] == total_tokens