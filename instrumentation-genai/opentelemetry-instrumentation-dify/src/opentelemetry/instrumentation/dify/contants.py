import os
# dify app name resource key
DIFY_APP_NAME_KEY = "app.name"
DIFY_APP_ID_KEY = "app.id"

def _get_dify_app_name_key():
    """Get the Dify application name key from environment variable or default value.

    This function is for internal use only and should not be exposed to external users.
    It is not included in the semantic conventions as it is an implementation detail.

    Returns:
        str: The Dify application name key, either from environment variable DIFY_APP_NAME_KEY
             or the default DIFY_APP_NAME_KEY value.
    """
    dify_app_name_key = os.getenv("DIFY_APP_NAME_KEY", DIFY_APP_NAME_KEY)
    return dify_app_name_key

class NodeType:
    """
    Node Types.
    """
    START = "start"
    END = "end"
    ANSWER = "answer"
    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge-retrieval"
    IF_ELSE = "if-else"
    CODE = "code"
    TEMPLATE_TRANSFORM = "template-transform"
    QUESTION_CLASSIFIER = "question-classifier"
    HTTP_REQUEST = "http-request"
    TOOL = "tool"
    VARIABLE_AGGREGATOR = "variable-aggregator"
    VARIABLE_ASSIGNER = "variable-assigner"
    LOOP = "loop"
    ITERATION = "iteration"
    ITERATION_START = "iteration-start"  # fake start node for iteration
    PARAMETER_EXTRACTOR = "parameter-extractor"
    CONVERSATION_VARIABLE_ASSIGNER = "assigner"
