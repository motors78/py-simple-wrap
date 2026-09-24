"""
easy_ai wraps common LangChain functionality to make it easier to use.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr


class EasyAIError(Exception):
    """
    Raised when a call to an AI model or provider cannot be completed.
    Args:
        message (str): Description of what went wrong.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _is_exit_command(text: str) -> bool:
    """
    Checks whether a piece of user input should end the chat loop.

    Args:
        text (str): The raw text the user typed.

    Returns:
        True if `text` matches "exit", "quit", "stop", or "bye"
        (case-insensitive), False otherwise.
    """
    return text.lower() in ("exit", "quit", "stop", "bye")


def get_model(
    provider: str,
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
) -> BaseChatModel:
    """
    Returns a LangChain chat model instance for the given provider,
    without you having to remember each provider's import path and
    constructor arguments.

    Args:
        provider (str): Name of the LLM provider. One of "openai",
            "ollama", "anthropic", "google", or "mistral". Case-insensitive.
        model_name (str): Name of the model to use (e.g., "gpt-4o",
            "llama3", "claude-sonnet-4-6").
        api_key (str): API key for the provider, if required. Not used
            for "ollama". Defaults to None.
        base_url (str): Custom base URL for the provider. Used for
            "openai" and "ollama" (defaults to
            "http://localhost:11434" for ollama if not provided).
            Defaults to None.
        timeout (int): Request timeout in seconds. Currently only used
            for "anthropic". Defaults to 30.

    Returns:
        A LangChain chat model instance corresponding to the given
        provider.

    Raises:
        EasyAIError: If `provider` isn't one of the supported providers.

    Example:
        === "The Py_simple Way"
            ```python
            from py_simple import get_model

            model = get_model("anthropic", "claude-sonnet-4-6")
