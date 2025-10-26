import os
from typing import Any, Dict, Optional

import openai

from pandasai.exceptions import APIKeyNotFoundError, UnsupportedModelError
from pandasai.helpers import load_dotenv
from .base import BaseOpenAI

load_dotenv()


class OpenAI(BaseOpenAI):
    """OpenAI LLM using BaseOpenAI.

    - gpt-3.5 / gpt-4.x -> Chat Completions API
    - legacy instruct   -> Completions API
    - gpt-5*            -> Responses API (recommended by OpenAI since GPT 5).
    """

    _supported_chat_models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
    ]

    _supported_completion_models = [
        "gpt-3.5-turbo-instruct"
    ]

    _supported_responses_models = [
        # GPT-5 family uses Responses API, with reasoning_effort + verbosity.
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ]

    model: str = "gpt-5-mini"

    def __init__(
            self,
            api_token: Optional[str] = None,
            **kwargs,
    ):
        """
        Args:
            api_token (str): API Token for OpenAI platform.
            **kwargs: Passed through to BaseOpenAI._set_params
        """
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None
        if not self.api_token:
            raise APIKeyNotFoundError("OpenAI API key is required")

        # base URL override (for self-host / Azure-style compat); default keeps public api
        self.api_base = (
                kwargs.get("api_base") or os.getenv("OPENAI_API_BASE") or self.api_base
        )

        self.openai_proxy = kwargs.get("openai_proxy") or os.getenv("OPENAI_PROXY")
        if self.openai_proxy:
            openai.proxy = {
                "http": self.openai_proxy,
                "https": self.openai_proxy,
            }

        self._set_params(**kwargs)
        root_client = openai.OpenAI(**self._client_params)
        model_name = self.model.split(":")[1] if "ft:" in self.model else self.model

        if model_name in self._supported_responses_models or self._is_responses_api_like(model_name):
            self._is_responses_model = True
            self._is_chat_model = True
            self.responses_client = root_client.responses
            self.client = root_client.chat.completions
        elif model_name in self._supported_chat_models:
            self._is_chat_model = True
            self._is_responses_model = False
            self.client = root_client.chat.completions
            # we *still* give responses_client to avoid attribute errors if someone
            # accidentally calls responses_completion()
            self.responses_client = root_client.responses
        elif model_name in self._supported_completion_models:
            self._is_chat_model = False
            self._is_responses_model = False
            self.client = root_client.completions
            self.responses_client = root_client.responses
        else:
            raise UnsupportedModelError(self.model)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        Merge BaseOpenAI params + current model name.
        """
        return {
            **super()._default_params,
            "model": self.model,
        }

    @property
    def type(self) -> str:
        return "openai"
