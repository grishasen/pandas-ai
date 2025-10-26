import os
from typing import Any, Callable, Dict, Optional, Union

import openai

from pandasai.exceptions import APIKeyNotFoundError, MissingModelError
from pandasai.helpers import load_dotenv
from .base import BaseOpenAI

load_dotenv()


class AzureOpenAI(BaseOpenAI):
    """OpenAI LLM via Microsoft Azure.

    Supports:
      - Chat Completions (`.chat.completions`)
      - Legacy Completions (`.completions`)
      - Responses API (`.responses`) for GPT-5-style reasoning models
        including `reasoning.effort` and `text.verbosity`.
    """

    """A function that returns an Azure Active Directory token.
    Will be invoked on every request.
    """
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    deployment_name: str
    api_version: str = ""
    api_base: str
    api_type: str = "azure"

    def __init__(
            self,
            api_token: Optional[str] = None,
            azure_endpoint: Union[str, None] = None,
            azure_ad_token: Union[str, None] = None,
            azure_ad_token_provider: Union[Callable[[], str], None] = None,
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            deployment_name: str = None,
            is_chat_model: bool = True,
            http_client: str = None,
            **kwargs,
    ):
        """
        Args:
            api_token (str): Azure OpenAI API token.
            azure_endpoint (str): <https://YOUR_RESOURCE_NAME.openai.azure.com/>
            azure_ad_token (str): AAD token.
            azure_ad_token_provider (Callable): provider for AAD token.
            api_version (str): Azure OpenAI API version.
            api_base (str): legacy param for openai<1.0 compatibility.
            deployment_name (str): name of your Azure deployment.
            is_chat_model (bool): legacy flag for chat vs completion.
            **kwargs: inference params (temperature, reasoning_effort, etc.)
        """

        self.api_token = (
                api_token
                or os.getenv("AZURE_OPENAI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
        )
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION")

        if self.api_token is None:
            raise APIKeyNotFoundError(
                "Azure OpenAI key is required. Please add an environment variable "
                "`AZURE_OPENAI_API_KEY` or `OPENAI_API_KEY` or pass `api_token` as a named parameter"
            )
        if self.azure_endpoint is None:
            raise APIKeyNotFoundError(
                "Azure endpoint is required. Please add an environment variable "
                "`AZURE_OPENAI_ENDPOINT` or pass `azure_endpoint` as a named parameter"
            )
        if self.api_version is None:
            raise APIKeyNotFoundError(
                "Azure OpenAI version is required. Please add an environment variable "
                "`OPENAI_API_VERSION` or pass `api_version` as a named parameter"
            )
        if deployment_name is None:
            raise MissingModelError(
                "No deployment name provided.",
                "Please include deployment name from Azure dashboard.",
            )

        self.azure_ad_token = azure_ad_token or os.getenv("AZURE_OPENAI_AD_TOKEN")
        self.azure_ad_token_provider = azure_ad_token_provider

        self.deployment_name = deployment_name

        self._is_chat_model = is_chat_model
        self.http_client = http_client

        self.openai_proxy = kwargs.get("openai_proxy") or os.getenv("OPENAI_PROXY")
        if self.openai_proxy:
            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}

        self._set_params(**kwargs)

        root_client = openai.AzureOpenAI(
            **self._client_params,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.deployment_name,
            azure_ad_token=self.azure_ad_token,
            azure_ad_token_provider=self.azure_ad_token_provider,
        )

        if self._is_responses_api_like(self.deployment_name):
            self._is_responses_model = True
            self._is_chat_model = True
            self.responses_client = root_client.responses
            self.client = root_client.chat.completions
        else:
            if self._is_chat_model:
                self.client = root_client.chat.completions
                self.responses_client = root_client.responses
            else:
                self.client = root_client.completions
                self.responses_client = root_client.responses

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        Default params, plus Azure deployment name instead of `model`.
        """
        return {
            **super()._default_params,
            "model": self.deployment_name,
        }

    @property
    def _client_params(self) -> Dict[str, Any]:
        client_params = {
            "api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.deployment_name,
            "azure_ad_token": self.azure_ad_token,
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "api_key": self.api_token,
            "http_client": self.http_client,
        }
        return {**client_params, **super()._client_params}

    @property
    def type(self) -> str:
        return "azure-openai"
