from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union, List

from pandasai.core.prompts.base import BasePrompt
from pandasai.helpers.memory import Memory
from pandasai.llm.base import LLM

if TYPE_CHECKING:
    from pandasai.agent.state import AgentState


class BaseOpenAI(LLM):
    """Base class to implement a new OpenAI LLM.

    LLM base class, extended by OpenAI and AzureOpenAI.
    """

    api_token: str
    api_base: str = "https://api.openai.com/v1"

    # legacy/chat-style sampling controls (still valid for gpt-4.x, gpt-3.5, etc.)
    temperature: float = 0
    max_tokens: int = 1000
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0.6
    best_of: int = 1
    n: int = 1
    stop: Optional[str] = None

    # Responses API / reasoning-era controls (GPT-5 etc.)  # NEW
    reasoning_effort: Optional[str] = "medium"  # "minimal" | "low" | "medium" | "high"
    verbosity: Optional[str] = "low"  # "low" | "medium" | "high"
    max_output_tokens: Optional[int] = 5000  # replaces max_tokens for GPT-5

    # misc
    request_timeout: Union[float, Tuple[float, float], Any, None] = None
    max_retries: int = 2
    seed: Optional[int] = None
    # support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client
    http_client: Union[Any, None] = None

    client: Any
    _is_chat_model: bool
    _is_responses_model: bool = False

    def _set_params(self, **kwargs):
        """
        Copy supported kwargs onto self so subclasses can pass through config.

        Keeping backward-compatible params like `temperature`, but going to conditionally drop them when talking to GPT-5 models.
        """
        valid_params = [
            # legacy/chat params
            "model",
            "deployment_name",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "n",
            "best_of",
            # connection
            "request_timeout",
            "max_retries",
            "openai_proxy",
            "default_headers",
            "default_query",
            "http_client",
            # Responses API params (GPT-5+)
            "reasoning_effort",
            "verbosity",
            "max_output_tokens",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    # Utility to know if current model is "GPT-5 style reasoning" and should use Responses API.
    @staticmethod
    def _is_responses_api_like(model_name: str) -> bool:
        return model_name.startswith("gpt-5")

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        Params that are *conceptually* available to this LLM, regardless of
        endpoint style.
        NOTE: We DO NOT filter here for GPT-5 unsupported params. That happens
        in the per-endpoint builders below.
        """
        params: Dict[str, Any] = {
            # classic knobs (chat/completions)
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "stop": self.stop,
            "n": self.n,
        }

        # classic token budget
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        return params

    @property
    def _responses_params(self) -> Dict[str, Any]:
        """
        Build params for Responses API (GPT-5 series).
        Intentionally omitting temperature/top_p/logprobs/etc., because GPT-5
        reasoning models don't support them. Instead, forwarding
        reasoning.effort and text.verbosity.
        """
        out: Dict[str, Any] = {}

        if self.reasoning_effort:
            out["reasoning"] = {"effort": self.reasoning_effort}

        if self.verbosity:
            out["text"] = {"verbosity": self.verbosity}

        if self.max_output_tokens is not None:
            out["max_output_tokens"] = self.max_output_tokens
        elif self.max_tokens is not None:
            out["max_output_tokens"] = self.max_tokens

        if self.stop is not None:
            out["stop"] = [self.stop]

        if self.seed is not None:
            out["seed"] = self.seed

        return out

    @property
    def _chat_params(self) -> Dict[str, Any]:
        """
        Build params for Chat Completions API.
        This is used for gpt-4.x / gpt-3.5 etc.
        """
        params = {**self._default_params}

        if params.get("stop") is not None:
            params["stop"] = [params["stop"]]

        return params

    @property
    def _completion_params(self) -> Dict[str, Any]:
        """
        Build params for the legacy text Completions API.
        Similar to _chat_params but with `prompt` instead of `messages`.
        """
        params = {**self._default_params}

        if params.get("stop") is not None:
            params["stop"] = [params["stop"]]

        return params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """
        Returns the object actually sent to the OpenAI client call.
        NOTE: This is *not* used directly anymore because different endpoints
        need different shapes. Leaving this for backward compatibility where
        subclasses expect _invocation_params to include credentials.
        """
        openai_creds: Dict[str, Any] = {}
        return {**openai_creds, **self._default_params}

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Params passed when constructing the OpenAI client."""
        return {
            "api_key": self.api_token,
            "base_url": self.api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
            "http_client": self.http_client,
        }

    def prepend_system_prompt(self, prompt: str, memory: Optional[Memory]) -> str:
        """Kept from previous codebase: combine memory/system style with new prompt."""
        if memory and hasattr(memory, "to_string"):
            return memory.to_string() + "\n" + prompt
        return prompt

    def completion(self, prompt: str, memory: Memory) -> str:
        """
        Legacy text completion endpoint (.completions.create).
        """
        full_prompt = self.prepend_system_prompt(prompt, memory)

        params = {
            **self._completion_params,
            "prompt": full_prompt,
        }

        response = self.client.create(**params)
        self.last_prompt = full_prompt
        return response.choices[0].text

    def chat_completion(self, value: str, memory: Memory) -> str:
        """
        Chat Completions endpoint (.chat.completions.create).
        """
        messages = memory.to_openai_messages() if memory else []
        messages.append(
            {
                "role": "user",
                "content": value,
            }
        )

        params = {
            **self._chat_params,
            "messages": messages,
        }

        response = self.client.create(**params)
        return response.choices[0].message.content

    def responses_completion(self, value: str, memory: Memory = None) -> str:
        """
        Responses API for GPT-5 / reasoning models (.responses.create).

        """
        input_messages: List[Dict[str, Any]] = (
            memory.to_openai_messages() if memory else []
        )
        input_messages.append({"role": "user", "content": value})

        params = {
            "model": getattr(self, "model", None)
                     or getattr(self, "deployment_name", None),
            "input": input_messages,
            **self._responses_params,
        }

        response = self.responses_client.create(**params)
        output_text = response.output_text
        self.last_prompt = value
        return output_text

    def call(self, instruction: BasePrompt, context: AgentState = None):
        """
        Unified entrypoint used by pandas-ai.

        We now branch 3 ways:
        - responses_completion (GPT-5 / Responses API)
        - chat_completion (gpt-4.x, gpt-3.5, etc.)
        - completion        (text-davinci-style legacy)
        """
        self.last_prompt = instruction.to_string()
        memory = context.memory if context else None

        if getattr(self, "_is_responses_model", False):
            return self.responses_completion(self.last_prompt, memory)

        if self._is_chat_model:
            return self.chat_completion(self.last_prompt, memory)

        return self.completion(self.last_prompt, memory)
