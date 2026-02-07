"""Multi-provider LLM inference client supporting OpenAI, Anthropic, Groq, DeepSeek, and HuggingFace."""

import logging
import time
import os
from typing import List, Optional, Dict

from .schemas import GenerationParams, GenerationResult

logger = logging.getLogger(__name__)

# Provider configuration reference
PROVIDER_CONFIGS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # Uses default OpenAI endpoint
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,  # Uses anthropic SDK directly
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "huggingface": {
        "env_key": "HF_TOKEN",
        "base_url": "https://router.huggingface.co/v1",
    },
}


class MultiProviderClient:
    """Unified inference client that routes requests to multiple LLM API providers.
    
    Supports:
      - OpenAI (GPT-5.2, etc.) via the openai SDK
      - Anthropic (Claude Opus 4.6, etc.) via the anthropic SDK
      - Groq (Llama 4 Maverick, etc.) via OpenAI-compatible API
      - DeepSeek (DeepSeek V3.2, etc.) via OpenAI-compatible API
      - HuggingFace (Llama 3.1, Qwen, etc.) via OpenAI-compatible API
    """

    def __init__(
        self,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize the multi-provider client.

        Args:
            initial_delay: Initial delay for rate-limit backoff (seconds).
            max_delay: Maximum delay for exponential backoff (seconds).
            backoff_factor: Multiplier for exponential backoff.
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        # OpenAI-compatible clients keyed by provider name
        self._openai_clients: Dict[str, object] = {}
        # Anthropic client (separate SDK)
        self._anthropic_client = None

        self._init_providers()

    # ------------------------------------------------------------------
    # Provider initialisation
    # ------------------------------------------------------------------

    def _init_providers(self):
        """Lazily initialise SDK clients for every provider whose API key is set."""
        try:
            from openai import OpenAI
        except ImportError:
            OpenAI = None
            logger.warning("openai package not installed – OpenAI / Groq / HuggingFace providers unavailable")

        # --- OpenAI ---
        if OpenAI is not None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._openai_clients["openai"] = OpenAI(api_key=api_key)
                logger.info("Initialized OpenAI client")

        # --- Groq (OpenAI-compatible) ---
        if OpenAI is not None:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                self._openai_clients["groq"] = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                logger.info("Initialized Groq client")

        # --- DeepSeek (OpenAI-compatible) ---
        if OpenAI is not None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                self._openai_clients["deepseek"] = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1",
                )
                logger.info("Initialized DeepSeek client")

        # --- HuggingFace (OpenAI-compatible) ---
        if OpenAI is not None:
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
            if api_key:
                self._openai_clients["huggingface"] = OpenAI(
                    api_key=api_key,
                    base_url="https://router.huggingface.co/v1",
                )
                logger.info("Initialized HuggingFace client")

        # --- Anthropic ---
        try:
            import anthropic as _anthropic_mod

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._anthropic_client = _anthropic_mod.Anthropic(api_key=api_key)
                logger.info("Initialized Anthropic client")
        except ImportError:
            logger.warning("anthropic package not installed – Anthropic models unavailable")

    def get_available_providers(self) -> List[str]:
        """Return list of providers that have been successfully initialised."""
        providers = list(self._openai_clients.keys())
        if self._anthropic_client is not None:
            providers.append("anthropic")
        return providers

    # ------------------------------------------------------------------
    # Low-level request helpers (with retry / backoff)
    # ------------------------------------------------------------------

    def _get_openai_client(self, provider: str):
        """Return the OpenAI-compatible client for *provider*, or raise."""
        client = self._openai_clients.get(provider)
        if client is None:
            env_key = PROVIDER_CONFIGS.get(provider, {}).get("env_key", "UNKNOWN")
            raise ValueError(
                f"No client initialised for provider '{provider}'. "
                f"Set the {env_key} environment variable."
            )
        return client

    def _call_openai_compatible(
        self,
        provider: str,
        model: str,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        max_retries: int = 5,
    ) -> str:
        """Make a chat-completion request to an OpenAI-compatible API with retry."""
        client = self._get_openai_client(provider)
        delay = self.initial_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                # OpenAI GPT-5+ models require max_completion_tokens instead of max_tokens
                if provider == "openai":
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                return response.choices[0].message.content
            except Exception as e:
                last_exception = e
                error_str = str(e)

                if "429" in error_str or "rate" in error_str.lower():
                    logger.warning(
                        f"[{provider}] Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {delay:.1f}s…"
                    )
                elif "503" in error_str or "loading" in error_str.lower() or "overloaded" in error_str.lower():
                    logger.warning(
                        f"[{provider}] Service unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {delay:.1f}s…"
                    )
                else:
                    logger.error(
                        f"[{provider}] API error (attempt {attempt + 1}/{max_retries}): "
                        f"{error_str[:200]}"
                    )

                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)

        raise Exception(
            f"[{provider}] Failed after {max_retries} retries. Last error: {last_exception}"
        )

    def _call_anthropic(
        self,
        model: str,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        max_retries: int = 5,
    ) -> str:
        """Make a request to the Anthropic Messages API with retry."""
        if self._anthropic_client is None:
            raise ValueError(
                "Anthropic client not initialised. Set the ANTHROPIC_API_KEY environment variable."
            )

        delay = self.initial_delay
        last_exception = None

        # Convert to Anthropic message format (only user/assistant roles)
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        for attempt in range(max_retries):
            try:
                response = self._anthropic_client.messages.create(
                    model=model,
                    messages=anthropic_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content[0].text
            except Exception as e:
                last_exception = e
                error_str = str(e)

                if "429" in error_str or "rate" in error_str.lower():
                    logger.warning(
                        f"[anthropic] Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {delay:.1f}s…"
                    )
                elif "overloaded" in error_str.lower() or "529" in error_str:
                    logger.warning(
                        f"[anthropic] Overloaded (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {delay:.1f}s…"
                    )
                else:
                    logger.error(
                        f"[anthropic] API error (attempt {attempt + 1}/{max_retries}): "
                        f"{error_str[:200]}"
                    )

                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)

        raise Exception(
            f"[anthropic] Failed after {max_retries} retries. Last error: {last_exception}"
        )

    def _make_request(
        self,
        model: str,
        provider: str,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        max_retries: int = 5,
    ) -> str:
        """Route a chat-completion request to the correct provider."""
        if provider == "anthropic":
            return self._call_anthropic(model, messages, temperature, max_tokens, max_retries)
        else:
            return self._call_openai_compatible(provider, model, messages, temperature, max_tokens, max_retries)

    # ------------------------------------------------------------------
    # High-level generation methods
    # ------------------------------------------------------------------

    def generate_greedy(
        self,
        model: str,
        provider: str,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> GenerationResult:
        """
        Generate a single answer using greedy / near-deterministic decoding.

        Args:
            model: Model identifier (e.g. "gpt-5.2", "claude-opus-4-6").
            provider: Provider name ("openai", "anthropic", "groq", "huggingface").
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with generated text and parameters.
        """
        messages = [{"role": "user", "content": prompt}]

        # Anthropic supports temperature=0 for fully deterministic output
        temperature = 0.0 if provider == "anthropic" else 0.01

        text = self._make_request(
            model=model,
            provider=provider,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        gen_params = GenerationParams(
            do_sample=False,
            temperature=temperature,
            top_p=1.0,
            top_k=None,
            max_new_tokens=max_new_tokens,
        )

        return GenerationResult(text=text.strip(), params=gen_params, logprobs=None)

    def generate_stochastic(
        self,
        model: str,
        provider: str,
        prompt: str,
        num_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 50,
    ) -> List[GenerationResult]:
        """
        Generate multiple stochastic samples for self-consistency analysis.

        Args:
            model: Model identifier.
            provider: Provider name.
            prompt: Input prompt.
            num_samples: Number of samples to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter (used where supported).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of GenerationResult objects.
        """
        messages = [{"role": "user", "content": prompt}]

        gen_params = GenerationParams(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=None,
            max_new_tokens=max_new_tokens,
        )

        results = []
        for i in range(num_samples):
            try:
                text = self._make_request(
                    model=model,
                    provider=provider,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                results.append(
                    GenerationResult(text=text.strip(), params=gen_params, logprobs=None)
                )
            except Exception as e:
                logger.error(f"Failed to generate sample {i + 1}/{num_samples}: {e}")
                continue

        if not results:
            raise Exception("Failed to generate any stochastic samples")

        return results


def build_qa_prompt(question: str, model: str = "") -> str:
    """
    Build a prompt for factual QA generation.

    The same prompt is used across all providers / models so that results are
    comparable.

    Args:
        question: The question text.
        model: Model identifier (reserved for future model-specific formatting).

    Returns:
        Formatted prompt string.
    """
    prompt = f"""Answer the following question with a short, factual answer. Give only the answer, no explanation.

Question: {question}

Answer:"""
    return prompt


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing MultiProviderClient …\n")

    try:
        from dotenv import load_dotenv

        load_dotenv()

        client = MultiProviderClient()
        available = client.get_available_providers()
        print(f"Available providers: {available}")

        test_prompt = build_qa_prompt("What is the capital of France?")
        print(f"\nTest prompt:\n{test_prompt}")

        # Try each available provider with a lightweight call
        test_models = {
            "openai": "gpt-5.2",
            "anthropic": "claude-opus-4-6",
            "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "deepseek": "deepseek-chat",
            "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
        }

        for provider in available:
            model = test_models.get(provider, "unknown")
            print(f"\nTesting greedy generation with {provider} / {model} …")
            result = client.generate_greedy(model, provider, test_prompt)
            print(f"  Result: {result.text}")

    except Exception as e:
        print(f"Error: {e}")
