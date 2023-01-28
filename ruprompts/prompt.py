import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils.hub import PushToHubMixin, cached_file
from typeguard import typechecked

from ruprompts.prompt_embedding import (
    BasePromptEmbedding,
    MultiPromptEmbedding,
    PromptEmbedding,
    PromptEmbeddingSafe,
)
from ruprompts.prompt_format import BasePromptFormat, PromptFormat, PromptFormatSafe
from ruprompts.prompt_provider import BasePromptProvider, TensorPromptProvider

PROMPT_PROVIDER_FILE_NAME = "prompt_provider.bin"
PROMPT_FILE_NAME = "prompt.json"


@dataclass
class PromptConfig:
    embedding_dim: Optional[int] = None
    prompt_length: Optional[int] = None
    pretrained_model_name: Optional[str] = None
    is_safe: bool = False

    def as_dict(self):
        return asdict(self)


@dataclass
class PromptContext:
    is_initialized: bool = False
    is_attached: bool = False
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    embedding: Optional[BasePromptEmbedding] = None


@typechecked
class Prompt(PushToHubMixin):
    """Core class combining :s:`ruprompts.prompt_format.PromptFormat` and :s:`ruprompts.prompt_provider.BasePromptProvider`.

    Implements saving/loading methods and HF hub integration.

    Examples:
        >>> p = Prompt(PromptFormat("<P*50>"), TensorPromptProvider())
        >>> p.patch(model, tokenizer)
        >>> trainer.train(model)
        >>> p.save_pretrained("./checkpoint/path")
        >>> p.push_to_hub("konodyuk/prompt_rugpt3large_detox")

        >>> p = Prompt.from_pretrained("./checkpoint/path")
        >>> p.patch(model, tokenizer)

        >>> a = p(toxic="...")
        >>> b = p.format(toxic="...")
        >>> assert a == b

        >>> p = Prompt.from_pretrained("konodyuk/prompt_rugpt3large_detox")
        >>> ppln = pipeline("text-generation", model=model, tokenizer=tokenizer)
        >>> ppln_prompt = pipeline("text-generation-with-prompt", prompt=p, model=model, tokenizer=tokenizer)
        >>> a = ppln(p("text"))
        >>> b = ppln_prompt("text")
        >>> assert a == b

    Args:
        format (# !s!`ruprompts.prompt_format.BasePromptFormat`): Format used to format text for training and inference
            and for adding special tokens to the tokenizer.
        provider (# !s!`ruprompts.prompt_provider.BasePromptProvider`): Provider used to insert trainable embeddings
            to the positions defined by prompt format.
    """

    def __init__(
        self,
        format: BasePromptFormat,
        provider: BasePromptProvider,
        config: Optional[PromptConfig] = None,
    ):
        self.format = format
        self.provider = provider

        if config is None:
            config = PromptConfig()
        self.config = config

        self.ctx = PromptContext()

    def patch(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        """Applies the prompt to model and tokenizer.

        Injects the prompt by adding special prompt tokens to the tokenizer
        and switching input embedding layer of the model with prompt embedding
        layer that inserts embeddings from prompt provider into the positions
        defined by special tokens specified in prompt format.

        Args:
            model: Model to patch.
            tokenizer: Tokenizer to patch.
        """

        self.initialize(model, tokenizer)
        self.attach(model, tokenizer)

    def initialize(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.ctx.model = model
        self.ctx.tokenizer = tokenizer

        self.format.initialize(tokenizer)

        default_embedding = _extract_default_embedding(model)

        config = self.config
        config.pretrained_model_name = model.config.name_or_path
        config.prompt_length = self.format.prompt_length
        config.embedding_dim = default_embedding.embedding_dim

        self.provider.initialize(
            prompt_length=config.prompt_length,
            embedding_dim=config.embedding_dim,
            embedding=default_embedding,
            init_tokens=self.format.init_tokens,
        )

        self.ctx.is_initialized = True

    def attach(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        if self.is_attached:
            self.detach()
            # raise UserWarning("Prompt can be attached only once")

        if not self.is_initialized:
            raise UserWarning("Prompt should be initialized to be attached")

        self.format.add_tokens(tokenizer)

        default_embedding = _extract_default_embedding(model)
        prompt_token_ids = self.format.prompt_token_ids

        prompt_embedding_cls = PromptEmbedding
        if isinstance(self.format, PromptFormatSafe):
            prompt_embedding_cls = PromptEmbeddingSafe
        self.ctx.embedding = prompt_embedding_cls(
            default_embedding, prompt_provider=self.provider, prompt_token_ids=prompt_token_ids
        )

        model.set_input_embeddings(self.ctx.embedding)

        self.ctx.is_attached = True

    def detach(self, model: Optional[PreTrainedModel] = None):
        if model is not None:
            default_embedding = _extract_default_embedding(model)
            model.set_input_embeddings(default_embedding)

        self.ctx.is_attached = False

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs
    ):
        """
        Save a prompt to a directory, so that it can be re-loaded using the
        :c:`ruprompts.prompt.Prompt.from_pretrained` class method.

        Args:
            save_directory: Directory to which to save. Will be created if it doesn't exist.
            push_to_hub: Whether or not to push your model to the Hugging Face model hub after saving it.

            !!! warning

                Using `push_to_hub=True` will synchronize the repository you are pushing to with
                `save_directory`, which requires `save_directory` to be a local clone of the repo you are
                pushing to if it's an existing folder. Pass along `temp_dir=True` to use a temporary directory
                instead.

            **kwargs: Additional key word arguments passed along to the
                [`PushToHubMixin.push_to_hub`](https://huggingface.co/docs/transformers/main_classes/model#transformers.file_utils.PushToHubMixin.push_to_hub) method.
        """
        if not self.is_initialized:
            raise UserWarning("Prompt should be initialized to be saved")

        if os.path.isfile(save_directory):
            raise UserWarning("save_directory should be a directory, got file instead")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        output_prompt_provider_file = os.path.join(save_directory, PROMPT_PROVIDER_FILE_NAME)
        self.provider.save_pretrained(output_prompt_provider_file)

        output_prompt_file = os.path.join(save_directory, PROMPT_FILE_NAME)
        json.dump(self.as_dict(), open(output_prompt_file, "w", encoding="utf-8"))

        if push_to_hub:
            self._push_to_hub(repo, commit_message=commit_message)

    @classmethod
    def from_pretrained(
        cls, pretrained_prompt_name_or_path: Union[str, os.PathLike], as_safe: bool = False
    ) -> "Prompt":
        """Loads a pretrained prompt from disk or HF Hub.

        Args:
            pretrained_prompt_name_or_path: Either a HF Hub identifier (`konodyuk/prompt_rugpt3large_detox`)
                or path to a directory containing prompt saved with :s:`ruprompts.prompt.Prompt.save_pretrained`.
            as_safe: Whether to load prompt format as :s:`ruprompts.prompt_format.PromptFormat`
                or :s:`ruprompts.prompt_format.PromptFormatSafe`.

        Returns:
            # !s!`ruprompts.prompt.Prompt`: Pretrained prompt instance.
        """
        if os.path.isdir(pretrained_prompt_name_or_path):
            prompt_file = os.path.join(pretrained_prompt_name_or_path, PROMPT_FILE_NAME)
            prompt_provider_file = os.path.join(
                pretrained_prompt_name_or_path, PROMPT_PROVIDER_FILE_NAME
            )
        else:
            prompt_file = _resolve_file(pretrained_prompt_name_or_path, PROMPT_FILE_NAME)
            prompt_provider_file = _resolve_file(
                pretrained_prompt_name_or_path, PROMPT_PROVIDER_FILE_NAME
            )

        with open(prompt_file, "r") as f:
            prompt_dict = json.load(f)

        prompt_format_cls = PromptFormat
        if as_safe:
            prompt_format_cls = PromptFormatSafe
        prompt_format = prompt_format_cls(**prompt_dict["format"])
        prompt_config = PromptConfig(**prompt_dict.get("config", {}))

        with open(prompt_provider_file, "rb") as f:
            prompt_provider_weights = torch.load(f)
        prompt_provider = TensorPromptProvider.from_pretrained(prompt_provider_weights)

        prompt = cls(format=prompt_format, provider=prompt_provider, config=prompt_config)
        prompt.ctx.is_initialized = True

        return prompt

    def get_default_model(self):
        if self.config.pretrained_model_name is None:
            raise UserWarning("Default model/tokenizer is not specified")
        return AutoModel.from_pretrained(self.config.pretrained_model_name)

    def get_default_tokenizer(self):
        if self.config.pretrained_model_name is None:
            raise UserWarning("Default model/tokenizer is not specified")
        return AutoTokenizer.from_pretrained(self.config.pretrained_model_name)

    def __call__(self, text: Optional[str] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        return self.format(**kwargs)

    def as_dict(self):
        if not self.is_initialized:
            raise UserWarning("Prompt should be initialized before serialization")
        return {"format": self.format.as_dict(), "config": self.config.as_dict()}

    @property
    def is_initialized(self) -> bool:
        return self.ctx.is_initialized

    @property
    def is_attached(self) -> bool:
        return self.ctx.is_attached


@typechecked
class MultiPrompt:
    """Implements serving multiple prompts with one model.

    Receives a dict of pretrained prompts with string keys.
    These keys are used to switch formats.

    Args:
        prompts (# dict with string keys and !s!`ruprompts.prompt.Prompt` values):

    Examples:

        >>> mp = MultiPrompt({
        ...     "key1": "path/to/pretrained/prompt1",
        ...     "key2": "hfhub/prompt_id",
        ...     "key3": Prompt.from_pretrained("another_hfhub/prompt_id"),
        ...     'key4": Prompt.from_pretrained("/path/to/another/checkpoint")
        ... })
        >>> mp.patch(model, tokenizer)
        >>> ppln = pipeline("text-generation", model=model, tokenizer=tokenizer)
        >>> ppln(mp(key="key2", "Text for second prompt"))
        >>> ppln(mp(key="key3", text="Text for third prompt"))
        >>> ppln(mp(key="key4", keyword="Keyword for fourth prompt"))
    """

    def __init__(self, prompts: Optional[Dict[str, Union[Prompt, str]]] = None):
        if prompts is None:
            prompts = {}
        self.prompts: Dict[str, Prompt] = {}

        self.ctx = PromptContext()

        for key, prompt in prompts.items():
            self.add_prompt(key=key, prompt=prompt)

    def add_prompt(self, key: str, prompt: Union[Prompt, str]):
        if key in self.prompts:
            raise UserWarning(f"Prompt with key '{key}' is already added")

        if isinstance(prompt, str):
            prompt = Prompt.from_pretrained(prompt)

        if self.is_attached and not prompt.is_initialized:
            prompt.initialize(self.ctx.model, self.ctx.tokenizer)

        prompt.format.set_key(key)

        self.prompts[key] = prompt

    def __call__(self, key: str, text: Optional[str] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        return self.formats[key](**kwargs)

    def patch(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.initialize(model, tokenizer)
        self.attach(model, tokenizer)

    def initialize(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        for prompt in self.prompts.values():
            prompt.initialize(model, tokenizer)

    def attach(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        if self.is_attached:
            raise UserWarning("MultiPrompt can be attached only once")

        if not self.is_initialized:
            raise UserWarning("MultiPrompt should be initialized to be attached")

        for format in self.formats.values():
            format.add_tokens(tokenizer)

        default_embedding = _extract_default_embedding(model)

        embeddings = []
        for prompt in self.prompts.values():
            prompt_embedding_cls = PromptEmbedding
            if isinstance(prompt.format, PromptFormatSafe):
                prompt_embedding_cls = PromptEmbeddingSafe

            prompt_token_ids = prompt.format.prompt_token_ids
            embedding = prompt_embedding_cls(
                default_embedding,
                prompt_provider=prompt.provider,
                prompt_token_ids=prompt_token_ids,
            )

            embeddings.append(embedding)

        self.ctx.embedding = MultiPromptEmbedding(
            default_embedding=default_embedding, embeddings=embeddings
        )

        model.set_input_embeddings(self.ctx.embedding)

        self.ctx.is_attached = True

    @property
    def is_initialized(self) -> bool:
        for prompt in self.prompts.values():
            if not prompt.is_initialized:
                return False
        return True

    @property
    def is_attached(self) -> bool:
        return self.ctx.is_attached

    @property
    def formats(self) -> Dict[str, BasePromptFormat]:
        return {key: prompt.format for key, prompt in self.prompts.items()}

    @property
    def providers(self) -> Dict[str, BasePromptProvider]:
        return {key: prompt.provider for key, prompt in self.prompts.items()}


def _resolve_file(prompt_id: str, filename: str):
    resolved_archive_file = cached_file(
        prompt_id,
        filename=filename,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
    )

    return resolved_archive_file


def _extract_default_embedding(model: PreTrainedModel) -> nn.Embedding:
    embedding = model.get_input_embeddings()
    if isinstance(embedding, BasePromptEmbedding):
        embedding = embedding.default_embeddings
    return embedding
