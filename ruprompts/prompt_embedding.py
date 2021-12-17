import abc
import sys
from typing import List, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from ruprompts.prompt_provider import BasePromptProvider

PROMPT_PROVIDER_KEY_NAME = "prompt_provider"


class BasePromptEmbedding(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self, input_ids: TensorType["batch", "sequence"]
    ) -> TensorType["batch", "sequence", "embedding"]:
        pass

    @abc.abstractproperty
    def default_embeddings(self) -> nn.Embedding:
        pass

    @abc.abstractmethod
    def check_contains_prompt_token_ids(self, input_ids: TensorType["batch", "sequence"]) -> bool:
        pass


class PromptEmbedding(BasePromptEmbedding):
    def __init__(
        self,
        embedding: nn.Embedding,
        prompt_provider: BasePromptProvider,
        prompt_token_ids: List[int],
    ):
        super().__init__()
        self.embedding = embedding
        self.prompt_provider = prompt_provider
        self.prompt_token_id = self._validate_prompt_token_ids(prompt_token_ids)

    def forward(
        self, input_ids: TensorType["batch", "sequence"]
    ) -> TensorType["batch", "sequence", "embedding"]:
        batch_size = input_ids.shape[0]

        # (0) get prompt embeddings:
        # [P1, P2, P3, P4]
        prompt_embeddings = self.prompt_provider()

        # (1) select prompt tokens:
        #  <P> <P> What is it? <P> <P>   <- tokens
        #  [58, 58, 2,  3,  1, 58, 58]   <- token_ids
        #  ^    ^              ^   ^     <- prompt_tokens_mask
        prompt_tokens_mask = input_ids == self.prompt_token_id

        # (2) set them to zero since default embedding doesn't contain the prompt token:
        # [0, 0, 2, 3, 1, 0, 0]
        input_ids = input_ids.clone()
        input_ids[prompt_tokens_mask] = 0

        # (3) get embeddings for all ids:
        # [E, E, E, E, E, E, E]
        output_embeddings = self.embedding(input_ids)

        # check there is enough positions for prompt_embeddings
        if prompt_tokens_mask.sum() == prompt_embeddings.shape[0] * batch_size:

            # (4) inject prompt embeddings to the positions selected in (1):
            # [P1, P2, E, E, E, P3, P4]
            output_embeddings[prompt_tokens_mask] = prompt_embeddings.repeat(batch_size, 1)
            pass
        else:
            if prompt_tokens_mask.sum() != 0:
                print(
                    "Invalid prompt token count:",
                    prompt_tokens_mask.sum() / batch_size,
                    file=sys.stderr,
                )

        return output_embeddings

    def _validate_prompt_token_ids(self, ids: List[int]) -> int:
        if len(ids) != 1:
            raise ValueError(
                "PromptEmbedding expects prompt_token_ids to contain a single index, "
                "but received:",
                ids,
            )
        return ids[0]

    @property
    def default_embeddings(self) -> nn.Embedding:
        return self.embedding

    def check_contains_prompt_token_ids(self, input_ids: TensorType["batch", "sequence"]) -> bool:
        return torch.any(input_ids == self.prompt_token_id)


class PromptEmbeddingSafe(BasePromptEmbedding):
    def __init__(
        self,
        embedding: nn.Embedding,
        prompt_provider: BasePromptProvider,
        prompt_token_ids: List[int],
    ):
        raise NotImplementedError()
        super().__init__()
        self.embedding = embedding
        self.prompt_provider = prompt_provider

        self.prompt_token_min, self.prompt_token_max = self._validate_prompt_token_ids(
            prompt_token_ids
        )

    def _validate_prompt_token_ids(self, ids: List[int]) -> Tuple[int, int]:
        id_min = min(ids)
        id_max = max(ids)
        ids_sequential = range(id_min, id_max + 1)
        if set(ids) != set(ids_sequential):
            raise ValueError(
                "PromptEmbeddingSafe expects prompt_token_ids to be sequential, "
                "but the following indices are missing:",
                set(ids) - set(ids_sequential),
            )
        return id_min, id_max

    def check_contains_prompt_token_ids(self, input_ids: TensorType["batch", "sequence"]) -> bool:
        return torch.any((self.prompt_token_min <= input_ids) & (input_ids < self.prompt_token_max))


class MultiPromptEmbedding(BasePromptEmbedding):
    def __init__(
        self,
        default_embedding: nn.Embedding,
        embeddings: List[BasePromptEmbedding],
    ):
        super().__init__()
        self._default_embeddings = default_embedding
        self.embeddings = nn.ModuleList(embeddings)

    def forward(
        self, input_ids: TensorType["batch", "sequence"]
    ) -> TensorType["batch", "sequence", "embedding"]:
        for embedding in self.embeddings:
            if embedding.check_contains_prompt_token_ids(input_ids):
                return embedding(input_ids)
        return self.default_embeddings(input_ids)

    @property
    def default_embeddings(self) -> nn.Embedding:
        return self._default_embeddings

    def check_contains_prompt_token_ids(self, input_ids: TensorType["batch", "sequence"]) -> bool:
        for embedding in self.embeddings:
            if embedding.check_contains_prompt_token_ids(input_ids):
                return True
        return False
