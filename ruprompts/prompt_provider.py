import abc
from typing import Dict

import torch
import torch.nn as nn
from torchtyping import TensorType
from typeguard import typechecked
from typing_extensions import Literal


class BasePromptProvider(nn.Module, abc.ABC):
    """Base class for all prompt providers."""

    @abc.abstractmethod
    def initialize(
        self,
        prompt_length: int,
        embedding_dim: int,
        embedding: nn.Embedding,
        init_tokens: Dict[int, int],  # dict: index -> token id
    ):
        pass

    @abc.abstractmethod
    def forward(self) -> TensorType["prompt_length", "embedding"]:
        pass

    @abc.abstractproperty
    def is_initialized(self) -> bool:
        pass

    def save_pretrained(self, output_file: str):
        if not self.is_initialized:
            raise UserWarning(f"{self.__class__.__name__} should be initialized before saving")
        tensor = self().detach().cpu()
        torch.save(tensor, output_file)


class TensorPromptProvider(BasePromptProvider):
    """Directly stores prompt embeddings as a tensor."""

    @typechecked
    def __init__(self, init: Literal["random", "vocab"] = "vocab"):
        """
        Args:
            init: Initialization mode. Initializes embbeddings from random embeddings
                from vocabulary when set to `vocab`, randomly otherwise.
        """
        super().__init__()
        self.init = init

        self.weights = None

    def initialize(
        self,
        prompt_length: int,
        embedding_dim: int,
        embedding: nn.Embedding,
        init_tokens: Dict[int, int],
    ):
        if self.is_initialized:
            return

        if prompt_length < 1:
            raise ValueError("prompt_length should be >= 1")

        if prompt_length > embedding.num_embeddings:
            raise UserWarning(
                f"prompt_length ({prompt_length}) cannot be greater "
                f"than vocab size ({embedding.num_embeddings})"
            )

        weights = torch.empty((prompt_length, embedding_dim), dtype=embedding.weight.dtype)
        if self.init == "random":
            weights.normal_()
        elif self.init == "vocab":
            weights = embedding(torch.randperm(embedding.num_embeddings)[:prompt_length])

        if init_tokens:
            init_indices = torch.tensor(list(init_tokens.keys()))
            init_token_ids = torch.tensor(list(init_tokens.values()))
            weights[init_indices] = embedding(init_token_ids)

        self.weights = nn.Parameter(weights)

    def forward(self) -> TensorType["prompt_length", "embedding"]:
        return self.weights

    @property
    def is_initialized(self) -> bool:
        return self.weights is not None

    @classmethod
    def from_pretrained(cls, weights) -> "TensorPromptProvider":
        out = cls()
        out.weights = nn.Parameter(weights)
        return out


class LSTMPromptProvider(BasePromptProvider):
    """Generates prompt embeddings from LSTM and MLP."""

    @typechecked
    def __init__(self, hidden_dim: int = -1, input_dim: int = -1, num_lstm_layers: int = 2):
        """
        Args:
            hidden_dim: Hidden dim of LSTM. Defaults to embedding dim of backbone when set to -1.
            input_dim: Input dim of LSTM. Defaults to embedding dim of backbone when set to -1.
            num_lstm_layers: Number of LSTM layers.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_lstm_layers = num_lstm_layers

        self.input = None
        self.lstm = None
        self.mlp = None

    def initialize(
        self,
        prompt_length: int,
        embedding_dim: int,
        embedding: nn.Embedding,
        init_tokens: Dict[int, int],
    ):
        if self.is_initialized:
            return

        if prompt_length < 1:
            raise ValueError("prompt_length should be >= 1")

        if init_tokens:
            raise ValueError("LSTMPromptProvider doesn't support initialization from tokens")

        if self.hidden_dim == -1:
            self.hidden_dim = embedding_dim

        if self.input_dim == -1:
            self.input_dim = embedding_dim

        self.input = nn.Parameter(torch.rand(1, prompt_length, self.input_dim))
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

    def forward(self) -> TensorType["prompt_length", "embedding"]:
        # size = (1, prompt_length, embedding_dim)
        x = self.input

        # size = (1, prompt_length, 2 * hidden_dim)
        x = self.lstm(x)[0]

        # size = (1, prompt_length, embedding_dim)
        x = self.mlp(x)

        # size = (prompt_length, embedding_dim)
        x = x.squeeze(0)
        return x

    @property
    def is_initialized(self) -> bool:
        return self.lstm is not None
