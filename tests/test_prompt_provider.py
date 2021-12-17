from contextlib import nullcontext

import pytest
import torch

from ruprompts.prompt_provider import BasePromptProvider, LSTMPromptProvider, TensorPromptProvider


@pytest.mark.parametrize("cls", [TensorPromptProvider, LSTMPromptProvider])
class TestPromptProvider:
    @pytest.mark.parametrize(
        "prompt_length,init_tokens",
        [(0, {}), (1, {}), (10, {}), (10, {0: 1, 1: 1, 2: 1, 8: 8, 9: 9}), (100, {})],
    )
    def test_initialize(self, cls, model, config, prompt_length, init_tokens):
        provider: BasePromptProvider = cls()

        context = nullcontext()
        if (cls is LSTMPromptProvider) and init_tokens:
            context = pytest.raises(ValueError)
        if (cls is TensorPromptProvider) and (prompt_length > config.vocab_size):
            context = pytest.raises(UserWarning)
        if prompt_length < 1:
            context = pytest.raises(ValueError)

        with context:
            provider.initialize(
                prompt_length=prompt_length,
                embedding_dim=config.n_embd,
                embedding=model.get_input_embeddings(),
                init_tokens=init_tokens,
            )

    @pytest.mark.parametrize(
        "prompt_length,init_tokens",
        [(10, {}), (10, {0: 1, 1: 1, 2: 1, 8: 8, 9: 9})],
    )
    def test_forward(self, cls, model, config, prompt_length, init_tokens):
        if (cls is LSTMPromptProvider) and init_tokens:
            pytest.skip("LSTMPromptProvider and init_tokens")

        provider: BasePromptProvider = cls()

        # commented out not to introduce overhead
        # with pytest.raises(UserWarning):
        #     out = provider()

        provider.initialize(
            prompt_length=prompt_length,
            embedding_dim=config.n_embd,
            embedding=model.get_input_embeddings(),
            init_tokens=init_tokens,
        )

        out = provider()
        assert out.size(0) == prompt_length, "invalid prompt length"
        assert out.size(1) == config.n_embd, "invalid embedding size"

    @pytest.mark.parametrize(
        "prompt_length,init_tokens",
        [(10, {}), (10, {0: 1, 1: 1, 2: 1, 8: 8, 9: 9})],
    )
    def test_save_load_pretrained(self, cls, model, config, prompt_length, init_tokens, tmp_path):
        if (cls is LSTMPromptProvider) and init_tokens:
            pytest.skip("LSTMPromptProvider and init_tokens")

        ckpt_path = tmp_path / "provider.bin"

        provider: BasePromptProvider = cls()

        provider.initialize(
            prompt_length=prompt_length,
            embedding_dim=config.n_embd,
            embedding=model.get_input_embeddings(),
            init_tokens=init_tokens,
        )

        provider.save_pretrained(ckpt_path)

        loaded_provider = TensorPromptProvider.from_pretrained(torch.load(ckpt_path))

        assert torch.all(provider() == loaded_provider())
