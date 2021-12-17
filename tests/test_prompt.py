import pytest
import torch
from transformers import PreTrainedModel

from ruprompts import Prompt, PromptFormat, TensorPromptProvider
from ruprompts.prompt import MultiPrompt


@pytest.fixture
def dummy_format():
    return PromptFormat(template="<P>{text}<P>")


@pytest.fixture
def dummy_format_2():
    return PromptFormat(template="<P><P>{text}<P>")


@pytest.fixture
def dummy_provider():
    return TensorPromptProvider()


@pytest.fixture
def dummy_provider_2():
    return TensorPromptProvider()


@pytest.fixture
def dummy_prompt(dummy_format, dummy_provider):
    return Prompt(format=dummy_format, provider=dummy_provider)


@pytest.fixture
def dummy_prompt_2(dummy_format_2, dummy_provider_2):
    return Prompt(format=dummy_format_2, provider=dummy_provider_2)


@pytest.fixture
def dummy_multiprompt(dummy_prompt, dummy_prompt_2):
    return MultiPrompt({"one": dummy_prompt, "two": dummy_prompt_2})


class TestPrompt:
    def test_initialize(self, dummy_prompt: Prompt, model, tokenizer):
        assert dummy_prompt.config.embedding_dim is None
        assert dummy_prompt.config.pretrained_model_name is None
        assert dummy_prompt.config.prompt_length is None
        assert not dummy_prompt.provider.is_initialized
        assert not dummy_prompt.format.is_initialized
        assert not dummy_prompt.is_initialized

        dummy_prompt.initialize(model, tokenizer)
        assert dummy_prompt.config.embedding_dim is not None
        assert dummy_prompt.config.pretrained_model_name is not None
        assert dummy_prompt.config.prompt_length is not None
        assert dummy_prompt.provider.is_initialized
        assert dummy_prompt.format.is_initialized
        assert dummy_prompt.is_initialized

    def test_attach(self, dummy_prompt: Prompt, model, tokenizer):
        assert dummy_prompt.is_initialized == False
        assert dummy_prompt.is_attached == False

        with pytest.raises(UserWarning):
            dummy_prompt.attach(model, tokenizer)

        assert dummy_prompt.is_initialized == False
        assert dummy_prompt.is_attached == False

        dummy_prompt.initialize(model, tokenizer)

        assert dummy_prompt.is_initialized == True
        assert dummy_prompt.is_attached == False

        dummy_prompt.attach(model, tokenizer)

        assert dummy_prompt.is_initialized == True
        assert dummy_prompt.is_attached == True

    def test_save_load_pretrained(self, dummy_prompt: Prompt, tmp_path, model, tokenizer):
        with pytest.raises(UserWarning):
            dummy_prompt.save_pretrained(tmp_path)
        dummy_prompt.initialize(model, tokenizer)
        dummy_prompt.save_pretrained(tmp_path)
        loaded_prompt = Prompt.from_pretrained(tmp_path)

        assert loaded_prompt is not dummy_prompt
        assert loaded_prompt.format is not dummy_prompt.format
        assert loaded_prompt.provider is not dummy_prompt.provider

        assert loaded_prompt.format.as_dict() == dummy_prompt.format.as_dict()
        for l_param, r_param in zip(
            loaded_prompt.provider.parameters(), dummy_prompt.provider.parameters()
        ):
            assert torch.all(l_param == r_param)

    def test_as_dict(self, dummy_prompt: Prompt, model, tokenizer):
        with pytest.raises(UserWarning):
            dummy_prompt.as_dict()
        dummy_prompt.initialize(model, tokenizer)
        dummy_prompt.as_dict()

    def test_forward(self, dummy_prompt: Prompt, model, tokenizer):
        dummy_prompt.patch(model, tokenizer)
        model(**tokenizer(dummy_prompt(text="text"), return_tensors="pt"))


class TestMultiPrompt:
    def test_add_prompt(self, dummy_prompt: Prompt, dummy_prompt_2: Prompt, model, tokenizer):
        mp = MultiPrompt()
        mp.add_prompt(key="one", prompt=dummy_prompt)
        mp.initialize(model, tokenizer)
        mp.add_prompt(key="two", prompt=dummy_prompt_2)
        with pytest.raises(UserWarning):
            mp.add_prompt(key="one", prompt=dummy_prompt_2)

    def test_initialize(self, dummy_multiprompt: MultiPrompt, model, tokenizer):
        dummy_multiprompt.initialize(model, tokenizer)

    def test_attach(self, dummy_multiprompt: MultiPrompt, model, tokenizer):
        with pytest.raises(UserWarning):
            dummy_multiprompt.attach(model, tokenizer)
        dummy_multiprompt.initialize(model, tokenizer)
        dummy_multiprompt.attach(model, tokenizer)

    def test_forward(self, dummy_multiprompt: MultiPrompt, model: PreTrainedModel, tokenizer):
        dummy_multiprompt.patch(model, tokenizer)
        out1 = model(**tokenizer(dummy_multiprompt(key="one", text="text"), return_tensors="pt"))
        out2 = model(**tokenizer(dummy_multiprompt(key="two", text="text"), return_tensors="pt"))
        assert out1.logits.shape != out2.logits.shape
