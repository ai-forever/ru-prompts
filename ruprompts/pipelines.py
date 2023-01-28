from typing import Dict, Union

from transformers.models.auto.modeling_auto import AutoModelWithLMHead
from transformers.pipelines import SUPPORTED_TASKS, TextGenerationPipeline

from ruprompts.prompt import Prompt


class PromptPipeline:
    default_generation_parameters = {}

    def __init__(self, prompt: Union[str, Prompt], *args, **kwargs):
        self.add_default_kwargs(kwargs)
        super().__init__(*args, **kwargs)

        self.load_prompt_patch_model(prompt)

        if self.device.type == "cuda":
            self.model = self.model.to(self.device)

    def load_prompt_patch_model(self, prompt: Union[str, Prompt]):
        if isinstance(prompt, str):
            prompt = Prompt.from_pretrained(prompt)

        self.prompt = prompt
        self.prompt.patch(self.model, self.tokenizer)

    def add_default_kwargs(self, kwargs):
        for key, default_value in self.default_generation_parameters.items():
            kwargs[key] = kwargs.get(key, default_value)


class TextGenerationWithPromptPipeline(PromptPipeline, TextGenerationPipeline):
    """Adds the trained prompt as prefix before passing text to TextGenerationPipeline.

    Alias: `transformers.pipeline('text-generation-with-prompt', ...)`.

    Args:
        prompt (# !s!`ruprompts.prompt.Prompt` or `str`): Prompt used to format input entries.
            If string is given, loads the prompt with :c:`ruprompts.prompt.Prompt.from_pretrained`.
        **kwargs: arguments for [`transformers.Pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.Pipeline)

    Examples:

        >>> from ruprompts import TextGenerationWithPromptPipeline, Prompt
        >>> prompt = Prompt.from_pretrained(...)
        >>> model = AutoLMHeadModel.from_pretrained(...)
        >>> ppln = TextGenerationWithPromptPipeline(prompt=prompt, model=model)

        >>> from transformers import pipeline
        >>> ppln = pipeline('text-generation-with-prompt', prompt=prompt, model=model)

        >>> ppln = pipeline('text-generation-with-prompt', prompt=prompt)

        >>> ppln = pipeline('text-generation-with-prompt', prompt='konodyuk/prompt_rugpt3large_joke')
        >>> a = ppln(text="Заходят в бар")
        >>> b = ppln("Заходят в бар")
        >>> assert a == b
    """

    default_generation_parameters = {
        "max_new_tokens": 100,
        "eos_token_id": 0,
        "return_full_text": True,
        "do_sample": True,
    }

    def preprocess(self, input_text: str, **kwargs):
        return super().preprocess(prefix=self.prompt(), prompt_text=input_text, **kwargs)


class Text2TextGenerationWithPromptPipeline(PromptPipeline, TextGenerationPipeline):
    """Formats text with the given prompt before passing it to TextGenerationPipeline.

    Alias: `transformers.pipeline('text2text-generation-with-prompt', ...)`.

    Args:
        prompt (# !s!`ruprompts.prompt.Prompt` or `str`): Prompt used to format input entries.
            If string is given, loads the prompt with :c:`ruprompts.prompt.Prompt.from_pretrained`.
        **kwargs: arguments for [`transformers.Pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.Pipeline)

    Examples:

        >>> from ruprompts import Text2TextGenerationWithPromptPipeline, Prompt
        >>> prompt = Prompt.from_pretrained(...)
        >>> model = AutoLMHeadModel.from_pretrained(...)
        >>> ppln = Text2TextGenerationWithPromptPipeline(prompt=prompt, model=model)

        >>> from transformers import pipeline
        >>> ppln = pipeline('text2text-generation-with-prompt', prompt=prompt, model=model)

        >>> ppln = pipeline('text2text-generation-with-prompt', prompt=prompt)

        >>> ppln = pipeline('text2text-generation-with-prompt', prompt='konodyuk/prompt_rugpt3large_qa_sberquad')

        >>> ppln = pipeline('text2text-generation-with-prompt', prompt='konodyuk/prompt_rugpt3large_qa_sberquad')
        >>> ppln(context="Трава зеленая.", question="Какая трава?")

        >>> ppln = pipeline('text2text-generation-with-prompt', prompt='konodyuk/prompt_rugpt3large_detox_russe')
        >>> a = ppln(text="Отвали, дурак")
        >>> b = ppln("Отвали, дурак")
        >>> assert a == b
    """

    default_generation_parameters = {
        "max_new_tokens": 100,
        "eos_token_id": 0,
        "return_full_text": False,
        "do_sample": True,
    }

    def preprocess(self, inputs: Union[str, Dict[str, str]], **kwargs):
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        input_text = self.prompt(**inputs)
        return super().preprocess(prompt_text=input_text, **kwargs)


SUPPORTED_TASKS["text2text-generation-with-prompt"] = {
    "impl": Text2TextGenerationWithPromptPipeline,
    "pt": (AutoModelWithLMHead,),
    "tf": (),
    "default": {
        "model": {
            "pt": "sberbank-ai/rugpt3large_based_on_gpt2",
        },
    },
}
SUPPORTED_TASKS["text-generation-with-prompt"] = {
    "impl": TextGenerationWithPromptPipeline,
    "pt": (AutoModelWithLMHead,),
    "tf": (),
    "default": {
        "model": {
            "pt": "sberbank-ai/rugpt3large_based_on_gpt2",
        },
    },
}
# SUPPORTED_TASKS["text-classification-with-prompt"] = {}
