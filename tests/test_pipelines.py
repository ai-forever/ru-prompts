import pytest
from transformers import pipeline

from ruprompts.pipelines import (
    Text2TextGenerationWithPromptPipeline,
    TextGenerationWithPromptPipeline,
)
from ruprompts.prompt import Prompt
from ruprompts.prompt_format import PromptFormat
from ruprompts.prompt_provider import TensorPromptProvider


@pytest.fixture
def prompt(model, tokenizer):
    pf = PromptFormat("<P>{text}<P>")
    pp = TensorPromptProvider()

    p = Prompt(format=pf, provider=pp)
    p.patch(model, tokenizer)

    return p


def test_text_generation(model, tokenizer):
    pf = PromptFormat("<P*5>")
    pp = TensorPromptProvider()

    p = Prompt(format=pf, provider=pp)
    p.patch(model, tokenizer)
    pipe = TextGenerationWithPromptPipeline(prompt=p, model=model, tokenizer=tokenizer)
    assert pipe("Text")[0]["generated_text"]

    pipe = pipeline("text-generation-with-prompt", prompt=p, model=model, tokenizer=tokenizer)
    assert pipe("Text")[0]["generated_text"]


def test_text2text(prompt, model, tokenizer):
    pipe = Text2TextGenerationWithPromptPipeline(prompt=prompt, model=model, tokenizer=tokenizer)
    assert pipe("Text")[0]["generated_text"]
    assert pipe({"text": "Text"})[0]["generated_text"]

    pipe = pipeline(
        "text2text-generation-with-prompt", prompt=prompt, model=model, tokenizer=tokenizer
    )
    assert pipe("Text")[0]["generated_text"]
    assert pipe({"text": "Text"})[0]["generated_text"]
