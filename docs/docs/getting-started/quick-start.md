## Installation

Install **ruPrompts** as follows:
```sh
pip install ruprompts
```

!!! note
    Make sure you have the right version of torch. If you don't have it installed when running the above command,
    an incorrect (CPU/GPU) version may be installed.

For more advanced installation options see [Installation](getting-started/installation).

## Loading a pretrained prompt

Let's download a prompt for joke generation with ruGPT-3 Large:

```python
>>> import ruprompts
>>> from transformers import pipeline

>>> ppln_joke = pipeline("text-generation-with-prompt", prompt="konodyuk/prompt_rugpt3large_joke")
>>> ppln_joke("Говорит кружка ложке")
[{"generated_text": 'Говорит кружка ложке: "Не бойся, не утонешь!".'}]
```

!!! tip
    When the model and tokenizer are not specified, they are inferred from prompt config and loaded automatically.

That's it! Prompts can also handle text-to-text tasks with more complex structure, for example question answering prompt takes two keyword arguments:

```python
>>> ppln_qa = pipeline('text2text-generation-with-prompt', prompt='konodyuk/prompt_rugpt3large_qa_sberquad')
>>> ppln_qa(context="Трава зеленая.", question="Какая трава?")
[{"generated_text": 'зеленая'}]
```

If you run these code snippets, you'll notice that a separate model is created each time. We can reuse models and tokenizers by passing them as pipeline arguments:

```python
from transformers import GPT2LMHeadModel, AutoTokenizer

model_id = "ai-forever/rugpt3large_based_on_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

ppln_joke = pipeline(
    "text-generation-with-prompt",
    prompt="konodyuk/prompt_rugpt3large_joke",
    model=model, tokenizer=tokenizer)
ppln_joke(...)

ppln_qa = pipeline(
    "text-generation-with-prompt",
    prompt="konodyuk/prompt_rugpt3large_qa_sberquad",
    model=model, tokenizer=tokenizer)
ppln_qa(...)
```

!!! note
    This approach still isn't suitable if you want to use multiple prompts simultaneously. See the :s:`ruprompts.prompt.MultiPrompt` for this purpose.

Inference API is very simple and doesn't require any understanding of the internals, but if you wish to learn how it works in theory, then proceed to [How it works](how-it-works.md). For a more practical introduction read [Walkthrough](walkthrough.md) first.

## Contents

The documentation is organized in the following parts:

-   **Getting Started** contains installation instructions, simple introduction to practical usage: inference and training, and a theoretical introduction, explaining the essence of the underlying technique
<!-- -   **User Guide** covers the most frequent use cases in more detail -->
-   **Python API** describes the public classes and their API
-   **Hydra API** contains command line reference and configuration schema, as well as brief introduction to Hydra
