# ruPrompts

**ruPrompts** is a high-level yet extensible library for fast language model tuning via automatic prompt search, featuring integration with HuggingFace Hub, configuration system powered by Hydra, and command line interface.

Prompt is a text instruction for language model, like

```
Translate English to French:
cat =>
```

For some tasks the prompt is obvious, but for some it isn't. With **ruPrompts** you can define only the prompt format, like `<P*10>{text}<P*10>`, and train it automatically for any task, if you have a training dataset.

You can currently use **ruPrompts** for text-to-text tasks, such as summarization, detoxification, style transfer, etc., and for styled text generation, as a special case of text-to-text.

## Features

- **Modular structure** for convenient extensibility
- **Integration with [HF Transformers](https://huggingface.co/transformers/)**, support for all models with LM head
- **Integration with [HF Hub](https://huggingface.co/models/)** for sharing and loading pretrained prompts
- **CLI** and configuration system powered by **[Hydra](https://hydra.cc)**
- **[Pretrained prompts](https://ai-forever.github.io/ru-prompts/pretrained/)** for **[ruGPT-3](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2)**

## Installation

**ruPrompts** can be installed with `pip`:

```sh
pip install ruprompts[hydra]
```

See [Installation](https://ai-forever.github.io/ru-prompts/getting-started/installation) for other installation options.

## Usage

Loading a pretrained prompt for styled text generation:

```py
>>> import ruprompts
>>> from transformers import pipeline

>>> ppln_joke = pipeline("text-generation-with-prompt", prompt="konodyuk/prompt_rugpt3large_joke")
>>> ppln_joke("Говорит кружка ложке")
[{"generated_text": 'Говорит кружка ложке: "Не бойся, не утонешь!".'}]
```

For text2text tasks:

```py
>>> ppln_detox = pipeline("text2text-generation-with-prompt", prompt="konodyuk/prompt_rugpt3large_detox_russe")
>>> ppln_detox("Опять эти тупые дятлы все испортили, чтоб их черти взяли")
[{"generated_text": 'Опять эти люди все испортили'}]
```

Proceed to [Quick Start](https://ai-forever.github.io/ru-prompts/getting-started/quick-start) for a more detailed introduction or start using **ruPrompts** right now with our [Colab Tutorials](https://ai-forever.github.io/ru-prompts/tutorials).

## License

**ruPrompts** is Apache 2.0 licensed. See the LICENSE file for details.
