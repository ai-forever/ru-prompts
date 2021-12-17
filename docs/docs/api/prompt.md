Prompt is a core class that combines [prompt format](prompt_format.md) and [prompt provider](prompt_provider.md). It takes care of all the internal modifications that should be applied to model and tokenizer to insert the prompt provider and make it trainable. In particular, when you call :c:`ruprompts.prompt.Prompt.patch`, the following things happen:

1. Underlying [prompt format](prompt_format.md) is initialized using the tokenizer. At this step, the `<P>Initial manual prompt</P>` patterns are tokenized, the prompt format is compiled, and initialization tokens and their positions are identified to be passed to [prompt provider](prompt_provider.md).
2. The [prompt provider](prompt_provider.md) is initialized: it is given the initialization tokens from step 1 and after regular weight initialization overrides them with the embeddings corresponding to the passed intialization tokens.
3. The special tokens needed by [prompt format](prompt_format.md) are added to the `tokenizer`, and their ids are stored.
4. A special drop-in `torch.nn.Embedding` replacement is initialized with the [prompt provider](prompt_provider.md) and prompt token ids from step 3.
5. The smart embedding layer from step 4 replaces the default embedding layer in the `model`.

!!! note
    Steps 1 and 2 are skipped if the prompt was created with :c:`ruprompts.prompt.Prompt.from_pretrained`.

Prompt class also implements sharing methods:

- :c:`ruprompts.prompt.Prompt.save_pretrained` - saves the trained prompt to disk ot pushes it to HF Hub
- [`Prompt.push_to_hub`](https://huggingface.co/docs/transformers/main_classes/model#transformers.file_utils.PushToHubMixin.push_to_hub) - pushes the trained prompt to HF Hub
- :c:`ruprompts.prompt.Prompt.from_pretrained` - loads the prompt from disk or HF Hub

::: ruprompts.prompt.Prompt
    selection:
        members:
            - __init__
            - patch
            - save_pretrained
            - from_pretrained

::: ruprompts.prompt.MultiPrompt
