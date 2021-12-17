When you run `import ruprompts`, we add two custom pipelines to `transformers`:

- `text-generation-with-prompt`
- `text2text-generation-with-prompts`

These pipelines are then accessible with standard syntax:
```python
transformers.pipeline('text-generation-with-prompt', ...)
```

Read more about pipelines in [HF docs](https://huggingface.co/docs/transformers/main_classes/pipelines).

::: ruprompts.pipelines.TextGenerationWithPromptPipeline
    selection:
        members:
            - __init__

::: ruprompts.pipelines.Text2TextGenerationWithPromptPipeline
    selection:
        members:
            - __init__
