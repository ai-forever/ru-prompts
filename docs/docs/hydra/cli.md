We provide a `ruprompts-train` endpoint, that implements the training loop described in [Walkthrough](../getting-started/walkthrough.md), parameterized by the config described in [Config Structure](config.md).

!!! info
    Note that the endpoint is not installed by default. To install it, you should add an extra:
    ```sh
    pip install ruprompts[hydra]
    ```

    See [Installation](../getting-started/installation.md) for details.

The endpoint behaves as a standard Hydra application, and its parameters should be passed in no hyphen format.

## Examples
Training a task defined in [Config Structure](config.md):
```sh
ruprompts-train \
    task=detoxification training.run_name=detox-tensor-linear-lr-1e-1 \
    prompt_provider=tensor training.learning_rate=1e-1 scheduler=linear_schedule_with_warmup
```

Defining task purely in command line:
```sh
ruprompts-train \
    task=text2text training.run_name=very-custom-run \
    task.task_name=detoxification \
    prompt_provider=tensor \
    +dataset=from_jsonl \
    +dataset.data_files.train=/path/to/train.jsonl \
    +dataset.data_files.validation=/path/to/validation.jsonl \
    prompt_format.template='"<P*60>{toxic}<P*20>"' \
    preprocessing.target_field=polite \
    preprocessing.truncation_field=toxic \
    preprocessing.max_tokens=1792
```
or with dataset from HF Hub:
```sh
ruprompts-train \
    task=text2text training.run_name=summarization-mlsum \
    task.task_name=summarization \
    prompt_provider=tensor \
    +dataset=default \
    +dataset.path=mlsum \
    +dataset.name=ru \
    prompt_format.template='"<P*60>{text}<P*20>"' \
    preprocessing.target_field=summary \
    preprocessing.truncation_field=text \
    preprocessing.max_tokens=1792
```


## Working directory behaviour

By default `ruprompts-train` creates a new working directory for each run, grouping them by task name and separating debug runs into the `debug` folder. Thus, for non-debug runs the workdir is switched to `./outputs/{task_name}/{datetime}` relatively to the directory where the endpoint was called, and to `./outputs/debug/{task_name}/{datetime}` for debug runs respectively.

The working directory contains all the logs and checkpoints. For example, loading a prompt after running `ruprompts-train` will be done with something like `Prompt.from_pretrained("./outputs/debug/detoxification/20211231_235959")`.

However, switching workdir in runtime makes using relative paths in configs (e.g. for data files) tricky and unreliable. For safety use only absolute paths.
