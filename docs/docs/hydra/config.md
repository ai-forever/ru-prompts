!!! info
    If you are not familiar with Hydra, please read [our short introduction](index.md) or the [Hydra docs](https://hydra.cc/docs/intro/).

Our config is located in `conf/` folder and consists of the following groups:

### Backbone
**Path:** `conf/backbone`

**Default:** `rugpt3large`

**Description:** Defined the name of pretrained model and tokenizer.

#### Options
- **rugpt3small** - loads [`ai-forever/rugpt3small_based_on_gpt2`](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2).
- **rugpt3medium** - loads [`ai-forever/rugpt3medium_based_on_gpt2`](https://huggingface.co/ai-forever/rugpt3medium_based_on_gpt2).
- **rugpt3large** - loads [`ai-forever/rugpt3large_based_on_gpt2`](https://huggingface.co/ai-forever/rugpt3large_based_on_gpt2).

#### Option format
```yaml
pretrained_model_name_or_path: <string>
```


### Model
**Path:** `conf/model`

**Default:** `default`

**Description:** Creates a model.

#### Options
- **default** - loads an AutoLMHeadModel based on [backbone](#backbone) option.
- **gpt** - the same as **default**, but loads a GPT2LMHeadModel.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of pretrained model:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Tokenizer
**Path:** `conf/tokenizer`

**Default:** `autotokenizer`

**Description:** Creates a tokenizer.

#### Options
- **autotokenizer** - loads tokenizer based on [backbone](#backbone) option.
- **rugpt3** - the same as **autotokenizer**, but also adds missing special tokens.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of pretrained tokenizer:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Dataset
**Path:** `conf/dataset`

**Default:** `default`

**Description:** Loads a dataset dict containing at least `train` and `validation` datasets.

#### Options
- **default** - loads a dataset dict using [`datasets.load_dataset`](https://huggingface.co/docs/datasets/loading_datasets.html) function.
- **from_jsonl** - inherits from **default**, allows to load the dataset dict from json files.  
    Required fields: `data_files.train` and `data_files.validation`.  
    Usage example: `dataset=from_jsonl data_files.train=/path/to/train.jsonl data_files.validation=/path/to/validation.jsonl`.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of dataset dict:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Preprocessing
**Path:** `conf/preprocessing`

**Default:** `text2text`

**Description:** Returns an instance of [preprocessor](../api/preprocessing.md).

#### Options
- **text2text** - creates an instance of :s:`ruprompts.preprocessing.Text2TextPreprocessor`.  
    Required fields match those of target class.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of preprocessor:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Prompt Format
**Path:** `conf/prompt_format`

**Default:** `default`

**Description:** Defines the [prompt format](../api/prompt_format.md).

#### Options
- **default** - creates an instance of :s:`ruprompts.prompt_format.PromptFormat`.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of prompt format:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Prompt Provider
**Path:** `conf/prompt_provider`

**Default:** `tensor`

**Description:** Defines the [prompt provider](../api/prompt_provider.md).

#### Options
- **tensor** - creates an instance of :s:`ruprompts.prompt_provider.TensorPromptProvider`.
- **lstm** - creates an instance of :s:`ruprompts.prompt_provider.LSTMPromptProvider`.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of prompt provider:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Optimizer
**Path:** `conf/optimizer`

**Default:** `adamw`

**Description:** Defines the optimizer.

#### Options
- **adamw** - creates an instance of AdamW optimizer.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of torch optimizer:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Scheduler
**Path:** `conf/scheduler`

**Default:** `adamw`

**Description:** Defines the learning rate schedule.

#### Options
- **linear_schedule_with_warmup** - creates a [linear schedule](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup).
- **constant_schedule_with_warmup** - creates a [constant schedule](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup).

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of torch lr scheduler:
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Training arguments
**Path:** `conf/training`

**Default:** `default`

**Description:** Defines the [training arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

#### Options
- **default** - creates an instance of [`TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

#### Option format
No other options are assumed.


### Callbacks
**Path:** `conf/callbacks`

**Default:**
  - `freeze_transformer_unfreeze_prompt`
  - `reduce_checkpoint`
  - `save_pretrained_prompt`
  - `wb_log_hydra_config`

**Description:** Selects the [trainer callbacks](https://huggingface.co/docs/transformers/main_classes/callback).

#### Options
- **freeze_transformer_unfreeze_prompt** - creates an instance of :s:`ruprompts.callbacks.FreezeTransformerUnfreezePrompt`. Freezes the pretrained transformer and unfreezes the prompt provider before training.
- **reduce_checkpoint** - creates an instance of :s:`ruprompts.callbacks.ReduceCheckpoint`. After each saving reduces the size of saved model by removing all weights but those of prompt provider.
- **save_pretrained_prompt** - creates an instance of :s:`ruprompts.callbacks.SavePretrainedPrompt`. Saves the trained prompt using :c:`ruprompts.prompt.Prompt.save_pretrained` in each checkpoint.
- **wb_log_hydra_config** - creates an instance of :s:`ruprompts.callbacks.WBLogHydraConfig`. Logs the composed Hydra config to Weights and Biases before training.

#### Option format
An [instantiatable](https://hydra.cc/docs/advanced/instantiate_objects/overview/) config, returning an instance of[`TrainerCallback`](https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback):
```yaml
_target_: <module>.<callable>
arg1: value1
arg2: value2
```


### Task
**Path:** `conf/task`

**Default:** `default`

**Description:** Overrides the parameters of other groups.

#### Options
- **text2text** - selects `model=gpt`, `dataset=default` and `preprocessing=text2text`.
- other configs that inherit **text2text** - should define required group parameters in their bodies.

#### Option format
```yaml
task_name: detoxification

defaults:
  - text2text
  - /dataset: from_jsonl

dataset:
  data_files:
    train: /path/to/train.jsonl
    validation: /path/to/validation.jsonl

prompt_format:
  template: "<P*60>{toxic}<P*20>"

preprocessing:
  target_field: "polite"
  truncation_field: "toxic"
  max_tokens: 1792
```
