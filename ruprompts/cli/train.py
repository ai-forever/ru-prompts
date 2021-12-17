try:
    # not checking `omegaconf`, since it is a dependency of `hydra`
    import hydra
except ImportError:
    raise UserWarning(
        "`ruprompts-train` entrypoint is not available. "
        "Install `hydra` with `pip install ruprompts[hydra]` or `pip install hydra-core`."
    )

import os
from typing import Optional, Union

from datasets import DatasetDict
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ruprompts.preprocessing import BasePreprocessor
from ruprompts.prompt import Prompt
from ruprompts.prompt_format import BasePromptFormat
from ruprompts.prompt_provider import BasePromptProvider

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Namespace(dict):
    def __getattr__(self, key: str):
        return self[key]

    def instantiate(
        self, instance_cfg: Union[DictConfig, dict], cfg_name: Optional[str] = None, **kwargs
    ):
        interp_kwargs = {}
        for key in instance_cfg.keys():
            if OmegaConf.is_missing(instance_cfg, key):
                if key in self:
                    interp_kwargs[key] = self[key]

            if isinstance(instance_cfg.get(key, None), ListConfig):
                instance_cfg[key] = {
                    "_target_": "builtins.list",
                    "_args_": [instance_cfg[key]],
                }

        return instantiate(instance_cfg, **interp_kwargs, **kwargs)

    def create(self, cfg_name: str, ns_name: Optional[str] = None, **kwargs):
        if ns_name is None:
            ns_name = cfg_name

        obj_config: DictConfig = self.cfg[cfg_name]

        self[ns_name] = self.instantiate(obj_config, cfg_name=cfg_name, **kwargs)
        return self[ns_name]


def run(cfg: DictConfig):
    ns = Namespace()
    ns["cfg"] = cfg

    set_seed(cfg.get("training").get("seed"))

    dataset_dict: DatasetDict = ns.create("dataset")
    model: PreTrainedModel = ns.create("model")
    tokenizer: PreTrainedTokenizerBase = ns.create("tokenizer")
    prompt_format: BasePromptFormat = ns.create("prompt_format")
    prompt_provider: BasePromptProvider = ns.create("prompt_provider")
    preprocessor: BasePreprocessor = ns.create("preprocessing")

    prompt = Prompt(format=prompt_format, provider=prompt_provider)
    prompt.patch(model=model, tokenizer=tokenizer)
    ns["prompt"] = prompt

    callbacks = list(cfg.get("callbacks").values())
    for i, callback in enumerate(callbacks):
        callbacks[i] = ns.instantiate(callback)

    for key in ["train", "validation"]:
        dataset_dict[key] = dataset_dict[key].map(preprocessor)

    train_dataset = dataset_dict["train"]
    valid_dataset = dataset_dict["validation"]

    training_args: TrainingArguments = ns.create("training")

    optimizer = ns.create("optimizer", params=prompt_provider.parameters())
    if cfg.get("scheduler"):
        scheduler = ns.create("scheduler")
    else:
        scheduler = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=preprocessor.collate_fn(),
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run(cfg)


if __name__ == "__main__":
    hydra_entry()
