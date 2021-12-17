import os

import torch
from transformers import TrainerControl, TrainerState
from transformers.file_utils import WEIGHTS_NAME
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import TrainingArguments

from ruprompts.prompt import Prompt
from ruprompts.prompt_embedding import PROMPT_PROVIDER_KEY_NAME

try:
    import omegaconf

    IS_OMEGACONF_AVAILABLE = True
except ImportError:
    omegaconf = None
    IS_OMEGACONF_AVAILABLE = False

try:
    import wandb

    IS_WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    IS_WANDB_AVAILABLE = False


class FreezeTransformerUnfreezePrompt(TrainerCallback):
    """Freezes all parameters but those of prompt provider."""

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        **kwargs,
    ):
        for name, param in model.transformer.named_parameters():
            if PROMPT_PROVIDER_KEY_NAME in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


class ReduceCheckpoint(TrainerCallback):
    """Reduces the checkpoint size by keeping only the weights of prompt provider."""

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        weights_path = os.path.join(output_dir, WEIGHTS_NAME)
        weights = torch.load(weights_path)

        keys_to_remove = []
        for weight_key in weights:
            if PROMPT_PROVIDER_KEY_NAME not in weight_key:
                keys_to_remove.append(weight_key)

        for key in keys_to_remove:
            weights.pop(key)
        torch.save(weights, weights_path)


class SavePretrainedPrompt(TrainerCallback):
    """Saves the prompt as pretrained on checkpoint.

    Args:
        prompt (# !s!`ruprompts.prompt.Prompt`): Prompt instance to be saved.
    """

    def __init__(self, prompt: Prompt):
        self.prompt = prompt

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        self.prompt.save_pretrained(output_dir)


class WBLogHydraConfig(TrainerCallback):
    """Logs Hydra config to Weights and Biases on training start.

    Args:
        cfg (omegaconf.DictConfig): Config to be logged.
    """

    def __init__(self, cfg):
        if not (IS_OMEGACONF_AVAILABLE and IS_WANDB_AVAILABLE):
            raise UserWarning(
                "WBLogHydraConfig is not available. Install `hydra` and `wandb` "
                "with `pip install hydra-core wandb`."
            )

        self.cfg = cfg

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        wandb.config.update({"hydra": omegaconf.OmegaConf.to_container(self.cfg)})
