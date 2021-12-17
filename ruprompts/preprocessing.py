import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from ruprompts.prompt_format import BasePromptFormat


class BasePreprocessor(abc.ABC):
    def __call__(
        self, items: Union[Dict[str, Any], Dict[str, List[Any]]]
    ) -> Union[Dict[str, Any], Dict[str, List[Any]]]:
        if isinstance(items, list):
            return self.batch_process(items)
        return self.process(items)

    @abc.abstractmethod
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def batch_process(self, item: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def collate_fn(self) -> Callable:
        pass


class TruncationMixin:
    @property
    def is_truncation_enabled(self):
        return self.truncation_field is not None and self.max_tokens is not None

    def truncate(self, encoding: BatchEncoding, ranges: Dict[str, slice]) -> int:
        if len(encoding["input_ids"]) <= self.max_tokens or not self.is_truncation_enabled:
            return 0

        truncated_field_range = ranges[self.truncation_field]
        truncated_field_start = encoding.char_to_token(truncated_field_range.start)
        truncated_field_end = encoding.char_to_token(truncated_field_range.stop)

        exceeding_tokens = len(encoding["input_ids"]) - self.max_tokens

        cut_start = max(truncated_field_end - exceeding_tokens, truncated_field_start)
        cut_end = truncated_field_end

        encoding["input_ids"] = encoding["input_ids"][:cut_start] + encoding["input_ids"][cut_end:]
        encoding["attention_mask"] = (
            encoding["attention_mask"][:cut_start] + encoding["attention_mask"][cut_end:]
        )

        return exceeding_tokens

    def batch_truncate(self, encoding: BatchEncoding, ranges: List[Dict[str, slice]]):
        pass


@dataclass
class Text2TextPreprocessor(BasePreprocessor, TruncationMixin):
    """Carries out preprocessing for text2text tasks.

    Applies prompt format, appends target sequence, tokenizes
    and truncates each dataset item.

    Examples:
        >>> prompt_format = PromptFormat("<P*20>{text}<P*10>")
        >>> preprocessor = Text2TextPreprocessor(
        ...     prompt_format=prompt_format,
        ...     tokenizer=tokenizer,
        ...     target_field="summary",
        ...     max_tokens=1024,
        ...     truncation_field="text"
        ... )
        >>> dataset = dataset.map(preprocessor)
        >>> Trainer(..., train_dataset=dataset, ...)

    Args:
        prompt_format (# !s!`ruprompts.prompt_format.BasePromptFormat`):
            Prompt format to be applied to dataset items.
        tokenizer (PreTrainedTokenizerBase):
        target_field (str): Target dataset field.
        max_tokens (Optional[int]): Max sequence length in tokens.
        truncation_field (Optional[str]):
            Field to be truncated when sequence length exceeds `max_tokens`.
    """

    prompt_format: BasePromptFormat
    tokenizer: PreTrainedTokenizerBase
    target_field: str
    max_tokens: Optional[int] = None
    truncation_field: Optional[str] = None

    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_truncation_enabled:
            formatted_text, ranges = self.prompt_format(item, return_ranges=True)
        else:
            formatted_text = self.prompt_format(item)
        target_sequence = item[self.target_field]

        result = self.tokenizer(formatted_text + target_sequence + self.tokenizer.eos_token)

        if self.is_truncation_enabled:
            truncated_tokens = self.truncate(result, ranges)
        else:
            truncated_tokens = 0

        target_sequence_start = result.char_to_token(len(formatted_text))
        if self.is_truncation_enabled:
            target_sequence_start -= truncated_tokens

        labels = [-100] * len(result["input_ids"])
        labels[target_sequence_start:] = result["input_ids"][target_sequence_start:]
        result["labels"] = labels

        return result

    def batch_process(self, item: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return super().batch_process(item)

    def collate_fn(self, **kwargs) -> Callable:
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            return_tensors="pt",
            **kwargs,
        )
