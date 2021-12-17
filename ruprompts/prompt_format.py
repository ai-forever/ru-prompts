import abc
import re
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase

PROMPT_TOKEN = "<|P|>"
PROMPT_TOKEN_FOR_KEY = "<|[{key}]P|>"
PROMPT_TOKEN_SERIAL = "<|P[{serial}]|>"
PROMPT_TOKEN_SERIAL_FOR_KEY = "<|[{key}]P[{serial}]|>"
PROMPT_TOKEN_SHORT = "<P>"
PROMPT_TOKEN_REPEATED_REGEX = re.compile("<P\*(\d+)>")
PROMPT_LITERAL_REGEX = re.compile("<P>(?P<sequence>.+?)</P>")

# PERF: using @typechecked slows code down from 2.5us to 85us
# @typechecked
class BasePromptFormat(abc.ABC):
    """Base class for all prompt formats."""

    def __call__(
        self,
        items: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        return_ranges: bool = False,
        **kwargs,
    ) -> Union[
        str, Tuple[str, Dict[str, slice]], List[str], Tuple[List[str], List[Dict[str, slice]]]
    ]:
        """Applies prompt format to either one or multiple items.

        Takes a either one item or list of them, where item is a dictionary with string keys.
        Each item is then formatted into a single string, where the keys are inserted
        the same way as in format string. If `return_ranges=True`, also returns a dict of slices,
        for the value of each key in item containing its start and end positions in the resulting string.

        Examples:
            >>> f = PromptFormat("<P>{text}<P>")
            >>> item = {"text": "one two three", "other": "value"}
            >>> s, r = f(item, return_ranges=True)
            >>> assert s == "<P>one two three<P>"
            >>> assert s[r] == item["text"]

            >>> f(text="one two three", return_ranges=True)

            >>> f([{"text": "a"}, {"text": "b"}], return_ranges=True)

        Args:
            items: Item or list of items.
            return_ranges: Whether to return ranges.
            **kwargs: Can be used instead of `items` (see examples).

        # Returns:

        | Type                                       | Description                          | Condition                                                     |
        | ------------------------------------------ | ------------------------------------ | ------------------------------------------------------------- |
        | `str`                                      | formatted string                     | `items` is a `Dict[str, Any]` and `return_ranges=False`       |
        | `Tuple[str, Dict[str, slice]]`             | formatted string and ranges          | `items` is a `Dict[str, Any]` and `return_ranges=True`        |
        | `List[str]`                                | list of formatted strings            | `items` is a `List[Dict[str, Any]]` and `return_ranges=False` |
        | `Tuple[List[str], List[Dict[str, slice]]]` | list of formatted strings and ranges | `items` is a `List[Dict[str, Any]]` and `return_ranges=True`  |

        """
        if items is None:
            items = kwargs
        if isinstance(items, list):
            return self.batch_format(items, return_ranges)
        return self.format(items, return_ranges)

    @abc.abstractmethod
    def format(
        self, item: Dict[str, Any], return_ranges: bool = False
    ) -> Union[str, Tuple[str, Dict[str, slice]]]:
        """Formats one item into a string and possibly returns ranges.

        Args:
            item: Item to be formatted.
            return_ranges: Whether to return ranges.

        Returns:
            Union[str, Tuple[str, Dict[str, slice]]]: Returns `str` when `return_ranges=False`
                and `Tuple[str, Dict[str, slice]]` when `return_ranges=True`
        """

    def batch_format(
        self, items: List[Dict[str, Any]], return_ranges: bool = False
    ) -> Union[List[str], Tuple[List[str], List[Dict[str, slice]]]]:
        """Formats a list of items into strings and possibly returns ranges.

        Args:
            item: Items to be formatted.
            return_ranges: Whether to return ranges.

        Returns:
            Union[List[str], Tuple[List[str], List[Dict[str, slice]]]]: Returns `List[str]` when `return_ranges=False`
                and `Tuple[List[str], List[Dict[str, slice]]]` when `return_ranges=True`
        """

        result = [self.format(item, return_ranges=return_ranges) for item in items]
        if return_ranges:
            result, ranges = list(zip(*result))
            return list(result), list(ranges)
        return result

    @abc.abstractproperty
    def prompt_length(self) -> int:
        """Count of prompt tokens."""

    @abc.abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        """Serializes the prompt object as dict.

        Returns such a dict `d` that running `__init__(**d)`
        results in an identical object.
        """

    @abc.abstractmethod
    def initialize(self, tokenizer: PreTrainedTokenizerBase):
        pass

    @abc.abstractmethod
    def add_tokens(self, tokenizer: PreTrainedTokenizerBase) -> List[int]:
        pass

    @abc.abstractmethod
    def set_key(self, key: str):
        pass

    @abc.abstractproperty
    def is_initialized(self) -> bool:
        pass

    @abc.abstractproperty
    def prompt_tokens(self) -> List[str]:
        pass

    @abc.abstractproperty
    def prompt_token_ids(self) -> List[int]:
        pass

    @abc.abstractproperty
    def init_tokens() -> Dict[int, int]:
        pass


class PromptFormat(BasePromptFormat):
    """Arranges trainable tokens and dataset fields.

    ## Format patterns:
    - Repeated tokens:
        - Pattern: `<P*{int}>`
        - Example: `<P*3>`
        - Compiled example: `<P><P><P>`
    - Initialization from phrase:
        - Pattern: `<P>{str}</P>`
        - Example: `<P>Two tokens</P>`
        - Compiled example: `<P><P>`, prompt provider is initialized with embeddings of tokens `Two` and `tokens`

    Examples:
        >>> PromptProvider("<P*20>{text}<P*10>")
        >>> PromptProvider("<P>Passage:</P>{passage}<P>\\nQuestion:</P>{question}<P>\\nAnswer:</P>")

    See also:
        :c:`ruprompts.prompt_format.BasePromptFormat.__call__`
    """

    def __init__(
        self,
        template: str,
        compiled_template: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Args:
            template (str): See [format patterns][ruprompts.prompt_format.PromptFormat--format-patterns].
            compiled_template (Optional[str]): Compiled template.
            tokenizer (Optional[PreTrainedTokenizerBase]): Tokenizer to process the `<P>Text</P>` patterns.
        """

        self.template = template
        self._is_initialized = False

        if compiled_template is not None:
            self.compiled_template = compiled_template
            self._is_initialized = True

        if tokenizer is not None:
            self.initialize(tokenizer)

        self.formattable = None
        self._prompt_token_ids = []
        self._init_tokens = {}
        self.prompt_token = PROMPT_TOKEN
        self.key = None

    def format(
        self, item: Dict[str, Any], return_ranges: bool = False
    ) -> Union[str, Tuple[str, Dict[str, slice]]]:
        if not self.is_initialized:
            raise UserWarning("Applying a non-initialized prompt")
        if return_ranges:
            return self.formattable.format(item)
        return self.compiled_template.format(**item)

    def initialize(self, tokenizer: PreTrainedTokenizerBase):
        if not self.is_initialized:
            template = self.template
            template = self.build_repeated_tokens(template)
            template, init_tokens = self.build_initialized_tokens(template, tokenizer)
            template = self.build_regular_tokens(template)

            self.compiled_template = template
            self.formattable = _FormatStringWithRanges(template)
            self._init_tokens = init_tokens
            self._is_initialized = True

        if self.key is not None:
            self.set_key(self.key)

    def set_key(self, key: Optional[str] = None):
        self.key = key

        if not self.is_initialized:
            return

        old_token = self.prompt_token
        if key is None:
            new_token = PROMPT_TOKEN
        else:
            new_token = PROMPT_TOKEN_FOR_KEY.format(key=key)

        self.compiled_template = self.compiled_template.replace(old_token, new_token)
        self.prompt_token = new_token

    def add_tokens(self, tokenizer: PreTrainedTokenizerBase) -> List[int]:
        tokenizer.add_tokens([self.prompt_token])
        prompt_token_id = tokenizer.get_added_vocab()[self.prompt_token]
        self._prompt_token_ids = [prompt_token_id]
        return self._prompt_token_ids

    @property
    def prompt_tokens(self) -> List[str]:
        return [self.prompt_token]

    @property
    def prompt_token_ids(self) -> List[int]:
        return self._prompt_token_ids

    @staticmethod
    def build_repeated_tokens(template: str, **kwargs):
        while True:
            match = PROMPT_TOKEN_REPEATED_REGEX.search(template)
            if not match:
                break
            n_repeats = int(match.group(1))
            template = template.replace(match.group(), PROMPT_TOKEN * n_repeats)
        return template

    @staticmethod
    def build_initialized_tokens(
        template: str, tokenizer: PreTrainedTokenizerBase, **kwargs
    ) -> Tuple[str, Dict[int, int]]:
        init_tokens = {}

        while True:
            match = PROMPT_LITERAL_REGEX.search(template)
            if not match:
                break
            sequence = match["sequence"]

            tokenizer_output = tokenizer(sequence)
            input_ids = tokenizer_output["input_ids"]

            preceding_template = template[: match.start()]
            preceding_prompt_tokens = preceding_template.count(
                PROMPT_TOKEN
            ) + preceding_template.count(PROMPT_TOKEN_SHORT)

            for idx, token_id in enumerate(input_ids):
                init_tokens[idx + preceding_prompt_tokens] = token_id

            template = template.replace(match.group(), PROMPT_TOKEN * len(input_ids))

        return template, init_tokens

    @staticmethod
    def build_regular_tokens(template: str, **kwargs):
        return template.replace(PROMPT_TOKEN_SHORT, PROMPT_TOKEN)

    @property
    def prompt_length(self) -> Optional[int]:
        if not self.is_initialized:
            return None
        return self.compiled_template.count(self.prompt_token)

    def as_dict(self) -> Dict[str, Any]:
        if not self.is_initialized:
            raise UserWarning("PromptFormat should be initialized before serialization")
        return {"template": self.template, "compiled_template": self.compiled_template}

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def init_tokens(self) -> Optional[Dict[int, int]]:
        if not self.is_initialized:
            return None
        return self._init_tokens


class PromptFormatSafe(BasePromptFormat):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


class _FormatStringWithRanges:
    """A drop-in replacement for `str.format` returning ranges for each inserted key.

    Works only 2-4 times slower than `str.format` depending on string lengths.

    Example:
        >>> # Benchmarking procedure:
        >>> s = "<P>{a}<P>{b}<P>"
        >>> d = {"a": "a" * 1000, "b": "b" * 1000}
        >>> %timeit s.format(**d);
        >>> f = _FormatStringWithRanges(s)
        >>> %timeit f.format(d)
    """

    def __init__(self, template: str):
        self.template = template
        self.format_names = re.findall("{\s*(\w*)\s*}", self.template)
        interpolation_regex = "{\s*(" + "|".join(self.format_names) + ")\s*}"
        self.format_list = re.split(interpolation_regex, self.template)

    def format(self, item: Dict[str, Any]) -> Tuple[str, Dict[str, slice]]:
        format_list = copy(self.format_list)
        ranges = {}
        current_length = 0
        for position in range(1, len(format_list), 2):
            current_length += len(format_list[position - 1])
            interpolation_name = format_list[position]
            interpolation_value = item.get(interpolation_name, None)
            if interpolation_value is None:
                raise KeyError(
                    f"Missing interpolation value for key {interpolation_name} while formatting {self.template}"
                )
            format_list[position] = interpolation_value
            ranges[interpolation_name] = slice(
                current_length, current_length + len(interpolation_value)
            )
            current_length += len(interpolation_value)
        return "".join(format_list), ranges
