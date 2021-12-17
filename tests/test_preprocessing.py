import numpy as np
import pytest

from ruprompts.preprocessing import Text2TextPreprocessor
from ruprompts.prompt_format import PromptFormat


@pytest.fixture
def format(tokenizer):
    result = PromptFormat("<P>{text}<P>")
    result.initialize(tokenizer)
    result.add_tokens(tokenizer)
    return result


class TestText2TextPreprocessor:
    def test_process(self, tokenizer, format):
        p = Text2TextPreprocessor(prompt_format=format, tokenizer=tokenizer, target_field="target")
        encoding = p({"text": "one", "target": "two"})
        golden_encoding = tokenizer("<|P|>one<|P|>two<|endoftext|>")
        assert encoding["input_ids"] == golden_encoding["input_ids"]

        labels_mask = np.array(encoding["labels"]) != -100
        assert np.all(
            np.array(encoding["labels"])[labels_mask]
            == np.array(golden_encoding["input_ids"])[labels_mask]
        )

    def test_truncation(self, tokenizer, format):
        p = Text2TextPreprocessor(
            prompt_format=format,
            tokenizer=tokenizer,
            target_field="target",
            max_tokens=20,
            truncation_field="text",
        )
        encoding = p({"text": "w" * 100, "target": ""})
        assert len(encoding["input_ids"]) == 20
        assert encoding["input_ids"][-1] == tokenizer.eos_token_id

        prompt_token_id = tokenizer.get_added_vocab()["<|P|>"]
        assert encoding["input_ids"][0] == prompt_token_id
        assert encoding["input_ids"][-2] == prompt_token_id

        encoding = p({"text": "w" * 100, "target": "ooo"})
        assert len(encoding["input_ids"]) == 20
        assert encoding["input_ids"][-1] == tokenizer.eos_token_id

        label_ids = tokenizer("ooo<|endoftext|>")["input_ids"]
        assert encoding["labels"][-4:] == label_ids

    @pytest.mark.xfail
    def test_call(self, tokenizer, format):
        p = Text2TextPreprocessor(prompt_format=format, tokenizer=tokenizer, target_field="target")
        p({"text": "a", "target": "b"})
        p([{"text": "a", "target": "b"}, {"text": "a", "target": "b"}])
