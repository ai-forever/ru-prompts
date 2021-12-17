from pathlib import Path

import pytest
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

CWD = Path(__file__).parent


@pytest.fixture
def config():
    return GPT2Config.from_pretrained(CWD / "fixtures")


@pytest.fixture
def model(config):
    return GPT2LMHeadModel(config)


@pytest.fixture
def tokenizer():
    return GPT2TokenizerFast.from_pretrained(CWD / "fixtures")
