import string

import pytest
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from ruprompts.prompt_format import PromptFormat, _FormatStringWithRanges

variable_type = st.text(
    st.characters(whitelist_characters=string.ascii_letters + "_", whitelist_categories=()),
    min_size=1,
    max_size=1000,
)
string_type = st.text(st.characters(blacklist_characters="{}"), max_size=1000)


@st.composite
def string_and_dict(draw):
    format_dict = draw(st.dictionaries(keys=variable_type, values=string_type))
    length = len(format_dict)
    strings = draw(st.lists(string_type, min_size=length + 1, max_size=length + 1))

    format_str = strings[0]
    for idx, key in enumerate(format_dict.keys()):
        format_str += "{" + key + "}"
        format_str += strings[idx + 1]
    return format_str, format_dict


class TestFormatStringWithRanges:
    @given(string_and_dict())
    # @settings(max_examples=200, verbosity=Verbosity.debug)
    def test_result(self, string_dict):
        s, item = string_dict
        f = _FormatStringWithRanges(s)
        assert f.format(item)[0] == s.format(**item)

    @given(string_and_dict())
    # @settings(max_examples=200, verbosity=Verbosity.debug)
    def test_slices_unsafe(self, string_dict):
        s, item = string_dict
        f = _FormatStringWithRanges(s)
        res, slices = f.format(item)
        for key in item:
            assert res[slices[key]] == item[key]


class TestPromptFormat:
    @pytest.mark.xfail
    def test_initialize(self, tokenizer):
        pf = PromptFormat("<P><P>text<P>")
        pf.initialize(tokenizer)
        assert pf.compiled_template == "<|P|><|P|>text<|P|>"
        assert len(pf.init_tokens) == 0

        pf = PromptFormat("<P*5>text<P>")
        pf.initialize(tokenizer)
        assert pf.compiled_template == "<|P|><|P|><|P|><|P|><|P|>text<|P|>"
        assert len(pf.init_tokens) == 0

        # <P*3> -> <P><P><P>
        # <P>www</P> -> <P><P><P>
        pf = PromptFormat("<P><P*3><P><P>www</P></P>text</P>")
        pf.initialize(tokenizer)
        assert pf.compiled_template == "<|P|><|P|><|P|><|P|><|P|><|P|><|P|><|P|></P>text</P>"
        assert len(pf.init_tokens) == 3

    def test_set_key(self, tokenizer):
        pf = PromptFormat("<P*3>{a}")
        pf.initialize(tokenizer)
        assert pf(a="a") == "<|P|>" * 3 + "a"

        pf = PromptFormat("<P*3>{a}")
        pf.set_key("one")
        pf.initialize(tokenizer)
        assert pf(a="a") == "<|[one]P|>" * 3 + "a"

        pf = PromptFormat("<P*3>{a}")
        pf.initialize(tokenizer)
        pf.set_key("one")
        assert pf(a="a") == "<|[one]P|>" * 3 + "a"

        pf.set_key(None)
        assert pf(a="a") == "<|P|>" * 3 + "a"

    def test_prompt_length(self, tokenizer):
        pf = PromptFormat("<P*10>")
        pf.initialize(tokenizer)
        assert pf.prompt_length == 10

        pf.set_key("test")
        assert pf.prompt_length == 10

    def test_call(self, tokenizer):
        pf = PromptFormat("<P>{text}<P>{text2}<P>")
        pf.initialize(tokenizer)

        # kwargs
        assert pf(text="a", text2="b") == "<|P|>a<|P|>b<|P|>"

        # format
        assert pf({"text": "a", "text2": "b"}) == "<|P|>a<|P|>b<|P|>"
        assert pf(text="a", text2="b", return_ranges=True)[0] == "<|P|>a<|P|>b<|P|>"

        # batch_format
        assert pf([{"text": "a", "text2": "b"}, {"text": "c", "text2": "d"}]) == [
            "<|P|>a<|P|>b<|P|>",
            "<|P|>c<|P|>d<|P|>",
        ]
        assert pf([{"text": "a", "text2": "b"}, {"text": "c", "text2": "d"}], return_ranges=True)[
            0
        ] == [
            "<|P|>a<|P|>b<|P|>",
            "<|P|>c<|P|>d<|P|>",
        ]

        # auxiliary keys
        assert pf({"text": "a", "text2": "b", "aux": "aux"}) == "<|P|>a<|P|>b<|P|>"

        # missing keys
        with pytest.raises(KeyError):
            pf({"text": "a"})
