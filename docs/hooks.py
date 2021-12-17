import re

from markdown import Extension
from markdown.preprocessors import Preprocessor

REF_REGEX = re.compile("[:!](?P<mode>c|l|s)[:!]`(?P<code>[^`]*)`")


def short_handler(identifier):
    name = identifier.rpartition(".")[-1]
    return f"[`{name}`][{identifier}]"


def class_handler(identifier):
    name = ".".join(identifier.rsplit(".")[-2:])
    return f"[`{name}`][{identifier}]"


def long_handler(identifier):
    return f"[`{identifier}`][{identifier}]"


MODE_HANDLERS = {"s": short_handler, "c": class_handler, "l": long_handler}


def match_handler(match):
    handler = MODE_HANDLERS[match["mode"]]
    return handler(match["code"])


class ShortRefPreprocessor(Preprocessor):
    """Skip any line with words 'NO RENDER' in it."""

    def run(self, lines):
        new_lines = []
        for line in lines:
            new_line = REF_REGEX.sub(match_handler, line)
            new_lines.append(new_line)
        return new_lines


class ShortRefExtension(Extension):
    def extendMarkdown(self, md):
        md.registerExtension(self)
        snippet = ShortRefPreprocessor()
        md.preprocessors.register(snippet, "shortref", 32)


def on_config(config, *args, **kwargs):
    ext = ShortRefExtension()
    config["markdown_extensions"].append(ext)
    return config
