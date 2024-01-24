Before installation make sure you have PyTorch installed, since **ruPrompts**' installer doesn't take your OS and compute platform into account and may install an unsuitable version of PyTorch. Refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) for details.

## Installation with pip

Use the following command to install only the library part of **ruPrompts**:
```sh
pip install ruprompts
```

To also install the `ruprompts-train` entrypoint, add the corresponding extra:
```sh
pip install ruprompts[hydra]
```

!!! info
    I you're using zsh, modify the command to escape the square brackets:
    ```sh
    pip install ruprompts\[hydra\]
    ```

## Installation from source

Installation from source may be your option if you wish to have direct access to `conf/` directory to modify [Hydra config](../hydra/config.md), e.g. to add custom tasks, datasets, etc.

In this case clone the repo and install the package in editable mode:
```sh
git clone https://github.com/ai-forever/ru-prompts
cd ru-prompts
pip install -e .[hydra]
```

### Poetry
Since **ruPrompts** is built with [Poetry](https://python-poetry.org), you may prefer to install it in virtual environment:
```sh
git clone https://github.com/ai-forever/ru-prompts
cd ru-prompts
pip install poetry
poetry install -E hydra
```

Since the `ruprompts-train` entrypoint willl also be installed to the virtualenv, it will be accessible with
```sh
poetry run ruprompts-train
```
or
```sh
poetry shell
ruprompts-train
```

Although this option may not be convenient for regular usage, it is preferable for development.
