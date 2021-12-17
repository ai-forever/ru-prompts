[Hydra](https://hydra.cc) is a powerful configuration framework for Python applications, focused on modularity.

The generic config layout consists of multiple groups, each containing multiple options, and a root config, composing the resulting config object by choosing one option in each group:

```
conf/
    group1/
        option1.yaml
        option2.yaml
    group2/
        option1.yaml
        option2.yaml
    config.yaml
```

Example:

=== "composed config"
    ```yaml
    group1:
        field1: value1
        field2: value2
    group2:
        key: "option one in group two"
    ```

=== "command"
    ```sh
    python3 main.py
    ```

=== "config.yaml"
    ```yaml
    defaults:
        - group1: option2
        - group2: option1
    ```

=== "group1/option2.yaml"
    ```yaml
    field1: value1
    field2: value2
    ```

=== "group2/option1.yaml"
    ```yaml
    key: "option one in group two"
    ```

You can also override single parameters or the entire options on group level via command line arguments:


=== "composed config"
    ```yaml
    group1:
        field1: other value
        field2: value2
    group2:
        key: "option two in group two"
    ```

=== "command"
    ```sh
    python3 main.py group2=option2 group1.field1="other value"
    ```

=== "config.yaml"
    ```yaml
    defaults:
        - group1: option2
        - group2: option1
    ```

=== "group1/option2.yaml"
    ```yaml
    field1: value1
    field2: value2
    ```

=== "group2/option2.yaml"
    ```yaml
    key: "option two in group two"
    ```

Read [Hydra docs](https://hydra.cc/docs/intro/) for a more detailed introduction into its features.
