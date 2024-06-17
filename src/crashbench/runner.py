from typing import Any


def transform_metavar(name: str, value: Any):
    if isinstance(value, str) and " " in value:
        # wrap strings with spaces in them
        value = f'"{value}"'
    return f"-D{name}={value}"


class Runner: 
    ...
