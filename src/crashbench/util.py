import contextlib
from dataclasses import asdict
from datetime import datetime
import json
from typing import Any


def json_default(thing):
    with contextlib.suppress(TypeError):
        return asdict(thing)

    if isinstance(thing, datetime):
        return thing.isoformat(timespec="microseconds")

    raise TypeError(f"object of type {type(thing).__name__} not json serializable")


def fnv1a(data: Any):
    hash_value = 0xCBF29CE484222325
    data = json.dumps(data, default=json_default).encode("utf-8")
    for char in data:
        hash_value = hash_value ^ char
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value

def as_json(data: Any):
    return json.dumps(data, default=json_default)