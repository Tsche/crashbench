from collections import defaultdict
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Optional, TypeVar


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


def which(query: re.Pattern | str, extra_search_paths: Optional[list[str]] = None):
    query = re.compile(query) if isinstance(query, str) else query
    env_path = os.environ.get("PATH", os.environ.get("Path", os.defpath))
    paths = [Path(path) for path in env_path.split(os.pathsep)]

    if extra_search_paths:
        paths.extend(Path(path) for path in extra_search_paths)

    for path in paths:
        if not path.exists():
            continue

        if not path.is_dir():
            if query.match(path.name):
                yield path.resolve()
            continue

        for file in path.iterdir():
            if query.match(file.name):
                # resolving here should get rid of symlink aliases
                yield file.resolve()


Element = TypeVar("Element")


def remove_duplicates(data: Iterable[Element]) -> list[Element]:
    return [*{entry: None for entry in data}.keys()]


@dataclass
class Result:
    command: str
    returncode: int
    stdout: str
    stderr: str
    start_time: int
    end_time: int

    @property
    def elapsed(self):
        assert self.end_time >= self.start_time, "Invalid time measurements"
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self):
        return self.elapsed / 1e6

    @property
    def elapsed_s(self):
        return self.elapsed / 1e9


def run(command: list[str] | str, env: Optional[dict[str, str]] = None):
    command_str = command if isinstance(command, str) else " ".join(command)
    logging.debug(command_str)
    start_time = time.monotonic_ns()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
        env=env,
    )
    end_time = time.monotonic_ns()

    return Result(
        command_str,
        result.returncode,
        result.stdout,
        result.stderr,
        start_time,
        end_time,
    )


handlers: dict[str, list[Any]] = defaultdict(set)


def handle(**criteria):
    class Handler:
        def __init__(self, fnc):
            self.fnc = fnc

            # TODO disambiguate functions with same name in the same module and class namespace
            # for which the parameter list differs in size
            self.name = self.fnc.__module__, self.fnc.__qualname__
            self.criteria = criteria

            if self in handlers[self.name]:
                raise ValueError(
                    f"Ambiguous overload for `{self.fnc.__qualname__}` with criteria {self.criteria}"
                )

            handlers[self.name].add(self)

        def __eq__(self, other: "Handler"):
            return other.criteria == self.criteria

        def __hash__(self):
            return hash(tuple(sorted(self.criteria.items())))

        @staticmethod
        def check_criterium(actual, expected):
            if isinstance(expected, Iterable):
                # allow matching against multiple options
                return actual in expected

            return actual == expected

        def matches(self, node):
            for key, value in self.criteria.items():
                if hasattr(node, key):
                    if self.check_criterium(getattr(node, key), value):
                        continue
                    return False

                if not hasattr(node, "__getitem__"):
                    # not subscriptable
                    continue

                try:
                    if (item := node[key]) is not None:
                        if self.check_criterium(item, value):
                            continue
                        return False
                except TypeError:
                    # cannot subscript with string key - probably a list or similar
                    return False

            return True

        def can_handle(self, node):
            for handler in handlers[self.name]:
                if handler.matches(node):
                    return True
            return False

        def __call__(self, node):
            for handler in handlers[self.name]:
                if handler.matches(node):
                    return handler.fnc(node)
            raise ValueError("No appropriate overload found")
    return Handler
