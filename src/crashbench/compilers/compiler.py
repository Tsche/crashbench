import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from crashbench.util import remove_duplicates, which

Settings = dict[str, Any]
Variables = dict[str, Any]


class Compiler:
    def __init__(self, path: Path, info: dict[str, str], dialects: dict[str, str]):
        self.path = path
        self.info = info
        self.dialects = dialects


class CompilerFamily(ABC):
    def __str__(self):
        return self.__class__.__name__

    @classmethod
    @abstractmethod
    def get_compiler_info(cls, compiler: Path) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_supported_dialects(cls, compiler: Path) -> dict[str, str]:
        raise NotImplementedError

    @cached_property
    def detected(self):
        assert hasattr(
            self, "executable_pattern"
        ), "Class lacks executable search pattern"

        extra: list[str] = []
        if env_cxx := os.environ.get("CXX"):
            extra.append(env_cxx)

        if env_cc := os.environ.get("CC"):
            extra.append(env_cc)

        return [
            Compiler(path, self.get_compiler_info(path), self.get_supported_dialects(path))
            for path in remove_duplicates(which(getattr(self, "executable_pattern"), extra))
        ]

    @staticmethod
    @abstractmethod
    def select_language(language: str) -> list[str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def select_dialect(language: str) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def define(language: str) -> str:
        raise NotImplementedError

    def get_compilers(self, settings):
        ...

    def compile_commands(self, source: Path, settings: Settings, variables: Variables):
        # TODO find actual compilers
        return [
            # str(self.executable),
            str(source),
            *[f"-D{option}={value}" for option, value in variables.items()],
        ]


@dataclass
class Dialect:
    name: str
    aliases: Optional[list[str]] = None

    def __str__(self):
        name = self.name
        if self.aliases:
            name += f" ({', '.join(self.aliases)})"
        return name
