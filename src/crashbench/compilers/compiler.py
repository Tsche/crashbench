from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

from crashbench.util import remove_duplicates, which

Settings = dict[str, Any]
Variables = dict[str, Any]


class Compiler(ABC):
    def __init__(self, path: Path):
        self.executable = path
        self.info = self.get_compiler_info(path)
        self.dialects = self.get_supported_dialects(path)

    def __str__(self):
        name = f"{self.__class__.__name__} {self.info['version']}"
        if "target" in self.info:
            name += " " + self.info["target"]
        return name

    @classmethod
    @abstractmethod
    def get_compiler_info(cls, compiler: Path) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_supported_dialects(cls, compiler: Path) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def discover(cls):
        assert hasattr(
            cls, "executable_pattern"
        ), "Class lacks executable search pattern"
        extra = []
        if env_cxx := os.environ.get("CXX"):
            extra.append(Path(env_cxx))

        if env_cc := os.environ.get("CC"):
            extra.append(Path(env_cc))

        for compiler in remove_duplicates(which(getattr(cls, "executable_pattern"), extra)):
            yield cls(compiler)

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

    def compile_commands(self, source: Path, settings: Settings, variables: Variables):
        return [
            str(self.executable),
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
