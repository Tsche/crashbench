from dataclasses import dataclass
import os
from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Optional, Self
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from crashbench.exceptions import ParseError

from crashbench.util import fnv1a, remove_duplicates, to_base58, which


class Dialect:
    def __init__(self, name: str, aliases: Optional[list[str]] = None):
        self.name = name
        self.aliases = aliases or []

        self.language, self.version = self.parse(name)
        if not self.version.isnumeric():
            # find numeric alias as primary
            for alias in self.aliases[:]:
                _, self.version = self.parse(alias)
                if not self.version.isnumeric():
                    continue

                self.aliases.remove(alias)
                self.aliases.append(name)
                self.name = alias
                break

        assert self.version.isnumeric(), "Could not find numeric standard version"
        self.version = int(self.version)

    @staticmethod
    def parse(name: str):
        query = r"(?P<language>[a-zA-Z\+]+)(?P<version>[0-9]+[a-z]*)"
        matches = re.match(query, name)
        assert matches is not None
        language, dialect = matches.groups()
        return language, dialect

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Dialect({self.name}, aliases=[{', '.join(self.aliases)}]"

    def __gt__(self, other: Self | int):
        value = other if isinstance(other, int) else other.version
        return self.version > value

    def __ge__(self, other: Self | int):
        value = other if isinstance(other, int) else other.version
        return self.version > value

    def __le__(self, other: Self | int):
        value = other if isinstance(other, int) else other.version
        return self.version < value

    def __lt__(self, other: Self | int):
        value = other if isinstance(other, int) else other.version
        return self.version < value

    def __eq__(self, other: Self | int | str):
        if isinstance(other, int):
            return self.version == other
        elif isinstance(other, str):
            return self.name == other or other in self.aliases

        return self.version == other.version

    def __iter__(self):
        yield self.name
        if self.aliases is not None:
            yield from self.aliases


COMPILER_BUILTINS: dict[type, dict[str, Callable]] = {}


def builtin(fnc=None, repeatable: bool = False):
    class Builtin:
        def __init__(self, target):
            self.fnc = target
            self.repeatable = repeatable

        def __repr__(self):
            return self.fnc.__qualname__

        def __set_name__(self, owner, name):
            if owner not in COMPILER_BUILTINS:
                # first builtin for this type
                COMPILER_BUILTINS[owner] = {}

                # inherit builtins from parents
                for base in owner.__bases__:
                    if base not in COMPILER_BUILTINS:
                        continue
                    COMPILER_BUILTINS[owner] |= COMPILER_BUILTINS[base]

            # store this object in COMPILER_BUILTINS
            COMPILER_BUILTINS[owner][name] = self

            # but replace the name in the class with the actual function
            setattr(owner, name, self.fnc)

        def __call__(self, *args, **kwargs):
            # make the actual function callable through Builtin objects
            return self.fnc(*args, **kwargs)

    return Builtin if fnc is None else Builtin(fnc)


@dataclass(frozen=True)
class Option:
    name: str
    value: Any = None
    separator: Optional[str] = None

    def __str__(self):
        if self.value is None:
            return self.name
        return f"{self.name}{self.separator or ''}{self.value or ''}"


class Compiler(ABC):
    name: str

    def __init__(self, path: Path,
                 language: Optional[str] = None,
                 options: Optional[dict[str, Option]] = None,
                 assertions: Optional[list[Any]] = None,
                 warnings: Optional[dict[str, bool]] = None, 
                 extra_files: Optional[list[str]] = None):

        self.path = path
        self.language = language or 'c++'
        self.options: dict[str, Option] = options.copy() if options else {
            'c': Option('c')  # disable linking by default
        }
        self.assertions: list[Any] = assertions.copy() if assertions else []
        self.warnings: dict[str, bool] = warnings.copy() if warnings else {}
        self.expected_return_code: int = 0
        self.extra_files: list[str] = extra_files or []

    @property
    def hash(self):
        return fnv1a((self.path,
                      self.get_compiler_info(self.path),
                      self.options))

    @property
    def supported_dialects(self):
        return self.get_supported_dialects(self.path)

    def copy(self):
        return type(self)(self.path, self.language,
                          self.options.copy(), self.assertions.copy(), self.warnings.copy(), self.extra_files.copy())

    def add_option(self, option: str, value: Optional[Any] = None, separator=None):
        default_separator = getattr(self, "default_option_separator", " ")
        self.options[option] = Option(option, value, default_separator if separator is None else separator)
        return self

    def remove_option(self, option: str):
        if option in self.options:
            del self.options[option]
        return self

    def add_assertion(self, assertion: Any):
        self.assertions.append(assertion)
        return self

    def expand_dialect(self, dialect: str | int):
        if isinstance(dialect, str):
            if not dialect.isnumeric():
                return dialect
            dialect = int(dialect)
        return f"{self.language}{dialect}"

    def filter_dialects(self, query: str):
        query = query.strip()
        if query[0] in ('(', '['):
            return self.filter_dialect_interval(query)
        elif query[0] not in ('<', '>', '='):
            return [self.supported_dialects[self.supported_dialects.index(self.expand_dialect(query.strip()))]]
        return self.filter_dialect_selector(query)

    def filter_dialect_selector(self, query: str):
        greater = query[0] == '>'
        include = query[1] == '='
        version = query[1 + include:]
        try:
            index = self.supported_dialects.index(self.expand_dialect(version.strip()))
            index += include ^ greater
            return self.supported_dialects[index:] if greater else self.supported_dialects[:index]
        except StopIteration as e:
            raise ParseError(f"Could not find value {version} in {self.supported_dialects}",
                             getattr(version, "node", None)) from e

    def filter_dialect_interval(self, query: str):
        # interval notation
        if ',' not in query or query.count(',') != 1:
            raise ParseError(f"Invalid dialect range `{query}`. Expected exactly 2 selectors.",
                             getattr(query, "node", None))

        if query[-1] not in (')', ']'):
            raise ParseError("Invalid dialect range `{query}` specified. Query must end in ] or )",
                             getattr(query, "node", None))

        left, right = query[1:-1].split(',', 1)
        exclude_left = query[0] != '['
        exclude_right = query[-1] != ']'
        left_index = self.supported_dialects.index(self.expand_dialect(left.strip()))
        right_index = self.supported_dialects.index(self.expand_dialect(right.strip()))
        return self.supported_dialects[left_index + exclude_left: right_index - exclude_right + 1]

    def matches_version(self, selector: SpecifierSet):
        return Version(self.get_compiler_info(self.path)['version']) in selector

    def __repr__(self):
        return f'{self.name}({self.path})'

    def __str__(self):
        return self.__class__.__name__

    def __init_subclass__(cls) -> None:
        setattr(cls, "name", cls.__name__)
        return super().__init_subclass__()

    @classmethod
    @abstractmethod
    def get_compiler_info(cls, compiler: Path) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_supported_dialects(cls, compiler: Path) -> Iterable[Dialect]:
        raise NotImplementedError

    @classmethod
    @cache
    def discover(cls):
        assert hasattr(
            cls, "executable_pattern"
        ), "Class lacks executable search pattern"

        extra: list[str] = []
        if env_cxx := os.environ.get("CXX"):
            extra.append(env_cxx)

        if env_cc := os.environ.get("CC"):
            extra.append(env_cc)

        return [
            cls(path)
            for path in remove_duplicates(which(getattr(cls, "executable_pattern"), extra))
        ]

    @staticmethod
    @abstractmethod
    def define(name: str, value: Optional[Any] = None) -> str:
        raise NotImplementedError

    @builtin(repeatable=True)
    def error(self, compilers: list['Compiler'], message: str, regex: bool = False):
        for compiler in compilers:
            yield compiler.add_assertion((message, regex))

    @builtin
    def return_code(self, compilers: list['Compiler'], expected: int):
        for compiler in compilers:
            compiler.expected_return_code = expected
            yield compiler

    # @classmethod
    @builtin
    def standard(cls, compilers: list['Compiler'], selector: str):
        for compiler in compilers:
            # TODO handle erroneous selections and empty results properly

            if dialects := compiler.filter_dialects(selector):
                for dialect in dialects:
                    yield compiler.copy().add_option('std', dialect.name)
            else:
                # TODO empty result might mean that no dialects matched the selection
                yield compiler.copy().remove_option('std')

    # @classmethod
    @builtin
    def enabled(cls, compilers: list['Compiler'], value: bool):
        return compilers if value else []

    # @classmethod
    @builtin
    def version(cls, compilers: list['Compiler'], selector: str):
        specifier = SpecifierSet(selector)
        for compiler in compilers:
            if compiler.matches_version(specifier):
                yield compiler

    # @classmethod
    @builtin
    def link(cls, compilers: list['Compiler'], enable: bool):
        for compiler in compilers:
            yield compiler.remove_option('c') if enable else compiler.add_option('c')

    @abstractmethod
    def set_output(self, path: Optional[Path]):
        raise NotImplementedError

    def expand_extra_files(self, basepath: Path, name: str):
        for file in self.extra_files:
            if "{}" in file:
                yield basepath / file.format(name)
            else:
                yield basepath / file

    def compile_command(self, source: Path, outpath: Path, test: str, *options, variables: Optional[dict[str, Any]] = None):
        var_hash = to_base58(fnv1a(variables))
        if outpath.is_dir() or (not outpath.exists() and '.' not in outpath.name):
            self.set_output(outpath / f"{var_hash}.o")
        
        return [
            str(self.path),
            str(source),
            *[getattr(self, "option_prefix") + str(option) for option in self.options.values()],
            *options,
            self.define(test.upper()),
            *[self.define(option, value) for option, value in (variables or {}).items()]
        ]

    def run_assertions(self):
        # print(f"Checking assertions {self.assertions}")
        ...
