import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Optional, Self
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from crashbench.exceptions import ParseError

from crashbench.util import remove_duplicates, which

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
        name = self.name
        if self.aliases:
            name += f" ({', '.join(self.aliases)})"
        return name

    __repr__ = __str__

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

class Compiler:
    def __init__(self, compiler_family: 'CompilerFamily', path: Path,
                 language: Optional[str] = None, 
                 dialect: Optional[Dialect] = None,
                 options: Optional[list[str]] = None,
                 assertions: Optional[Any] = None):

        self.compiler_family = compiler_family
        self.path = path
        self.language = language or 'c++'
        self.dialect = dialect
        self.options: list[str] = options or []
        self.assertions: list[str] = assertions or []
        self.warnings: dict[str, bool] = {}

    @cached_property
    def info(self):
        return self.compiler_family.get_compiler_info(self.path)

    @cached_property
    def supported_dialects(self):
        return list(self.compiler_family.get_supported_dialects(self.path))

    def copy(self):
        return Compiler(self.compiler_family, self.path, self.language, self.dialect, self.options.copy(), self.assertions.copy())

    def add_option(self, option: str):
        self.options.append(option)

    def add_assertion(self, assertion: Any):
        self.assertions.append(assertion)

    def expand_dialect(self, dialect: str | int):
        if isinstance(dialect, str):
            if not dialect.isnumeric():
                return dialect
            dialect = int(dialect)
        return f"c++{dialect}"

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
        return self.supported_dialects[left_index + exclude_left : right_index - exclude_right + 1]

    def matches_version(self, selector: SpecifierSet):
        return Version(self.info['version']) in selector

    def select_dialects(self, query: str):
        if dialects := self.filter_dialects(query):
            for dialect in dialects:
                yield Compiler(self.compiler_family, self.path, self.language, dialect, self.options)

        else:
            yield Compiler(self.compiler_family, self.path, self.language, None, self.options)

    def __repr__(self):
        return f'Compiler({self.compiler_family}, {self.path}, "{self.language}", "{self.dialect}", {" ".join(self.options)})'

    def compile_command(self, source: Path, test: str, *options, variables: Optional[dict[str, Any]] = None):
        return [
            str(self.path),
            str(source),
            *self.options,
            *options,
            self.compiler_family.define(test.upper()),
            *[self.compiler_family.define(option, value) for option, value in (variables or {}).items()]
        ]

COMPILER_BUILTINS: dict[type, dict[str, Callable]] = {}
def builtin(fnc=None, assertion: bool = False):
    class Builtin:
        def __init__(self, target):
            self.fnc = target
            self.assertion = assertion

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

class CompilerFamily(ABC):
    name: str

    def __init__(self):
        self.assertions: list[Any] = []

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
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
            Compiler(self, path)
            for path in remove_duplicates(which(getattr(self, "executable_pattern"), extra))
        ]

    @staticmethod
    @abstractmethod
    def select_language(language: str) -> list[str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def select_dialect(dialect: Dialect) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def define(name: str, value: Optional[Any] = None) -> str:
        raise NotImplementedError

    @builtin(assertion=True)
    def error(self, message: str, regex: bool=False):
        # self.assertions.append((message, regex))
        ...

    @builtin
    def standard(self, compilers: list[Compiler], selector):
        for compiler in compilers:
            yield from compiler.select_dialects(selector)

    @builtin
    def enabled(self, compilers: list[Compiler], value: bool):
        return compilers if value else []
    
    @builtin
    def version(self, compilers: list[Compiler], selector: str):
        specifier = SpecifierSet(selector)
        for compiler in compilers:
            if compiler.matches_version(specifier):
                yield compiler

