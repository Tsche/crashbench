from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional, TypeAlias, TypeVar

from .sysinfo import SYSTEM_INFO, SYSTEM_INFO_HASH, SystemInfo, SystemInfoHash
from .util import Result, as_json, fnv1a, to_base58

from .parser import Compiler

CompilerHash: TypeAlias = int

Variables: TypeAlias = dict[str, str]
VariablesHash: TypeAlias = int


class Hash:
    __slots__ = ['data']

    def __init__(self, data: Any):
        self.data: int = fnv1a(data)

    def __str__(self):
        return to_base58(self.data)

    def __eq__(self, other): 
        if isinstance(other, Hash):
            return self.data == other.data
        return self.data == other

    def __hash__(self):
        return hash(self.data)


V = TypeVar("V")


@dataclass
class CompilerInfo:
    family: str  # gcc/clang/msvc
    version: dict[str, str]
    path: str  #
    options: tuple[str, ...]  # compile options, excluding macro definitions

    def __init__(self, compiler: Compiler):
        self.family = compiler.name
        self.version =  compiler.get_compiler_info(compiler.path)
        self.path = str(compiler.path)
        options: list[str] = [str(option) for key, option in compiler.options.items() if key != 'o']
        self.options = tuple(sorted(options))


@dataclass
class Run:
    variables: Variables
    results: Result
    extra_files: list[Path] = field(default_factory=list)

@dataclass
class Report:
    file: Path
    system: SystemInfo = field(default_factory=lambda: SYSTEM_INFO)
    compilers: dict[Hash, CompilerInfo] = field(default_factory=dict)
    tests: dict[str, dict[Hash, list[Run]]] = field(default_factory=dict)

    def add_compiler(self, compiler: CompilerInfo):
        hash_value = Hash(compiler)
        if hash_value not in self.compilers:
            self.compilers[hash_value] = compiler

    def add_result(self, compiler: Compiler, test: str, variables: Variables, result: Result, extra_files: Optional[list[Path]] = None):
        compiler_info = CompilerInfo(compiler)
        compiler_hash = Hash(compiler_info)
        if compiler_hash not in self.compilers:
            self.compilers[compiler_hash] = compiler_info

        if test not in self.tests:
            self.tests[test] = {}

        if compiler_hash not in self.tests[test]:
            self.tests[test][compiler_hash] = []

        self.tests[test][compiler_hash].append(Run(variables, result, extra_files or []))

    def as_dict(self):
        return {
            'file': self.file,
            'system': self.system,
            'compilers': {str(key): value for key, value in self.compilers.items()},
            'tests': {test_name: {str(key): value for key, value in results.items()} for test_name, results in self.tests.items()}
        }