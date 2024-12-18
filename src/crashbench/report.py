from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeAlias

from .sysinfo import SYSTEM_INFO, SystemInfo
from .util import ExecResult, fnv1a, to_base58
from .parser import Compiler

CompilerHash: TypeAlias = int

Variables: TypeAlias = dict[str, str]
VariablesHash: TypeAlias = int


@dataclass
class CompilerInfo:
    family: str  # gcc/clang/msvc
    version: dict[str, str]
    path: str  #
    options: tuple[str, ...]  # compile options, excluding macro definitions

    def __init__(self, compiler: Compiler):
        self.family = compiler.name
        self.version = compiler.get_compiler_info(compiler.path)
        self.path = str(compiler.path)
        options: list[str] = [str(option) for key, option in compiler.options.items() if key != 'o']
        self.options = tuple(sorted(options))

    def __str__(self):
        return f"{self.family} {self.version['version']}"

    def __hash__(self):
        return fnv1a(asdict(self))

    def __int__(self):
        return hash(self)

    def __eq__(self, other):
        return int(self) == int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def __ge__(self, other):
        return int(self) >= other

    def __le__(self, other):
        return int(self) <= int(other)

    def to_hash(self):
        return to_base58(hash(self))

@dataclass
class Result:
    compiler: CompilerInfo
    variables: Variables
    results: ExecResult
    extra_files: list[Path] = field(default_factory=list)

    def to_json(self):
        items = asdict(self)
        # replace compiler info with hash
        items["compiler"] = self.compiler.to_hash()
        return items

@dataclass
class Report:
    file: Path
    system: SystemInfo = field(default_factory=lambda: SYSTEM_INFO)
    compilers: set[CompilerInfo] = field(default_factory=set)
    tests: dict[str, list[Result]] = field(default_factory=dict)

    def add_compiler(self, compiler: CompilerInfo):
        self.compilers.add(compiler)

    def add_result(self, test: str, result: Result):
        if test not in self.tests:
            self.tests[test] = []
        self.add_compiler(result.compiler)
        self.tests[test].append(result)

    def as_dict(self):
        return {
            'file': self.file,
            'system': self.system,
            'compilers': {compiler.to_hash(): compiler for compiler in self.compilers},
            'tests': {test_name: results for test_name, results in self.tests.items()}
        }
