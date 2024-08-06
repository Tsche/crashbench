from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from .sysinfo import SYSTEM_INFO, SYSTEM_INFO_HASH, SystemInfoHash
from .util import fnv1a

CompilerHash: TypeAlias = int

Variables: TypeAlias = dict[str, str]
VariablesHash: TypeAlias = int

Results: TypeAlias = list[int]

V = TypeVar("V")


class HashedDict(dict[int, V]):
    def __getitem__(self, key: int | V | Any) -> V:
        # key may be Any to support foreign keys
        if not isinstance(key, int):
            key = fnv1a(key)
        return super().__getitem__(key)

    def append(self, item: V):
        key = fnv1a(item)
        if key not in self:
            self[key] = item

@dataclass
class Compiler:
    family: str  # gcc/clang/msvc
    version: str
    dialect: str  # selected C++ standard
    options: tuple[str]  # compile options, excluding macro definitions

    def __post_init__(self):
        self.options = tuple(sorted(self.options))

@dataclass
class Run:
    variables: Variables
    results: dict[CompilerHash, Results]

@dataclass
class Test:
    runs: list[Run]

@dataclass
class Report:
    file: Path
    system: dict[str, str | int] =  field(default_factory=SYSTEM_INFO)
    compilers: HashedDict[Compiler] = field(default_factory=HashedDict)
    tests: list[Test] = field(default_factory=list)

    def add_compiler(self, compiler: Compiler):
        hash_value = fnv1a(compiler)
        if hash_value not in self.compilers:
            self.compilers[hash_value] = compiler

@dataclass
class Commit:
    commit_id: str
    results: list[Report]


# if __name__ == "__main__":
#     compilers = HashedDict()
#     compilers.append(Compiler("gcc", "14", "26", ("-O3", "-W")))
#     compilers.append(Compiler("clang", "19", "26", ("-O3",)))
#     tests = [Test([
#             Run({"COUNT": 1, "STRATEGY": "recursive"}, HashedDict({
#                 fnv1a(Compiler("gcc", "14", "26", ("-O3",))): [200, 203, 201],
#                 fnv1a(Compiler("clang", "19", "26", ("-O3",))): [420, 353, 401]
#             })),
#             Run({"COUNT": 2, "STRATEGY": "recursive"}, HashedDict({
#                 fnv1a(Compiler("gcc", "14", "26", ("-O3",))): [200, 203, 201],
#                 fnv1a(Compiler("clang", "19", "26", ("-O3",))): [420, 353, 401]
#             }))])]

#     report = Report(SYSTEM_INFO, compilers, tests)
#     from pprint import pprint
#     pprint(report)
