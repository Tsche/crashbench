from collections import defaultdict
from functools import cache
import re

from pathlib import Path
from typing import Optional

from crashbench.util import run

from .compiler import Compiler, Dialect


class GCC(Compiler):
    has_gnu_extensions = True
    executable_pattern = r"^gcc(-[0-9]+)?(\.exe|\.EXE)?$"
    version_pattern = re.compile(
        r"((Target: (?P<target>.*))|(Thread model: (?P<thread_model>.*))|"
        r"((gcc|clang) version (?P<version>[0-9\.]+)))"
    )
    standard_pattern = re.compile(
        r"-std=(?P<standard>[^\s]+)[\s]*(Conform.*(C\+\+( draft)? standard))"
        r"((.|(\n    )\s+)*Same.as(.|(\n    )\s+)*-std=(?P<alias>[^\. ]+))?"
    )

    def __init__(self, path: Path, cpp_driver: Optional[Path] = None):
        super().__init__(path)
        self.cpp_driver = cpp_driver

    @classmethod
    def get_compiler_info(cls, compiler: Path) -> dict[str, str]:
        # invoke gcc -v --version
        result = run([str(compiler), "-v", "--version"])

        info: dict[str, str] = {}
        for source in result.stderr, result.stdout:
            for match in re.finditer(cls.version_pattern, source):
                info |= {k: v for k, v in match.groupdict().items() if v}

        assert "version" in info, "Automatic version detection failed"
        return info

    @staticmethod
    @cache
    def get_supported_dialects(compiler: Path):
        standards = defaultdict(list)
        # invoke gcc -v --help
        result = run([str(compiler), "-v", "--help"])
        for match in GCC.standard_pattern.finditer(result.stdout):
            standard = match['standard']

            if standard.startswith("gnu"):
                continue

            if alias := match['alias']:
                standards[alias].append(standard)
            else:
                standards[standard].append(standard)
        return [Dialect(standard, [alias for alias in aliases if alias != standard])
                for standard, aliases in standards.items()]

    # @classmethod
    # def discover(cls):
    #     c_pattern = r"^gcc(-[0-9]+)?(\.exe|\.EXE)?$"
    #     cpp_pattern = r"^g\+\+(-[0-9]+)?(\.exe|\.EXE)?$"

    #     def get_compilers(pattern):
    #         for compiler in remove_duplicates(which(pattern)):
    #             info = cls.get_compiler_info(compiler)
    #             yield info['version'], compiler

    #     c_compilers = dict(get_compilers(c_pattern))
    #     cpp_compilers = dict(get_compilers(cpp_pattern))
    #     assert len(c_compilers) >= len(cpp_compilers), "Could not find C compiler for all C++ language drivers"
    #     for version, compiler in c_compilers.items():
    #         yield GCC(compiler,  cpp_compilers.get(version))

    @staticmethod
    def select_language(language: str) -> list[str]:
        if language == 'c++':
            return ['-xc++', '-lstdc++']
        elif language == 'c':
            return []
        raise ValueError(f"Language {language} is not supported.")

    @staticmethod
    def select_dialect(dialect: str) -> str:
        return f'-std={dialect}'

    @staticmethod
    def define(name: str, value: Optional[str] = None) -> str:
        return f"-D{name}" if value is None else f"-D{name}={value}"
