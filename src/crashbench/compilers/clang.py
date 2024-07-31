from functools import cache
from pathlib import Path
import re

from crashbench.util import run
from .compiler import Compiler, Dialect, builtin
from .gcc import GCC


class Clang(GCC):
    has_gnu_extensions     = True
    standard_pattern       = re.compile(r"use '(?P<standard>[^']+)'")
    standard_alias_pattern = re.compile(r"(( or|,) '(?P<alias>[^']+))")
    executable_pattern     = r"clang(-[0-9]+)?(\.exe|\.EXE)?$"

    @staticmethod
    def get_supported_dialects(compiler: Path):
        result = run([str(compiler), "-xc++", "-std=dummy", "-"])
        for line in result.stderr.splitlines():
            standard_match = re.search(Clang.standard_pattern, line)
            if standard_match is None:
                continue
            standard = standard_match["standard"]

            if standard.startswith("gnu"):
                continue

            aliases = [
                match["alias"] for match in Clang.standard_alias_pattern.finditer(line)
            ]
            yield Dialect(standard, aliases)

    @builtin
    def trace(self, compilers: list[Compiler], enabled: bool):
        if not enabled:
            return compilers

        for compiler in compilers:
            compiler.add_option("-ftime-trace")

        return compilers
