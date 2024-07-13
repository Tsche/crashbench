from functools import cache
from pathlib import Path
import re

from crashbench.util import run
from .compiler import Compiler, Dialect
from .gcc import GCC


class Clang(Compiler):
    has_gnu_extensions     = True
    standard_pattern       = re.compile(r"use '(?P<standard>[^']+)'")
    standard_alias_pattern = re.compile(r"(( or|,) '(?P<alias>[^']+))")
    executable_pattern     = r"clang(-[0-9]+)?(\.exe|\.EXE)?$"
    version_pattern        = GCC.version_pattern
    get_compiler_info      = GCC.get_compiler_info
    select_language        = GCC.select_language
    select_dialect         = GCC.select_dialect
    define                 = GCC.define

    @staticmethod
    @cache
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