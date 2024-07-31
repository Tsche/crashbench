from dataclasses import asdict
from pathlib import Path
from typing import Optional

import click

from .runner import Runner
from .parser import TranslationUnit, print_tree
from .sysinfo import SYSTEM_INFO
from .compilers import CompilerFamily

@click.command()
@click.option("--tree-query", type=str, default=None)
@click.option("--system-info", type=bool, is_flag=True, default=False)
@click.option("--emit-tree", type=bool, is_flag=True, default=False)
@click.option("--preprocess", type=bool, is_flag=True, default=False)
@click.option("--list-runs", type=bool, is_flag=True, default=False)
@click.option("--list-compilers", type=bool, is_flag=True, default=False)
@click.option("--pin-cpu", type=str, default=None)
@click.option("--jobs", type=int, default=None)
@click.argument("file", type=Path)
def main(
    tree_query: Optional[str],
    system_info: bool,
    emit_tree: bool,
    preprocess: bool,
    list_runs: bool,
    list_compilers: bool,
    pin_cpu: Optional[str],
    jobs: Optional[int],
    file: Path,
) -> int:
    if system_info:
        for key, value in asdict(SYSTEM_INFO).items():
            print(f"{key:<15}: {value}")
        return 0

    if list_compilers:
        # TODO do compiler discovery
        # compilers: list[CompilerFamily] = list(discover())

        # for compiler in compilers:
        #     print(compiler)
        #     for dialect in getattr(compiler, "dialects", []):
        #         print(f"    {dialect!s}")
        return 0

    has_flags = any([emit_tree, tree_query, preprocess, list_runs])
    # todo dispatch flag-like commands properly
    if has_flags:
        if emit_tree or tree_query:
            print_tree(file, tree_query)
            print()
            return 0

        parsed = TranslationUnit.from_file(file)
        if preprocess:
            print(parsed.source)
            print()
            return 0

        if list_runs:
            print(f"File: {file}")
            print(parsed)
            print()
            for test in parsed.tests:
                print(f"Evaluated: \n    {test.evaluated}")
                print(f"Runs: \n    {test.runs}")
                print()
            return 0

    runner = Runner(jobs, pin_cpu)
    runner.run(file)
    return 0
