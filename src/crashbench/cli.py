from pathlib import Path
from pprint import pprint
from subprocess import check_call
import time
from typing import Optional
import click
from .parser import parse, tree
import os


def generate_trace(file, output, arguments: list[str]):
    includes = [
        # '-isystem', str(Path.cwd() / 'include')
    ]

    standard = "-std=c++2c"
    profile_flags = [
        # '-ftime-trace',
        "-c"
    ]

    taskset_prefix = ["taskset", "--cpu-list", "23"]
    call = [
        *taskset_prefix,
        os.environ.get("CXX") or "clang++",
        str(file),
        "-o",
        str(output),
        *arguments,
        *includes,
        standard,
        *profile_flags,
    ]

    output.parent.mkdir(exist_ok=True, parents=True)
    start_time = time.time()
    check_call(call, cwd=file.parent)
    end_time = time.time()
    output.unlink()

    trace_file = output.with_suffix(".json")
    # print(f"Generated {trace_file}")
    return trace_file, end_time - start_time


@click.command()
@click.option("--emit-tree", type=bool, is_flag=True, default=False)
@click.option("--preprocess", type=bool, is_flag=True, default=False)
@click.option("--list-runs", type=bool, is_flag=True, default=False)
@click.option("--tree-query", type=str, default=None)
@click.argument("file", type=Path)  # , nargs=-1)
def main(
    file: Path,
    emit_tree: bool,
    preprocess: bool,
    list_runs: bool,
    tree_query: Optional[str],
):
    if emit_tree or tree_query:
        pprint(str(tree(file, tree_query)))
        return

    source, tests = parse(file)
    if preprocess:
        print(source)
        return

    if list_runs:
        for test in tests:
            print(test.runs)
        return

    build_folder = Path.cwd() / "build"
    build_folder.mkdir(exist_ok=True, parents=True)

    preprocessed = build_folder / file.name
    preprocessed.write_text(source)
    idx = 0
    for test in tests:
        print(f"Name: {test.name}")
        for run in test.runs:
            idx += 1
            trace = generate_trace(preprocessed, preprocessed.with_stem(f"{preprocessed.stem}_run_{idx}"), run)
            print(f"{trace[1]*1000} {run}")
