from dataclasses import asdict
from pathlib import Path
from typing import Optional

import click

from .runner import Pool, Runner
from .parser import TranslationUnit, print_tree
from .sysinfo import SYSTEM_INFO
from .compilers import COMPILERS, Compiler
from .util import as_json, fnv1a

@click.command()
@click.option("--tree-query", type=str, default=None)
@click.option("--system-info", type=bool, is_flag=True, default=False)
@click.option("--emit-tree", type=bool, is_flag=True, default=False)
@click.option("--preprocess", type=bool, is_flag=True, default=False)
@click.option("--list-runs", type=bool, is_flag=True, default=False)
@click.option("--list-compilers", type=bool, is_flag=True, default=False)
@click.option("--dry", type=bool, is_flag=True, default=False)
@click.option("--keep", type=bool, is_flag=True, default=False)
@click.option("--pin-cpu", type=str, default=None)
@click.option("--jobs", type=int, default=0)
@click.argument("file", type=Path, required=False)
@click.pass_context
def main(
    ctx,
    tree_query: Optional[str],
    system_info: bool,
    emit_tree: bool,
    preprocess: bool,
    list_runs: bool,
    list_compilers: bool,
    dry: bool,
    keep: bool,
    pin_cpu: Optional[str],
    jobs: int,
    file: Optional[Path],
) -> int:
    if system_info:
        for key, value in asdict(SYSTEM_INFO).items():
            print(f"{key:<15}: {value}")
        return 0

    if list_compilers:
        # TODO do compiler discovery
        compilers: list[Compiler] = list([compiler.discover() for compiler in COMPILERS])

        for compiler in compilers:
            print(compiler)
            for dialect in getattr(compiler, "dialects", []):
                print(f"    {dialect!s}")
        return 0

    # all commands after this require a file, so check it now
    if file is None:
        raise click.MissingParameter(ctx=ctx, param_hint="FILE", param_type="argument")

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

    # TODO parse cpu pinning argument
    with Pool(num_jobs=jobs, pin_cpu=None) as pool:
        runner = Runner(pool, keep_files=keep)
        result = runner.run(file, dry=dry)

        if keep:
            print(f"Temporary build files written to: {runner.build_dir.name}")

        path = Path.cwd() / "scratchpad.json"
        path.write_text(as_json(result, indent=4))
        draw_plot(result)
    return 0

def draw_plot(report):
    from bokeh.plotting import figure, show
    from bokeh.models import ResetTool, CrosshairTool, HoverTool, PanTool, WheelZoomTool
    from collections import defaultdict
    for name, runs in report.tests.items():
        for compiler, results in runs.items():
            compiler_info = report.compilers[compiler]
            name = f"{compiler_info.family}-{compiler_info.version['version']} ({compiler_info.version['target']})"
            plot = figure(width=1200,
                          height=800,
                          title=name,
                          x_axis_label="Count",
                          y_axis_label="Translation Time (ms)",
                          active_scroll=WheelZoomTool(),
                          tools=[ResetTool(), WheelZoomTool(), PanTool(), HoverTool(), CrosshairTool()])

            output: dict = defaultdict(list)
            for result in results:
                output[result.variables['STRATEGY']].append((result.variables['COUNT'], result.results.elapsed_ms))
            output = {label: list(sorted(values)) for label, values in output.items()}

            for label, values in output.items():
                values = sorted(values)
                xvals, yvals = zip(*values)

                color = f'#{hex(fnv1a(label))[2:8]}'
                plot.step(xvals, yvals, mode="center", legend_label=label, color=color)
                # plot.line(xvals, yvals, legend_label=label, color=color)
                # plot.scatter(xvals, yvals, fill_color="white", size=8)
            plot.legend.click_policy = "hide"
            show(plot)
