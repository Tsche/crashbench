from itertools import groupby
from typing import Iterable


def group(data, key_fnc):
    data = sorted(data, key=key_fnc)
    return [(key, list(group)) for key, group in groupby(data, key_fnc)]


def sort(data, key_fnc):
    return sorted(data, key=key_fnc)


def by_variable(var: str):
    return lambda x: x.variables[var]


def by_compiler():
    return lambda data: data.compiler


def step(label: str, data: Iterable):
    return label, list(data)


def figure(title: str,
           data: Iterable,
           x_axis_label: str = "",
           y_axis_label: str = ""):
    
    return title, list(data)

def table(data: Iterable, title: str = ""):
    if title:
        print(title)

    data = list(data)
    if not data:
        return []

    header = list(dict(data[0]).keys())
    for label in header:
        print(f"{label:<15}", end="")
    print()
    for item in data:
        for value in item.values():
            print(f"{value:<15}", end="")
        print()
    print()


OUTPUT_BUILTINS = {
    'group': group,
    'sort': sort,
    'by_variable': by_variable,
    'by_compiler': by_compiler,

    'figure': figure,
    'step': step,
    'table': table
}
