from pprint import pprint
from pathlib import Path
from itertools import product
from glob import glob
from typing import Any
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node

from .builtins import BUILTINS

CPP = Language(tree_sitter_cpp.language())

QUERY = {
    file.stem: CPP.query(file.read_text())
    for file in (Path(__file__).parent / "queries").iterdir()
    if file.suffix == ".lisp"
}


def parse_arguments(node: Node):
    for child in node.named_children:
        if child.type != "comment":
            yield parse_constant(child)

class PendingCall:
    def __init__(self, fnc, *arguments):
        self.fnc = fnc
        self.arguments = arguments

    def evaluate_args(self):
        for argument in self.arguments:
            if isinstance(argument, PendingCall):
                yield argument()
            else:
                yield argument

    def __call__(self):
        return self.fnc(*list(self.evaluate_args()))

    def __repr__(self):
        return f"{self.fnc.__name__}({','.join(str(argument) for argument in self.arguments)})"

def parse_call(node: Node):
    name_node = node.child_by_field_name("function")
    name = name_node.text.decode()
    if name not in BUILTINS:
        print("ERR: invalid function called")
        return

    arguments = list(parse_arguments(node.child_by_field_name("arguments")))
    return PendingCall(BUILTINS[name], *arguments)


def parse_constant(node: Node):
    match node.type:
        case "string_literal":
            assert node.named_child_count == 1
            return node.named_children[0].text.decode()
        case "number_literal":
            # TODO integer literal suffixes
            # TODO hex, octal, binary literals
            # TODO separators
            text = node.text.decode()
            try:
                if "." in text:
                    return float(text)
                return int(text)
            except ValueError:
                return text
        case "char_literal":
            assert node.named_child_count == 1
            return node.named_children[0].text.decode()
        case "concatenated_string":
            return "".join(parse_constant(child) for child in node.named_children)
        case "true":
            return True
        case "false":
            return False
        case "null":
            return None
        case "call_expression":
            # TODO when and how is this evaluated?
            # should var(range(0,2), range(0,3)) expand to `"0,1", "0,1,2"`?
            return parse_call(node)
        case "identifier":
            ident = node.text.decode()
            if ident in BUILTINS:
                return BUILTINS[ident]
            # TODO variables
            raise ValueError(f"Could not resolve identifier {ident}")


class Variable(PendingCall):
    def __init__(self, name: str, generator: str, arguments):
        if generator not in BUILTINS:
            raise ValueError(f"Builtin {generator} not recognized")

        self.name = name
        super().__init__(BUILTINS[generator], *list(parse_arguments(arguments)))

    def __call__(self):
        return [f'-D{self.name}={value}' for value in super().__call__()]

    def __repr__(self):
        return f"{self.name}={super().__repr__()}"

class VariableGroup(list[Variable]):
    def __call__(self):
        data = [generator() for generator in self]
        max_length = max(len(sublist) for sublist in data)
        return [[sublist[idx % len(sublist)] for sublist in data] for idx in range(max_length)]


def find_variables(tree):
    for _, node in QUERY["attribute"].matches(tree):
        assert (
            len(node["name"])
            == len(node["generator"])
            == len(node["arguments"])
        )
        count = len(node["name"])
        position = node["variable"].start_byte, node["variable"].end_byte
        if count > 1:
            yield position, VariableGroup([
                Variable(
                    node["name"][idx].text.decode(),
                    node["generator"][idx].text.decode(),
                    node["arguments"][idx],
                )
                for idx in range(len(node["name"]))
            ])

        else:
            yield position, Variable(
                node["name"][0].text.decode(),
                node["generator"][0].text.decode(),
                node["arguments"][0],
            )

class Test:
    def __init__(self, node: dict[str, Node]):
        self.kind = node["kind"].text.decode()
        assert self.kind in ("benchmark", "test")

        self.name = node["name"].text.decode()

        self.start_byte = node["attributes"].start_byte
        self.end_byte = node["code"].end_byte
        self.node = node
        self.variables = dict(find_variables(node["code"]))

    @property
    def code(self) -> list[int]:
        code_node: Node = self.node["code"]
        text = b""
        last = 0
        for start, end in self.variables.keys():
            text += code_node.text[last:start - code_node.start_byte]
            last = end - code_node.start_byte

            if code_node.text[last] == ord('\n'):
                # remove newline if this was the end of the line
                last += 1

        text += code_node.text[last:]
        return text.decode()

    @staticmethod
    def flatten_vars(vars):
        for var in vars:
            if isinstance(var, list):
                yield from var
            else:
                yield var

    @property
    def runs(self) -> list[list[Any]]:
        generators = [generator() for generator in self.variables.values()]
        return [[f'-D{self.name.upper()}', *self.flatten_vars(run)] for run in list(product(*generators))]


def parse(source_path: Path):
    parser = Parser(CPP)
    source = source_path.read_bytes()
    tree_obj = parser.parse(source)
    tests = [Test(node) for _, node in QUERY["compound"].matches(tree_obj.root_node)]
    if len(tests) == 0:
        # no tests discovered
        return

    processed_source = ''
    last = 0
    for test in tests:
        processed_source += source[last: test.start_byte].decode()
        processed_source += f"#ifdef {test.name.upper()}\n"

        processed_source += test.code
        processed_source += '\n'
        processed_source += "#endif"
        last = test.end_byte

    processed_source += source[last:].decode()
    return processed_source, tests


def tree(source: Path, query=None):
    parser = Parser(CPP)
    tree_obj = parser.parse(source.read_bytes())
    if query is None:
        return tree_obj.root_node
    else:
        matches = QUERY[query].matches(tree_obj.root_node)
        print(f"{len(matches)} matches:")
        return matches
