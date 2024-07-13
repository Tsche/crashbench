from dataclasses import dataclass
from functools import cached_property
from pprint import pprint
from pathlib import Path
from itertools import product
from typing import Any, Iterable, Optional, Protocol, runtime_checkable
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node

from .builtins import BUILTINS

CPP = Language(tree_sitter_cpp.language())

QUERY = {
    file.stem: CPP.query(file.read_text())
    for file in (Path(__file__).parent / "queries").iterdir()
    if file.suffix == ".lisp"
}


@runtime_checkable
class Unevaluated(Protocol):
    def evaluate(self, scope: "Scope"): ...

class PendingCall:
    def __init__(self, fnc, *arguments):
        self.fnc = fnc
        self.arguments = arguments

        self.value = None

    def evaluate_args(self, scope: "Scope"):
        for argument in self.arguments:
            if isinstance(argument, Unevaluated):
                yield argument.evaluate(scope)
            else:
                yield argument

    def evaluate(self, scope: "Scope"):
        if self.value is None:
            # evaluate only once, recall afterwards
            self.value = self.fnc(*list(self.evaluate_args(scope)))

        return self.value

    def __repr__(self):
        return f"{self.fnc.__name__}({', '.join(str(argument) for argument in self.arguments)})"


class Variable:
    def __init__(self, name: str):
        self.name = name

    @property
    def __name__(self):
        return self.name

    def __repr__(self):
        return self.name

    def evaluate(self, scope: "Scope"):
        return scope.evaluate_variable(self.name)


class Setting:
    def __init__(self, name: str, args: Node): ...


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent

        self.settings: dict[str, Any] = {}
        self.variables: dict[str, Any] = {}
        self.functions: dict[str, Lambda] = {}
        self.used: set[str] = set()

    def parse_argument_list(self, node: Node):
        for child in node.named_children:
            if child.type != "comment":
                yield self.parse_argument(child)

    def parse_argument(self, node: Node):
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
                    return float(text) if "." in text else int(text)
                except ValueError:
                    return text
            case "char_literal":
                assert node.named_child_count == 1
                return node.named_children[0].text.decode()
            case "concatenated_string":
                return "".join(
                    self.parse_argument(child) for child in node.named_children
                )
            case "true":
                return True
            case "false":
                return False
            case "null":
                return None
            case "call_expression":
                name_node = node.child_by_field_name("function")
                name = name_node.text.decode()
                arguments = list(
                    self.parse_argument_list(node.child_by_field_name("arguments"))
                )
                function = self.get_function(name)
                assert function is not None, f"No function {name} found"
                return PendingCall(function, *arguments)

            case "identifier":
                ident = node.text.decode()
                if fnc := self.get_function(ident):
                    # evaluate functions right away
                    return fnc

                return Variable(ident)
            case _:
                raise ValueError(f"Unexpected node type {node.type}")

    def parse_attr_node(self, node: Node):
        assert node.type == "attributed_statement", f"Wrong type: {node.type}"
        assert node.named_children[-1].type in (
            "expression_statement",
            "labeled_statement",
        ), f"Wrong type: {node.named_children[-1].type}"

        remove = False

        if node.named_children[-1].type == "expression_statement":
            # regular attribute
            if node.named_child_count == 2:
                # only one attribute => independent
                remove |= self.parse_attribute(node.named_children[0].named_children[0])
            else:
                # dependent attributes
                # TODO reuse VariableGroup pattern
                for child in node.named_children[:-1]:
                    assert child.type == "attribute_declaration"
                    remove |= self.parse_attribute(child.named_children[0])
        elif node.named_children[-1].type == "labeled_statement":
            # attribute using
            assert (
                node.has_error
            ), "Node doesn't have an error. Did tree-sitter-cpp get fixed?"

            for _, match in QUERY["using_attr"].matches(node):
                assert "using" in match
                assert "name" in match
                name = match["name"].text.decode()

                if "value" in match:
                    # print(f"{match['name'].text.decode()} = {match['value'].text.decode()}")
                    self.add_variable(name, self.parse_argument(match["value"]))
                    remove |= True
                elif "function" in match:
                    self.add_function(name, Lambda(name, match["function"], self))
                    remove |= True
                    # print(f"{match['name'].text.decode()} = {match['function'].text.decode()}")
                # print(match)
            # print(node)
        return remove

    def parse_attribute(self, node: Node):
        assert node.type == "attribute", f"Wrong type: {node.type}"

        if node.named_children[-1].type != "argument_list":
            # no args, currently not used
            return False
        args = node.named_children[-1]
        name = node.child_by_field_name("name").text.decode()

        if prefix := node.child_by_field_name("prefix"):
            # metavar
            varname = prefix.text.decode()
            if name not in BUILTINS and name not in self.functions:
                print(f"skipped {varname}: {name} is not a known function or builtin")
                return False

            arguments = list(self.parse_argument_list(args))
            if name == "var" and len(arguments) == 1:
                # special case var(x) with arity of one
                self.add_variable(varname, arguments[0])
                return True

            self.add_variable(varname, PendingCall(self.get_function(name), *arguments))
        else:
            # setting
            if name == "use":
                self.parse_use(args)

            # TODO check if name matches a known setting
            self.settings[name] = Setting(name, args)
        return True

    def parse_use(self, args: Node):
        arguments = list(self.parse_argument_list(args))
        for variable in arguments:
            assert isinstance(
                variable, Variable
            ), f"Wrong argument type. All args must be variables"
            self.set_used(variable.name)

    def set_used(self, variable: str):
        assert variable is not None
        assert self.get_variable(
            variable
        ), f"Unrecognized variable {variable} marked used"

        self.used.add(variable)

    def add_variable(self, variable: str, value: Any):
        assert variable not in self.variables, f"Overwriting variable {variable}"
        self.variables[variable] = value

    def add_function(self, function: str, value: Any):
        self.functions[function] = value

    def get_function(self, function: str):
        if function in BUILTINS:
            return BUILTINS[function]

        if function in self.functions:
            return self.functions[function]

        if self.parent:
            return self.parent.get_function(function)

    def get_variable(self, variable: str):
        if value := self.variables.get(variable, None):
            return value

        if self.parent is not None:
            return self.parent.get_variable(variable)

    def evaluate_variable(self, variable: str | Unevaluated):
        if isinstance(variable, Unevaluated):
            return (
                variable.evaluate(self)
                if isinstance(variable, Unevaluated)
                else variable
            )

        if (value := self.variables.get(variable, None)) is not None:
            return value.evaluate(self) if isinstance(value, Unevaluated) else value

        if self.parent is not None:
            return self.parent.evaluate_variable(variable)

    def add_setting(self, setting: str, value: Any):
        # it's okay to overwrite settings
        self.settings[setting] = value


class Lambda:
    def __init__(self, name: str, node: Node, scope: Scope):
        self.name = name
        *args, function = list(self.parse_nested_comma(node))
        self.arg_names = args

        parse_scope = Scope(scope)
        parse_scope.add_function(name, self)

        self.function = scope.parse_argument(function)
        self.scope = scope
        assert isinstance(self.function, Unevaluated)

    def parse_nested_comma(self, node: Node):
        assert node.type == "comma_expression"
        left = node.child_by_field_name("left")
        assert left.type == "identifier", "Expected an identifier"
        yield left.text.decode()

        right = node.child_by_field_name("right")
        if right.type == "comma_expression":
            yield from self.parse_nested_comma(right)
        else:
            yield right

    @property
    def __name__(self):
        return str(self.function)

    @property
    def definition(self):
        return f"{self.name} {', '.join(self.arg_names)} = {self.function!s}"

    def __repr__(self):
        return self.name

    def __call__(self, *args):
        assert (
            len(args) == len(self.arg_names)
        ), f"Invalid amount of arguments given, expected {len(self.arg_names)}, got {len(args)}"
        function_scope = Scope(self.scope)
        # function_scope.add_function(self.name, self)
        for name, value in zip(self.arg_names, args):
            function_scope.add_variable(name, value)

        return self.function.evaluate(function_scope)


@dataclass
class Range:
    start_byte: int
    end_byte: int

    def __contains__(self, element):
        assert self.end_byte > self.start_byte, "Invalid range"
        return self.start_byte <= element <= self.end_byte


class SkipList(list[Range]):
    def __contains__(self, element):
        return any(element in value for value in self)


class Test(Scope):
    def __init__(self, node: dict[str, Node], parent: Optional[Scope] = None):
        super().__init__(parent)
        self.kind = node["kind"].text.decode()
        assert self.kind in ("benchmark", "test")

        self.name = node["test_name"].text.decode()
        self.head = node["test_head"]
        self.start_byte = node["test_head"].start_byte
        self.end_byte = node["code"].end_byte
        self.node = node

        # note that parse_attr_node has side effects!
        self.skip_list = SkipList(
            Range(attr.start_byte, attr.end_byte)
            for attr in node.get("attr_node", [])
            if self.parse_attr_node(attr)
        )
        self.find_metavar_uses(node["code"])

    def find_metavar_uses(self, node: Node):
        for _, match in QUERY["identifier"].matches(node):
            start_byte = match["ident"].start_byte
            if start_byte in self.skip_list:
                continue

            ident = match["ident"].text.decode()
            if self.get_variable(ident) is not None:
                self.used.add(ident)

    @property
    def code(self) -> list[int]:
        code_node: Node = self.node["code"]
        text = b""
        last = 0
        for skip in self.skip_list:
            text += code_node.text[last : skip.start_byte - code_node.start_byte]
            last = skip.end_byte - code_node.start_byte

            if code_node.text[last] == ord("\n"):
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

    @cached_property
    def evaluated(self) -> dict[str, list[Any]]:
        variables = {}
        for name in self.used:
            variable = self.evaluate_variable(name)
            if isinstance(variable, Iterable) and not isinstance(variable, str):
                # force evaluation of generators
                variables[name] = list(variable)
                continue

            # convert scalars to lists
            variables[name] = [variable]
        return variables

    @property
    def runs(self) -> list[list[Any]]:
        def expand(name, var):
            for value in var:
                yield (name, value)

        expanded = [expand(name, vars) for name, vars in self.evaluated.items()]
        return list(dict(run) for run in product(*expanded))

class TranslationUnit(Scope):
    def __init__(self, source_path: Path):
        super().__init__()

        self.source_path = source_path
        self.raw_source = source_path.read_bytes()
        self.tests: list[Test] = []

        parser = Parser(CPP)
        tree = parser.parse(self.raw_source)
        self.skip_list: list[Range | Test] = []
        for _, node in QUERY["test"].matches(tree.root_node):
            if "kind" in node and "code" in node:
                # test
                test = Test(node, self)
                self.tests.append(test)
                self.skip_list.append(test)
            else:
                # metavar or config
                if "using" in node or "attr" in node:
                    for attr in node.get("attr_node", []):
                        if self.parse_attr_node(attr):
                            self.skip_list.append(Range(attr.start_byte, attr.end_byte))

    @cached_property
    def source(self):
        processed_source = ""
        last = 0
        for skip in self.skip_list:
            processed_source += self.raw_source[last : skip.start_byte].decode()
            if isinstance(skip, Test):
                processed_source += f"#ifdef {skip.name.upper()}\n"

                processed_source += skip.code
                processed_source += "\n"
                processed_source += "#endif"
            last = skip.end_byte

        processed_source += self.raw_source[last:].decode()
        return processed_source

    def __str__(self):
        def stringify_functions(dictionary):
            return "\n      ".join(fnc.definition for fnc in dictionary.values())

        lines = []
        lines.append("Variables:")
        lines.append("  Global:")
        lines.append(f"    Variables: {self.variables}")
        lines.append("    Functions: ")
        if self.functions:
            lines.append(f"      {stringify_functions(self.functions)}")
        for test in self.tests:
            variables = {
                name: str(variable) for name, variable in test.variables.items()
            }
            lines.append(f"  {test.name}:")
            lines.append(f"    Variables: {variables}")
            lines.append(f"    Functions:")
            if test.functions:
                lines.append(f"      {stringify_functions(test.functions)}")
            lines.append(f"    Used: {test.used}")

        return '\n'.join(lines)


def parse(source_path: Path):
    source = TranslationUnit(source_path)
    # source.parse()
    return source.source, source.tests


def print_tree(source: Path, query=None):
    parser = Parser(CPP)
    tree_obj = parser.parse(source.read_bytes())
    if query is None:
        print(str(tree_obj.root_node))
    else:
        matches = QUERY[query].matches(tree_obj.root_node)
        print(f"{len(matches)} matches:")
        pprint(matches)