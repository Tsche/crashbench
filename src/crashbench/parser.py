import contextlib
from dataclasses import dataclass
from functools import cached_property
from colorama import Fore, Back, Style
from pprint import pprint
from pathlib import Path
from itertools import product
from typing import Any, Iterable, Optional, Protocol, runtime_checkable
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node

from .builtins import BUILTINS
from .settings import Settings
from .exceptions import ParseError, UndefinedError
from .util import get_closest_match

CPP = Language(tree_sitter_cpp.language())

QUERY = {
    file.stem: CPP.query(file.read_text())
    for file in (Path(__file__).parent / "queries").iterdir()
    if file.suffix == ".lisp"
}

namespace_blacklist = ["std", "gnu", "clang", "gsl", "msvc", "riscv"]
setting_blacklist = [
    "carries_dependency",
    "deprecated",
    "fallthrough",
    "likely",
    "unlikely",
    "maybe_unused",
    "nodiscard",
    "noreturn",
    "no_unique_address",
    "assume",
    "indeterminate",
]


def decorate(
    message: str,
    fg: Optional[Fore] = None,
    bg: Optional[Back] = None,
    style: Optional[Style] = None,
):
    out = []
    if fg:
        out.append(fg)
    if bg:
        out.append(bg)
    if style:
        out.append(style)
    out.extend((message, Style.RESET_ALL))
    return "".join(out)


class DiagnosticLevel:
    @dataclass
    class Level:
        label: str
        color: Fore

        def format(
            self, filename: Optional[str], row: int, column: int, message: str
        ) -> str:
            return f"{filename or '<unknown>'}:{row}:{column}: {self.color}{self.label}:{Fore.RESET} {message}"

        def colored(self, message: str):
            return decorate(message, fg=self.color)

    WARNING = Level("warning", Fore.YELLOW)
    ERROR = Level("error", Fore.RED)


@runtime_checkable
class Unevaluated(Protocol):
    def evaluate(self, scope: "Scope"): ...


class PendingCall:
    def __init__(self, fnc, args, kwargs):
        self.fnc = fnc
        self.args = args
        self.kwargs = kwargs
        self.value = None

    def evaluate_arg(self, scope, argument):
        if isinstance(argument, Unevaluated):
            return argument.evaluate(scope)
        else:
            return argument

    def evaluate(self, scope: "Scope"):
        if self.value is None:
            # evaluate only once, recall afterwards
            positional_args = [
                self.evaluate_arg(scope, argument) for argument in self.args
            ]
            keyword_args = {
                key: self.evaluate_arg(scope, argument)
                for key, argument in self.kwargs.items()
            }
            self.value = self.fnc(*positional_args, **keyword_args)

        return self.value

    def __repr__(self):
        args = ", ".join(str(argument) for argument in self.args)
        kwargs = ", ".join(f"{key}={value}" for key, value in self.kwargs.items())
        argument_list = f"{args}{', ' if len(kwargs) and len(args) else ''}{kwargs}"
        return f"{self.fnc.__name__}({argument_list})"


class Variable:
    def __init__(self, name: str):
        self.name = name
        self.__name__ = name

    def __repr__(self):
        return self.name

    def evaluate(self, scope: "Scope"):
        return scope.evaluate_variable(self.name)


def parse_constant(node: Node):
    match node.type:
        case "string_literal":
            if node.named_child_count == 0:
                return ""

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
            return "".join(parse_constant(child) for child in node.named_children)
        case "true":
            return True
        case "false":
            return False
        case "null":
            return None
        case _:
            raise ParseError(f"Unexpected node type {node.type}", node)


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent

        self.settings = Settings(parent.settings if parent else None)
        self.variables: dict[str, Any] = {}
        self.functions: dict[str, Lambda] = {}
        self.used: set[str] = set()

    @property
    def all_variables(self):
        yield from self.variables.keys()
        if self.parent:
            yield from self.parent.variables.keys()

    def emit_diagnostic(self, level: DiagnosticLevel.Level, message: str, node: Node):
        if self.parent is not None:
            # forward up to reach TU scope
            return self.parent.emit_diagnostic(level, message, node)

        # if we ended up here, something went wrong. Emit the diagnostic without context
        start_row = node.range.start_point.row
        start_column = node.range.start_point.column
        # unfortunately we do not have the file name available here
        diagnostic = level.format(None, start_row, start_column, message)
        print(diagnostic)

    def emit_warning(self, message: str, node: Node):
        self.emit_diagnostic(DiagnosticLevel.WARNING, message, node)

    def emit_error(self, message: str, node: Node):
        self.emit_diagnostic(DiagnosticLevel.ERROR, message, node)

    def parse_argument_list(self, node: Node):
        args = []
        kwargs = {}

        for child in node.named_children:
            if child.type == "comment":
                continue

            if child.type == "assignment_expression":
                lhs = child.child_by_field_name("left")
                if lhs.type != "identifier":
                    raise ParseError("Keyword argument name must be identifier", lhs)

                key = lhs.text.decode()
                if key in kwargs:
                    raise ParseError(f"Duplicate keyword argument {key}", lhs)

                rhs = child.child_by_field_name("right")
                kwargs[key] = self.parse_argument(rhs)
            elif kwargs:
                raise ParseError(
                    "Positional arguments may not follow keyword arguments", child
                )
            else:
                args.append(self.parse_argument(child))

        # TODO return the used Node for every argument as well for better diagnostics
        return args, kwargs

    def parse_argument(self, node: Node):
        with contextlib.suppress(ParseError):
            return parse_constant(node)

        match node.type:
            case "call_expression":
                name_node = node.child_by_field_name("function")
                argument_list = node.child_by_field_name("arguments")
                return self.parse_call(name_node, argument_list)
            case "identifier":
                ident = node.text.decode()
                if fnc := self.get_function(ident):
                    # evaluate functions right away
                    return fnc

                return Variable(ident)
            # TODO kw args via assignment
            case _:
                raise ParseError(f"Unexpected node type {node.type}", node)

    def parse_call(self, name_node: Node, argument_list: Node):
        name = name_node.text.decode()
        if not (fnc := self.get_function(name)):
            raise UndefinedError(
                f"Undefined function `{name}` used", name_node, name, self
            )

        args, kwargs = self.parse_argument_list(argument_list)

        if name == "var":
            if len(kwargs) != 0:
                raise ParseError("var does not take keyword arguments", argument_list)
            if len(args) == 0:
                raise ParseError("var needs at least one argument", argument_list)
            elif len(args) == 1:
                # special case var(x) with arity of one
                # ie [[metavar::var(12)]]
                return args[0]

        return PendingCall(fnc, *args, **kwargs)

    def parse_attr_node(self, node: Node):
        assert node.type == "attributed_statement", f"Wrong type: {node.type}"
        assert node.named_children[-1].type in (
            "expression_statement",
            "labeled_statement",
        ), f"Wrong type: {node.named_children[-1].type}"

        remove = False

        if node.named_children[-1].type == "expression_statement":
            # regular attribute
            # [[identifier]], [[identifier(foo)]] etc
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
            # [[using ident1: ident2]], [[using ident1: ident2, ident3]], [[using ident1: ident2(foo)]] and so on
            assert (
                node.has_error
            ), "Node doesn't have an error. Did tree-sitter-cpp get fixed?"

            for _, match in QUERY["using_attr"].matches(node):
                assert "using" in match
                assert "name" in match
                name = match["name"].text.decode()

                if "value" in match:
                    self.add_variable(name, self.parse_argument(match["value"]))
                    remove |= True
                elif "function" in match:
                    self.add_function(name, Lambda(name, match["function"], self))
                    remove |= True
        return remove

    def parse_attribute(self, node: Node):
        assert node.type == "attribute", f"Wrong type: {node.type}"

        if node.named_children[-1].type != "argument_list":
            # no args, currently not used
            # for example: [[foo]]
            return False

        argument_list = node.named_children[-1]
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode()

        if prefix := node.child_by_field_name("prefix"):
            # has namespace prefix => metavar
            # [[metavar::generator(foo)]]
            varname = prefix.text.decode()
            if varname in namespace_blacklist:
                # variables matching one of the builtin attribute namespace names
                # are disallowed, do not attempt to parse them
                return False

            self.add_variable(varname, self.parse_call(name_node, argument_list))
            return True
        else:
            # no namespace prefix => setting or weak builtin
            # ie [[use(foo)]], [[language("c++")]]

            if name in setting_blacklist:
                # do not attempt to parse builtin attributes
                return False

            args, kwargs = self.parse_argument_list(argument_list)
            if name == "use":
                # special case [[use(ident)]]
                if len(args) == 0:
                    raise ParseError("Use needs at least one argument", argument_list)
                if len(kwargs) != 0:
                    raise ParseError(
                        f"Unrecognized keyword arguments {kwargs}", argument_list
                    )
                for variable in args:
                    if not isinstance(variable, Variable):
                        raise ParseError(
                            "Wrong argument type. All args must be variables",
                            argument_list,
                        )

                    if not self.get_variable(variable.name):
                        raise UndefinedError(
                            f"Unrecognized variable {variable} marked used",
                            argument_list,
                            name,
                            self,
                        )

                    self.used.add(variable)
                return True

            self.settings.add(name, args, kwargs)
        return True

    def add_variable(self, variable: str, value: Any):
        if variable in self.variables:
            # TODO parse context
            raise ParseError(f"Redefining {variable} is not allowed", None)

        if self.parent and variable in self.parent.all_variables:
            self.emit_warning(
                f"definition of `{decorate(variable, style=Style.BRIGHT)}` shadows global definition",
                None,
            )

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
        if self.kind not in ("benchmark", "test"):
            raise ParseError(f"Invalid test kind {self.kind} specified", node["kind"])

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

        try:
            self._parse()
        except ParseError as exc:
            self.emit_error(exc.message, exc.node)
            raise SystemExit(1)
        except UndefinedError as exc:
            if similar := get_closest_match(exc.name, exc.scope.all_variables):
                # add suggestions
                start_column = exc.node.range.start_point.column
                decorated_name = decorate(similar, fg=Fore.GREEN, style=Style.BRIGHT)
                self.emit_error(
                    f"{exc.message}; did you mean `{decorated_name}`?", exc.node
                )
                print(f"{' '*5}| {' '*start_column}{decorated_name}")
            else:
                # only print the error if there are no close matches
                self.emit_error(exc.message, exc.node)

            raise SystemExit(1)

    def _parse(self):
        parser = Parser(CPP)
        tree = parser.parse(self.raw_source)
        self.skip_list: list[Range | Test] = []
        for _, node in QUERY["test"].matches(tree.root_node):
            if "kind" in node and "code" in node:
                # test
                test = Test(node, self)
                self.tests.append(test)
                self.skip_list.append(test)
            elif "using" in node or "attr" in node:
                # metavar or config
                for attr in node.get("attr_node", []):
                    if self.parse_attr_node(attr):
                        self.skip_list.append(Range(attr.start_byte, attr.end_byte))

    @cached_property
    def lines(self):
        return self.raw_source.decode().splitlines()

    def emit_diagnostic(self, level: DiagnosticLevel.Level, message: str, node: Node):
        if node is None:
            print(level.format(None, -1, -1, message))
            return

        start_row = node.range.start_point.row
        end_row = node.range.end_point.row
        start_column = node.range.start_point.column
        end_column = node.range.end_point.column

        print(level.format(self.source_path.name, start_row, start_column, message))
        if end_row - start_row == 0:
            line = self.lines[start_row]
            line = (
                line[:start_column]
                + level.colored(line[start_column:end_column])
                + line[end_column:]
            )

            squiggle_amount = max(end_column - start_column - 1, 0)
            squiggles = level.colored("^" + "~" * (squiggle_amount))

            print(f"{start_row:<4} | {line}")
            print(f"{' '*5}| {' '*start_column}{squiggles}")
            return

        for idx in range(end_row - start_row + 1):
            line = self.lines[start_row + idx]
            if idx == 0:
                error_line = " " * start_column + level.colored(
                    "^" + "~" * (len(line) - start_column - 2)
                )
                line = line[:start_column] + level.colored(line[start_column:])

            elif idx == end_row - start_row:
                error_line = level.colored("~" * end_column)
                line = level.colored(line[:end_column]) + line[end_column:]

            print(f"{start_row + idx:<4} | {line}")
            print(f"{' ' * 5}| {error_line}")

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
        lines.append("    Functions:")
        if self.functions:
            lines.append(f"      {stringify_functions(self.functions)}")
        lines.append("    Settings:")

        lines.append("      " + "\n      ".join(str(self.settings).split("\n")))
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

        return "\n".join(lines)


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
