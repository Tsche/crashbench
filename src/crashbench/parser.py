import contextlib
import difflib
from sys import stderr
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
            return f"{filename or '<unknown>'}:{row}:{column}: {self.color}{self.label}:{Fore.RESET} {message}\n"

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

    def evaluate(self, scope: "Scope"):
        positional_args = [
            scope.evaluate(argument) for argument in self.args
        ]
        keyword_args = {
            key: scope.evaluate(argument)
            for key, argument in self.kwargs.items()
        }

        return self.fnc(*positional_args, **keyword_args)

    def __repr__(self):
        args = ", ".join(str(argument) for argument in self.args)
        kwargs = ", ".join(f"{key}={value}" for key, value in self.kwargs.items())
        argument_list = f"{args}{', ' if kwargs and args else ''}{kwargs}"
        return f"{self.fnc.__name__}({argument_list})"


class Variable:
    def __init__(self, name: str):
        self.name = name
        self.__name__ = name

    def __repr__(self):
        return self.name

    def evaluate(self, scope: "Scope"):
        return scope.evaluate(scope.get_variable(self.name))


class Conditional:
    def __init__(self, condition: Node, true_branch: Node, false_branch: Node):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return f"if {self.condition} then {self.true_branch} else {self.false_branch}"

    def evaluate(self, scope: "Scope"):
        if scope.evaluate(self.condition):
            return scope.evaluate(self.true_branch)
        return scope.evaluate(self.false_branch)


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
        self.used: set[str] = set()

    @property
    def all_variables(self):
        yield from self.variables.keys()
        if self.parent:
            yield from self.parent.all_variables

    def nearest_match(self, name: str, is_function: Optional[bool] = None):
        unique_canonicalized = {
            variable.upper(): variable for variable in self.all_variables
        }
        if close_matches := difflib.get_close_matches(
            name.upper(), unique_canonicalized, 1
        ):
            if is_function is None:
                # we don't know whether we're looking for a function or variable
                # simply return the closest match
                return unique_canonicalized[close_matches[0]]

            decanonicalized = [unique_canonicalized[match] for match in close_matches]
            # return first name that matches the category (function/var)
            with contextlib.suppress(StopIteration):
                return next(
                    variable
                    for variable in decanonicalized
                    if is_function and callable(self.get_variable(variable))
                )
        return None

    def evaluate(self, obj):
        return obj.evaluate(self) if isinstance(obj, Unevaluated) else obj

    def emit_diagnostic(self, level: DiagnosticLevel.Level, message: str, node: Node):
        if self.parent is not None:
            # forward up to reach TU scope
            return self.parent.emit_diagnostic(level, message, node)

        # if we ended up here, something went wrong. Emit the diagnostic without context
        start_row = node.range.start_point.row
        start_column = node.range.start_point.column
        # unfortunately we do not have the file name available here
        stderr.write(level.format(None, start_row, start_column, message))

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
                # if var := self.get_variable(ident):
                #     # alias - do not wrap in Variable
                #     return var

                #return self.get_variable(ident) or 
                return Variable(ident)
            # TODO kw args via assignment
            case _:
                raise ParseError(f"Unexpected node type {node.type}", node)

    def parse_call(self, name_node: Node, argument_list: Node):
        name = name_node.text.decode()
        if (fnc := self.get_variable(name)) is None:
            raise UndefinedError(
                f"Undefined function `{name}` used",
                name_node,
                name,
                self,
                is_function=True,
            )

        if not callable(fnc):
            raise ParseError(f"`{name}` is not callable.", name_node)
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

        elif name == "if":
            # special case conditionals to allow lazy evaluation
            if len(kwargs) != 0:
                raise ParseError("if does not take keyword arguments", argument_list)
            if len(args) != 3:
                raise ParseError(
                    "`if` expects exactly 3 positional arguments", argument_list
                )
            return Conditional(
                condition=args[0], true_branch=args[1], false_branch=args[2]
            )

        return PendingCall(fnc, args, kwargs)

    def find_error_node(self, node: Node):
        assert (
            node.has_error
        ), "Cannot find error node within a node that isn't erroneous"

        for child in node.named_children:
            if child.has_error:
                return self.find_error_node(child)

        # no named children were erroneous -> can't descend further, return current node
        return node

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

            if node.has_error:
                raise ParseError("Invalid syntax", self.find_error_node(node) or node)

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
                name_node = match["name"]
                name = match["name"].text.decode()

                if "value" in match:
                    self.add_variable(name_node, self.parse_argument(match["value"]))
                    remove |= True
                elif "function" in match:
                    self.add_variable(name, Lambda(name, match["function"], self))
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

            self.add_variable(prefix, self.parse_call(name_node, argument_list))
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

                    if self.get_variable(variable.name) is None:
                        raise UndefinedError(
                            f"Unrecognized variable {variable} marked used",
                            argument_list,
                            name,
                            self,
                            is_function=False,
                        )

                    self.used.add(variable)

                return True

            self.settings.add(name, args, kwargs)
        return True

    def add_variable(self, variable: Node, value: Any):
        variable_name = (
            variable.text.decode() if isinstance(variable, Node) else variable
        )
        if variable_name in self.variables:
            # TODO parse context
            raise ParseError(f"Redefining {variable} is not allowed", variable)

        if self.parent and variable_name in self.parent.all_variables:
            self.emit_warning(
                f"definition of `{decorate(variable_name, style=Style.BRIGHT)}` shadows global definition",
                variable,
            )

        self.variables[variable_name] = value

    def get_variable(self, variable: str):
        if variable in self.variables:
            return self.variables[variable]

        if self.parent is not None:
            return self.parent.get_variable(variable)


class Lambda:
    def __init__(self, name: str, node: Node, scope: Scope):
        self.name = name
        self.__name__ = name

        *args, function = list(self.parse_nested_comma(node))
        # TODO: capture actual Nodes
        self.arg_names = args

        self.scope = Scope(scope)
        self.scope.variables[self.name] = self

        name_node = function.child_by_field_name("function")
        argument_list = function.child_by_field_name("arguments")

        self.function = self.scope.parse_call(name_node, argument_list)

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
    def definition(self):
        return f"{self.name} {', '.join(self.arg_names)} = {self.function!s}"

    def __repr__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        assert (
            len(args) == len(self.arg_names)
        ), f"Invalid amount of arguments given, expected {len(self.arg_names)}, got {len(args)}"

        function_scope = Scope(self.scope)
        for name, value in zip(self.arg_names, args):
            function_scope.add_variable(name, self.scope.evaluate(value))

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
        for var in self.used:
            # TODO
            variable = self.evaluate(var)
            if isinstance(variable, Iterable) and not isinstance(variable, str):
                # force evaluation of generators
                variables[var.name] = list(variable)
                continue

            # convert scalars to lists
            variables[var.name] = [variable]
        return variables

    @property
    def runs(self) -> list[list[Any]]:
        def expand(name, var):
            for value in var:
                yield (name, value)

        expanded = [expand(name, vars) for name, vars in self.evaluated.items()]
        return [dict(run) for run in product(*expanded)]


class TranslationUnit(Scope):
    def __init__(self, source_path: Path):
        super().__init__()

        self.source_path = source_path
        self.raw_source = source_path.read_bytes()
        self.tests: list[Test] = []

        # builtins shall be treated as globals
        self.variables |= BUILTINS

        try:
            self._parse()
        except ParseError as exc:
            self.emit_error(exc.message, exc.node)
            raise SystemExit(1)
        except UndefinedError as exc:
            if similar := exc.scope.nearest_match(exc.name, exc.is_function):
                # add suggestions
                start_column = exc.node.range.start_point.column
                decorated_name = decorate(similar, fg=Fore.GREEN, style=Style.BRIGHT)
                self.emit_error(
                    f"{exc.message}; did you mean `{decorated_name}`?", exc.node
                )
                stderr.write(f"{' '*5}| {' '*start_column}{decorated_name}\n")
            else:
                # print only the error if there are no close matches
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
            stderr.write(level.format(None, -1, -1, message))
            return

        start_row = node.range.start_point.row
        end_row = node.range.end_point.row
        start_column = node.range.start_point.column
        end_column = node.range.end_point.column

        stderr.write(
            level.format(self.source_path.name, start_row, start_column, message)
        )
        if end_row - start_row == 0:
            line = self.lines[start_row]
            line = (
                line[:start_column]
                + level.colored(line[start_column:end_column])
                + line[end_column:]
            )

            squiggle_amount = max(end_column - start_column - 1, 0)
            squiggles = level.colored("^" + "~" * (squiggle_amount))

            stderr.write(f"{start_row:^5}| {line}\n")
            stderr.write(f"{' '*5}| {' '*start_column}{squiggles}\n")
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
            else:
                error_line = level.colored("~" * len(line))
                line = level.colored(line)

            stderr.write(f"{start_row + idx:^5}| {line}\n")
            stderr.write(f"{' ' * 5}| {error_line}\n")

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
        lines = []
        lines.append("Variables:")
        lines.append("  Global:")
        lines.append(f"    Variables: ")
        for variable, value in self.variables.items():
            if variable in BUILTINS:
                continue
            if isinstance(value, Lambda):
                lines.append(f"       {value.definition}")
            else:
                lines.append(f"       {variable} = {value}")
        # lines.append("    Functions:")
        # if self.functions:
        #     lines.append(f"      {stringify_functions(self.functions)}")
        lines.append("    Settings:")

        lines.append("      " + "\n      ".join(str(self.settings).split("\n")))
        for test in self.tests:
            variables = {
                name: str(variable) for name, variable in test.variables.items()
            }
            lines.append(f"  {test.name}:")
            lines.append(f"    Variables: {variables}")
            # lines.append(f"    Functions:")
            # if test.functions:
            #     lines.append(f"      {stringify_functions(test.functions)}")
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
