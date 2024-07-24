import contextlib
import difflib
from sys import stderr
from dataclasses import dataclass
from functools import cached_property
from colorama import Fore, Back, Style
from pprint import pprint
from pathlib import Path
from itertools import product
from typing import Any, Iterable, Optional, Protocol, Self, runtime_checkable
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node

from .builtins import BUILTINS, BINARY_OPERATORS, UNARY_OPERATORS
from .exceptions import ParseError, UndefinedError
from .compilers import compilers, is_valid_compiler

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
        positional_args = [scope.evaluate(argument) for argument in self.args]
        keyword_args = {key: scope.evaluate(argument) for key, argument in self.kwargs.items()}

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


class Settings:
    def __init__(self, parent: Optional['Settings'] = None):
        self.parent = parent

    def parse(self, name_node: Node, argument_list: Node):
        print(name_node.text.decode())
        print(argument_list.text.decode())
        return True


def parse_constant(node: Node):
    match node.type:
        case "string_literal":
            if node.named_child_count == 0:
                return ""

            assert node.named_child_count == 1
            assert node.named_children[0].text is not None, "Invalid text node"

            return node.named_children[0].text.decode()
        case "number_literal":
            # TODO integer literal suffixes
            # TODO hex, octal, binary literals
            # TODO separators
            assert node.text is not None, "Invalid text node"
            text = node.text.decode()
            try:
                return float(text) if "." in text else int(text)
            except ValueError:
                return text
        case "char_literal":
            assert node.named_child_count == 1
            assert node.named_children[0].text is not None, "Invalid text node"

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

        self.settings: Settings = Settings(parent.settings if parent else None)
        self.variables: dict[str, Any] = {}
        self.used: set[Variable] = set()

    @property
    def all_variables(self):
        yield from self.variables.keys()
        if self.parent:
            yield from self.parent.all_variables

    def nearest_match(self, name: str, is_function: Optional[bool] = None):
        unique_canonicalized = {variable.upper(): variable for variable in self.all_variables}
        if close_matches := difflib.get_close_matches(name.upper(), unique_canonicalized, 1):
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
                assert lhs is not None, "Could not get lhs of binary expression"

                if lhs.type != "identifier":
                    raise ParseError("Keyword argument name must be identifier", lhs)

                assert lhs.text is not None, "Invalid text node"
                key = lhs.text.decode()
                if key in kwargs:
                    raise ParseError(f"Duplicate keyword argument {key}", lhs)

                rhs = child.child_by_field_name("right")
                assert rhs is not None, "Could not get rhs of binary expression"
                kwargs[key] = self.parse_argument(rhs)
            elif kwargs:
                raise ParseError("Positional arguments may not follow keyword arguments", child)
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
                assert name_node is not None, "Could not parse function name"

                argument_list = node.child_by_field_name("arguments")
                assert argument_list is not None, "Could not parse argument list"

                return self.parse_call(name_node, argument_list)
            case "identifier":
                assert node.text is not None, "Invalid text node"
                ident = node.text.decode()
                return Variable(ident)

            case "unary_expression":
                assert node.child_count == 2

                op_node, argument = node.children
                assert op_node.text is not None, "Invalid text node"

                operation = op_node.text.decode()
                if operation not in UNARY_OPERATORS:
                    raise ParseError(f"Invalid unary operation `{operation}`", op_node)

                return PendingCall(UNARY_OPERATORS[operation], [self.parse_argument(argument)], {})

            case "binary_expression":
                assert node.child_count == 3

                lhs, op_node, rhs = node.children
                assert op_node.text is not None, "Invalid text node"

                operation = op_node.text.decode()
                assert op_node.text is not None, "Invalid text node"

                if operation not in BINARY_OPERATORS:
                    raise ParseError(f"Invalid binary operation `{operation}`", op_node)

                return PendingCall(
                    BINARY_OPERATORS[operation],
                    [self.parse_argument(lhs), self.parse_argument(rhs)],
                    {},
                )

            case "conditional_expression":
                condition = node.child_by_field_name("condition")
                assert condition is not None, "Could not parse condition"

                true_branch = node.child_by_field_name("consequence")
                assert true_branch is not None, "Could not parse consequence"

                false_branch = node.child_by_field_name("alternative")
                assert false_branch is not None, "Could not parse alternative"

                return Conditional(
                    self.parse_argument(condition),
                    self.parse_argument(true_branch),
                    self.parse_argument(false_branch),
                )

            case "parenthesized_expression":
                assert node.named_child_count == 1
                return self.parse_argument(node.named_children[0])

            case _:
                raise ParseError(f"Unexpected node type {node.type}", node)

    def parse_call(self, name_node: Node, argument_list: Node):
        assert name_node.text is not None, "Invalid text node"
        name = name_node.text.decode()
        if (fnc := self.get_variable(name)) is None:
            raise UndefinedError(f"Undefined function `{name}` used", name_node, name, self, is_function=True)

        if not callable(fnc):
            raise ParseError(f"`{name}` is not callable.", name_node)
        args, kwargs = self.parse_argument_list(argument_list)

        if name in ("var", "return"):
            if len(kwargs) != 0:
                raise ParseError(
                    f"{name} does not take keyword arguments", argument_list
                )
            if len(args) == 0:
                raise ParseError(f"{name} needs at least one argument", argument_list)
            elif len(args) == 1:
                # special case var(x) and return(x) with arity of one
                # ie [[metavar::var(12)]]
                return args[0]

            if name == "return":
                raise ParseError("return must have exactly one argument", argument_list)
        elif name == "if":
            # special case conditionals to allow lazy evaluation
            if len(kwargs) != 0:
                raise ParseError("if does not take keyword arguments", argument_list)
            if len(args) != 3:
                raise ParseError("`if` expects exactly 3 positional arguments", argument_list)
            return Conditional(
                condition=args[0], true_branch=args[1], false_branch=args[2]
            )
        return PendingCall(fnc, args, kwargs)

    def find_error_node(self, node: Node):
        assert node.has_error, "Cannot find error node within a node that isn't erroneous"

        for child in node.named_children:
            if child.has_error:
                return self.find_error_node(child)

        # no named children were erroneous -> can't descend further, return current node
        return node

    def parse_attr_node(self, node: Node):
        assert node.type == "attributed_statement", f"Wrong type: {node.type}"
        assert node.named_children[-1].type in ("expression_statement", "labeled_statement"), \
            f"Wrong type: {node.named_children[-1].type}"

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
            assert node.has_error, "Node doesn't have an error. Did tree-sitter-cpp get fixed?"

            for _, match in QUERY["using_attr"].matches(node):
                assert "using" in match
                name_node = match.get("name")

                assert name_node is not None, "Could not get name node"
                assert not isinstance(name_node, list), "More than one name node found"
                assert name_node.text is not None, "Invalid text node"
                name = name_node.text.decode()

                if "value" in match:
                    value = match.get("value")
                    assert value is not None, "Could not get value node"
                    assert not isinstance(value, list), "More than one value node found"

                    self.add_variable(name_node, self.parse_argument(value))
                    remove |= True
                elif "function" in match:
                    fnc = match.get("function")
                    assert fnc is not None, "Could not get function node"
                    assert not isinstance(fnc, list), "More than one function node found"
                    self.add_variable(name_node, Lambda(name, fnc, self))
                    remove |= True
        return remove

    def parse_attribute(self, node: Node):
        assert node.type == "attribute", f"Wrong type: {node.type}"
        argument_list = node.named_children[-1] if node.named_children[-1].type == "argument_list" else None
        name_node = node.child_by_field_name("name")
        assert name_node is not None, "Could not get name node"

        if prefix := node.child_by_field_name("prefix"):
            return self.parse_metavar(prefix, name_node, argument_list)

        return self.parse_setting(name_node, argument_list)

    def parse_setting(self, name_node: Node, argument_list: Optional[Node]):
        # no namespace prefix => setting or weak builtin
        # ie [[use(foo)]], [[language("c++")]]
        assert name_node.text, "Invalid text node"
        name = name_node.text.decode()

        if name in setting_blacklist:
            # do not attempt to parse builtin attributes
            return False

        if argument_list is None:
            # currently unused => abort if there is no argument list
            # ie [[foo]]
            return False

        if name == "use":
            # special case [[use(ident)]]
            self.parse_use_args(argument_list)
            return True

        return self.settings.parse(name_node, argument_list)

    def parse_use_args(self, argument_list: Node):
        args, kwargs = self.parse_argument_list(argument_list)
        if len(args) == 0:
            raise ParseError("Use needs at least one argument", argument_list)
        if len(kwargs) != 0:
            raise ParseError(f"Unrecognized keyword arguments {kwargs}", argument_list)

        for variable in args:
            if not isinstance(variable, Variable):
                raise ParseError("Wrong argument type. All args must be variables", argument_list)

            if self.get_variable(variable.name) is None:
                raise UndefinedError(f"Unrecognized variable {variable} marked used",
                                     argument_list, str(variable), self, is_function=False)

            self.used.add(variable)

    def parse_metavar(self, prefix: Node, name_node: Node, argument_list: Optional[Node]):
        # has namespace prefix => metavar
        # [[metavar::generator(foo)]]

        assert prefix.text is not None, "Invalid text node"
        varname = prefix.text.decode()
        if varname in namespace_blacklist:
            # variables matching one of the builtin attribute namespace names
            # are disallowed, do not attempt to parse them
            return False

        if is_valid_compiler(varname):
            # special case - prefix is a compiler
            # => this isn't a metavar, it's a compiler-specific setting or function
            if argument_list is None:
                raise ParseError("Compiler settings and metafunctions must have arguments", name_node)
            args, kwargs = self.parse_argument_list(argument_list)

            assert name_node.text, "Invalid text node"
            name = name_node.text.decode()

            # TODO
            print(f"{varname} {name} {args} {kwargs}")

            return True

        self.add_variable(prefix, self.parse_argument(name_node)
                          if argument_list is None else self.parse_call(name_node, argument_list))

        return True

    def add_variable(self, variable: Node, value: Any):
        assert variable.text is not None, "Invalid text node"
        variable_name: str = variable.text.decode()
        if variable_name in self.variables:
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

        *args, fnc = list(self.parse_nested_comma(node))
        self.arg_names = args

        self.scope = Scope(scope)
        self.scope.variables[self.name] = self

        name_node = fnc.child_by_field_name("function")
        argument_list = fnc.child_by_field_name("arguments")

        self.function = self.scope.parse_call(name_node, argument_list)

    def parse_nested_comma(self, node: Node):
        assert node.type == "comma_expression"
        left = node.child_by_field_name("left")

        assert left is not None, "Could not get lhs"
        assert left.type == "identifier", "Expected an identifier"
        yield left

        right = node.child_by_field_name("right")
        assert right is not None, "Could not get rhs"

        if right.type == "comma_expression":
            yield from self.parse_nested_comma(right)
        else:
            yield right

    @property
    def definition(self):
        return f"{self.name} {', '.join(arg.text.decode() for arg in self.arg_names)} = {self.function!s}"

    def __repr__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        assert len(args) == len(self.arg_names), \
            f"Invalid amount of arguments given, expected {len(self.arg_names)}, got {len(args)}"

        function_scope = Scope(self.scope)
        for name, value in zip(self.arg_names, args):
            function_scope.add_variable(name, self.scope.evaluate(value))

        return self.function.evaluate(function_scope)


@dataclass
class Range:
    start_byte: int
    end_byte: int

    def __contains__(self, element: int):
        assert self.end_byte > self.start_byte, "Invalid range"
        return self.start_byte <= element <= self.end_byte


class SkipList(list[Range]):
    def __contains__(self, element: object):
        if isinstance(element, int):
            return any(element in value for value in self)
        return super().__contains__(element)


class Test(Scope):
    def __init__(self, node: dict[str, Node], parent: Optional[Scope] = None):
        super().__init__(parent)
        kind_node = node.get("kind")
        assert kind_node is not None, "Could not parse test kind"
        assert kind_node.text is not None, "Invalid text node"

        self.kind = kind_node.text.decode()
        if self.kind not in ("benchmark", "test"):
            raise ParseError(f"Invalid test kind {self.kind} specified", kind_node)

        test_name = node.get("test_name")
        assert test_name is not None, "Could not parse test name"
        assert test_name.text is not None, "Invalid text node"
        self.name: str = test_name.text.decode()
        self.head: Node = node["test_head"]

        self.start_byte: int = node["test_head"].start_byte
        self.end_byte: int = node["code"].end_byte
        self.node = node

        attributes: list[Node] | Node = node.get("attr_node", [])
        if isinstance(attributes, Node):
            attributes = [attributes]

        # note that parse_attr_node has side effects!
        # TODO
        self.skip_list = SkipList(
            Range(attr.start_byte, attr.end_byte)
            for attr in attributes
            if self.parse_attr_node(attr)
        )
        self.find_metavar_uses(node["code"])

    def find_metavar_uses(self, node: Node):
        for _, match in QUERY["identifier"].matches(node):
            ident_node = match["ident"]
            assert not isinstance(ident_node, list), "Got more than one ident node"
            start_byte = ident_node.start_byte
            assert start_byte is not None, "Invalid node start byte"

            if start_byte in self.skip_list:
                continue

            assert ident_node.text is not None, "Invalid text node"
            ident = ident_node.text.decode()
            if self.get_variable(ident) is not None:
                # TODO check if not callable
                # TODO check if not builtin
                self.used.add(self.parse_argument(ident_node))

    @property
    def code(self) -> str:
        code_node: Node | None = self.node.get("code")
        assert code_node is not None, "Invalid code node"
        assert code_node.text is not None, "Invalid text node"

        text = b""
        last = 0
        for skip in self.skip_list:
            text += code_node.text[last: skip.start_byte - code_node.start_byte]
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
    def runs(self) -> list[dict[Any, Any]]:
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
                self.emit_error(f"{exc.message}; did you mean `{decorated_name}`?", exc.node)
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
                attributes = node.get("attr_node", [])
                if isinstance(attributes, Node):
                    attributes = [attributes]

                for attr in attributes:
                    if self.parse_attr_node(attr):
                        self.skip_list.append(Range(attr.start_byte, attr.end_byte))

    @cached_property
    def lines(self):
        return self.raw_source.decode().splitlines()

    def emit_diagnostic(self, level: DiagnosticLevel.Level, message: str, node: Node | None):
        if node is None:
            stderr.write(level.format(None, -1, -1, message))
            return

        start_row = node.range.start_point.row
        end_row = node.range.end_point.row
        start_column = node.range.start_point.column
        end_column = node.range.end_point.column

        stderr.write(level.format(self.source_path.name, start_row, start_column, message))
        if end_row - start_row == 0:
            line = self.lines[start_row]
            line = line[:start_column] + level.colored(line[start_column:end_column]) + line[end_column:]

            squiggle_amount = max(end_column - start_column - 1, 0)
            squiggles = level.colored("^" + "~" * (squiggle_amount))

            stderr.write(f"{start_row:^5}| {line}\n")
            stderr.write(f"{' '*5}| {' '*start_column}{squiggles}\n")
            return

        for idx in range(end_row - start_row + 1):
            line = self.lines[start_row + idx]
            if idx == 0:
                error_line = " " * start_column + level.colored("^" + "~" * (len(line) - start_column - 2))
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
            processed_source += self.raw_source[last: skip.start_byte].decode()
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
        # lines.append("    Settings:")

        # lines.append("      " + "\n      ".join(str(self.settings).split("\n")))
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
