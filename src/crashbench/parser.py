from collections import UserDict
import contextlib
import difflib
from dataclasses import dataclass
from functools import cached_property
import importlib
from itertools import product
from pathlib import Path
from pprint import pprint
from sys import stderr
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

import tree_sitter_cpp
from colorama import Fore, Style
from tree_sitter import Language, Node, Parser

from .builtins import BINARY_OPERATORS, BUILTINS, UNARY_OPERATORS
from .compilers import builtin, COMPILERS, COMPILER_BUILTINS, is_valid_compiler, Compiler
from .exceptions import ParseError, UndefinedError
from .util import proxy, decorated

# TODO:
#! - add .node to all possible parse results


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


def find_error_node(node: Node):
    assert node.has_error, "Cannot find error node within a node that isn't erroneous"

    for child in node.named_children:
        if child.has_error:
            return find_error_node(child)

    # no named children were erroneous -> can't descend further, return current node
    return node


class DiagnosticLevel:
    @dataclass
    class Level:
        label: str
        color: Fore

        def format(
            self, filename: Optional[Path], row: int, column: int, message: str
        ) -> str:
            return f"{filename.name if filename else '<unknown>'}:{row}:{column}: {self.color}{self.label}:{Fore.RESET} {message}\n"

        def colored(self, message: str):
            return decorated(message, fg=self.color)

    WARNING = Level("warning", Fore.YELLOW)
    ERROR = Level("error", Fore.RED)


class Diagnostics:
    def __init__(self, source: str, source_path: Optional[Path] = None):
        self.source_path = source_path
        self.source_lines = source.splitlines()

    def emit_diagnostic(self, level: DiagnosticLevel.Level, message: str, node: Optional[Node] = None):
        if node is None:
            # we didn't get a node - output without line/column information
            stderr.write(level.format(None, -1, -1, message))
            return

        start_row = node.range.start_point.row
        end_row = node.range.end_point.row
        start_column = node.range.start_point.column
        end_column = node.range.end_point.column

        stderr.write(level.format(self.source_path, start_row, start_column, message))
        if end_row - start_row == 0:
            line = self.source_lines[start_row]
            line = line[:start_column] + level.colored(line[start_column:end_column]) + line[end_column:]

            squiggle_amount = max(end_column - start_column - 1, 0)
            squiggles = level.colored("^" + "~" * (squiggle_amount))

            stderr.write(f"{start_row:^5}| {line}\n")
            stderr.write(f"{' '*5}| {' '*start_column}{squiggles}\n")
            return

        for idx in range(end_row - start_row + 1):
            line = self.source_lines[start_row + idx]
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

    def emit_warning(self, message: str, node: Optional[Node] = None):
        self.emit_diagnostic(DiagnosticLevel.WARNING, message, node)

    def emit_error(self, message: str, node: Optional[Node] = None):
        self.emit_diagnostic(DiagnosticLevel.ERROR, message, node)


class Scope(UserDict[str, Any]):
    def __init__(self, parent: Optional["Scope"] = None, data: Optional[dict[str, Any]] = None):
        super().__init__(data)
        self.parent = parent

    def nearest_match(self, name: str, is_function: Optional[bool] = None):
        unique_canonicalized = {variable.upper(): variable for variable in self.all}
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
                    if is_function and callable(self.get(variable))
                )
        return None

    @property
    def all(self):
        parent = self.parent.all if self.parent is not None else {}
        return {**parent, **self.data}

    @property
    def enclosing_scope(self):
        if self.parent is not None:
            yield from self.parent.all

    def add(self, key: str, value: Any) -> bool:
        overwritten = key in self
        super().__setitem__(key, value)
        return overwritten

    def get(self, key: str, default: Any = None):
        if key in self.data:
            return self.data.get(key, default)

        return default if self.parent is None else self.parent.get(key, default)

    def copy(self):
        return type(self)(parent=None if self.parent is None else self.parent.copy(), data=self.data.copy())

    # def __contains__(self, key: Any) -> bool:
    #     if key in self.data:
    #         return True
    #     return False if self.parent is None else key in self.parent

    def __getitem__(self, key: str):
        if key not in self:
            raise KeyError(f"Key {key} not contained in scope")
        return self.get(key)

    # leave __setitem__ as is

    def __delitem__(self, key: str):
        if key in self.data:
            del self.data[key]
            return

        if self.parent is not None:
            return self.parent.__delitem__(key)


@proxy
class ParseResult:
    def __init__(self, value: Any, node: Node):
        self.node = node
        self.value = value

    def evaluate(self, scope: Scope):
        return self.value


def get_node(obj: ParseResult | Any):
    return getattr(obj, "node", None)


class VariableScope(Scope):
    def evaluate(self, obj):
        return obj.evaluate(self) if isinstance(obj, Unevaluated) else obj

    def parse(self, node: Node):
        match node.type:
            case "string_literal":
                if node.named_child_count == 0:
                    return ParseResult("", node)

                assert node.named_child_count == 1
                assert node.named_children[0].text is not None, "Invalid text node"

                return ParseResult(node.named_children[0].text.decode(), node)

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
                # TODO ensure all children are string_literal or concatenated_string
                return "".join(self.parse(child) for child in node.named_children)

            case "true":
                return ParseResult(True, node)

            case "false":
                return ParseResult(False, node)

            case "null":
                return ParseResult(None, node)

            case "call_expression":
                name_node = node.child_by_field_name("function")
                assert name_node is not None, "Could not parse function name"

                argument_list = node.child_by_field_name("arguments")
                assert argument_list is not None, "Could not parse argument list"

                return self.parse_call(name_node, argument_list)

            case type_ if type_ in ("identifier", "namespace_identifier", "type_identifier"):
                assert node.text is not None, "Invalid text node"
                ident = node.text.decode()
                return Variable(ident, node)

            case "unary_expression":
                assert node.child_count == 2

                op_node, argument = node.children
                assert op_node.text is not None, "Invalid text node"

                operation = op_node.text.decode()
                if operation not in UNARY_OPERATORS:
                    raise ParseError(f"Invalid unary operation `{operation}`", op_node)

                return PendingCall(UNARY_OPERATORS[operation], [self.parse(argument)], {})

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
                    [self.parse(lhs), self.parse(rhs)],
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
                    self.parse(condition),
                    self.parse(true_branch),
                    self.parse(false_branch),
                )

            case "parenthesized_expression":
                assert node.named_child_count == 1
                return self.parse(node.named_children[0])

            case "initializer_list":
                # treat initializer lists as lists
                return [self.parse(child) for child in node.named_children]

            # TODO
            # case "field_expression":
            #     argument_node = node.child_by_field_name("argument")
            #     assert argument_node is not None
            #     argument = self.parse(argument_node)

            #     field_node = node.child_by_field_name("field")
            #     assert field_node is not None
            #     assert field_node.text is not None
            #     field = field_node.text.decode()

            #     return getattr(argument.evaluate(self) if hasattr(argument, "evaluate") else argument, field)

            case _:
                raise ParseError(f"Unexpected node type {node.type}", node)

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
                kwargs[key] = self.parse(rhs)
            elif kwargs:
                raise ParseError("Positional arguments may not follow keyword arguments", child)
            else:
                args.append(self.parse(child))

        return args, kwargs

    def parse_call(self, name_node: Node, argument_list: Node):
        assert name_node.text is not None, "Invalid text node"
        name = name_node.text.decode()
        if (fnc := self.get(name)) is None:
            raise UndefinedError(f"Undefined function `{name}` used", name_node, name, self, is_function=True)

        if not callable(fnc):
            raise ParseError(f"`{name}` is not callable.", name_node)
        args, kwargs = self.parse_argument_list(argument_list)

        if name in ("var", "return"):
            if len(kwargs) != 0:
                raise ParseError(
                    f"{name} does not take keyword arguments", get_node(kwargs[0]) or argument_list
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
                raise ParseError("if does not take keyword arguments", get_node(kwargs[0]) or argument_list)
            if len(args) != 3:
                raise ParseError("`if` expects exactly 3 positional arguments", argument_list)
            return Conditional(
                condition=args[0], true_branch=args[1], false_branch=args[2]
            )
        return PendingCall(fnc, args, kwargs)


@runtime_checkable
class Unevaluated(Protocol):
    def evaluate(self, scope: VariableScope): ...


class PendingCall:
    def __init__(self, fnc, args, kwargs):
        self.fnc = fnc
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, scope: VariableScope):
        positional_args = [scope.evaluate(argument) for argument in self.args]
        keyword_args = {key: scope.evaluate(argument) for key, argument in self.kwargs.items()}

        return self.fnc(*positional_args, **keyword_args)

    def __repr__(self):
        args = ", ".join(str(argument) for argument in self.args)
        kwargs = ", ".join(f"{key}={value}" for key, value in self.kwargs.items())
        argument_list = f"{args}{', ' if kwargs and args else ''}{kwargs}"
        return f"{self.fnc.__name__}({argument_list})"


class Variable:
    def __init__(self, name: str, node: Node):
        self.name = name
        self.__name__ = name
        self.node = node

    def __repr__(self):
        return self.name

    def evaluate(self, scope: VariableScope):
        return scope.evaluate(scope.get(self.name))


class Conditional:
    def __init__(self, condition: Node, true_branch: Node, false_branch: Node):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return f"if {self.condition} then {self.true_branch} else {self.false_branch}"

    def evaluate(self, scope: VariableScope):
        if scope.evaluate(self.condition):
            return scope.evaluate(self.true_branch)
        return scope.evaluate(self.false_branch)


class CompilerSetting:
    def __init__(self, compiler: type[Compiler], settings: Optional[dict[str, Any]] = None):
        self.compiler = compiler
        self.settings: dict[str, Any] = settings or {}
        self.assertions: list[Any] = []

    def has_builtin(self, name: str) -> bool:
        return name in COMPILER_BUILTINS[self.compiler]

    def merge(self, global_settings: dict[str, Any]) -> dict[str, Any]:
        return global_settings | self.settings

    def add(self, key: str, value: tuple[list, dict]):
        if key not in COMPILER_BUILTINS[self.compiler]:
            return False

        target_builtin = COMPILER_BUILTINS[self.compiler][key]
        if getattr(target_builtin, "repeatable", False):
            args, kwargs = value

            if key not in self.settings:
                self.settings[key] = []

            self.settings[key].append((args, kwargs))
        else:
            self.settings[key] = value

        return True

    def copy(self):
        return CompilerSetting(self.compiler, self.settings.copy())


class SettingScope(VariableScope):
    def __init__(self, parent: Optional["SettingScope"] = None, data: dict[str, Any] | None = None):
        super().__init__(parent, data)

        # deep copy compiler configurations
        self.compilers: dict[str, CompilerSetting] = (
            {compiler.name: CompilerSetting(compiler) for compiler in COMPILERS} if parent is None else
            {key: settings.copy() for key, settings in parent.compilers.items()})

    def parse_setting(self, name_node: Node, argument_list: Node, log: Diagnostics):
        assert name_node.text is not None, "Invalid text node"
        name = name_node.text.decode()
        args, kwargs = self.parse_argument_list(argument_list)
        if name in self.compilers:
            # ie [[Clang(enabled=false)]];
            if len(args) != 0:
                raise ParseError(f"Compiler configuration for `{name}` expects no positional arguments",
                                 get_node(args[0]) or argument_list)

            for key, value in kwargs.items():
                self.compilers[name].add(key, ([value], {}))
            return True

        if name in COMPILER_BUILTINS[SettingScope]:
            # affect all compilers
            self.add(name, (args, kwargs))
            return True

        found = False
        for compiler in self.compilers.values():
            if not compiler.has_builtin(name):
                continue
            compiler.add(name, (args, kwargs))
            found = True

        return found

    def run_builtin(self, compiler: type[Compiler], name: str, *args, **kwargs) -> Any:
        return COMPILER_BUILTINS[compiler][name](compiler, *args, **kwargs)

    def effective_configurations(self):
        for settings in self.compilers.values():
            compilers = settings.compiler.discover()
            effective_config = self.all | settings.settings
            for config, value in effective_config.items():
                if not settings.has_builtin(config):
                    continue

                for args, kwargs in value if isinstance(value, list) else [value]:
                    compilers = self.run_builtin(settings.compiler, config, compilers, *args, **kwargs)

            yield from compilers

    @builtin
    def standard(self): ...

    @builtin
    def language(self): ...


class ParseContext:
    def __init__(self, diagnostics: Diagnostics, variables: Optional[VariableScope], settings: Optional[SettingScope]):
        self.log = diagnostics

        self.variables: VariableScope = VariableScope(variables)
        self.settings: SettingScope = SettingScope(settings)
        self.used: set[Variable] = set()

    def add_variable(self, variable: Node, value: Any):
        assert variable.text is not None, "Invalid text node"
        variable_name: str = variable.text.decode()
        if variable_name in self.variables:
            raise ParseError(f"Redefining {variable} is not allowed", variable)

        if variable_name in self.variables.enclosing_scope:
            self.log.emit_warning(f"definition of `{decorated(variable_name, style=Style.BRIGHT)}` shadows global definition",
                                  variable)

        self.variables.add(variable_name, value)

    def parse_attr_node(self, node: Node):
        assert node.type == "attributed_statement", f"Wrong type: {node.type}"
        assert node.named_children[-1].type in ("expression_statement", "labeled_statement"), \
            f"Wrong type: {node.named_children[-1].type}"

        remove = False

        if node.named_children[-1].type == "expression_statement":
            # regular attribute
            # [[identifier]], [[identifier(foo)]] etc

            if node.has_error:
                raise ParseError("Invalid syntax", find_error_node(node) or node)

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

                    self.add_variable(name_node, self.variables.parse(value))
                    remove |= True
                elif "function" in match:
                    fnc = match.get("function")
                    assert fnc is not None, "Could not get function node"
                    assert not isinstance(fnc, list), "More than one function node found"
                    self.add_variable(name_node, Lambda(name, fnc, self.variables))
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
        elif name == "import":
            # special case [[import("package")]];
            self.parse_import(argument_list)
            return True

        return self.settings.parse_setting(name_node, argument_list, self.log)

    def parse_use_args(self, argument_list: Node):
        args, kwargs = self.variables.parse_argument_list(argument_list)
        if len(args) == 0:
            raise ParseError("Use needs at least one argument", argument_list)
        if len(kwargs) != 0:
            raise ParseError(f"Unrecognized keyword arguments {kwargs}", get_node(kwargs[0]) or argument_list)

        for variable in args:
            if not isinstance(variable, Variable):
                raise ParseError("Wrong argument type. All args must be variables",
                                 get_node(variable) or argument_list)

            if self.variables.get(variable.name) is None:
                raise UndefinedError(f"Unrecognized variable {variable} marked used",
                                     get_node(variable) or argument_list,
                                     str(variable), self, is_function=False)

            self.used.add(variable)

    def parse_import(self, argument_list: Node):
        args, kwargs = self.variables.parse_argument_list(argument_list)
        if len(args) == 0:
            raise ParseError("Must specify what to import.", argument_list)

        valid_kwargs = ["symbols", "alias"]
        for key in kwargs:
            if key in valid_kwargs:
                continue
            raise ParseError(f"Unrecognized keyword argument {key}", get_node(key) or argument_list)

        module_name = args[0]
        symbols = args[1] if len(args) == 2 else kwargs.get("symbols")
        alias = kwargs.get("alias")

        if not module_name:
            raise ParseError(f"Invalid package {module_name}", get_node(module_name) or argument_list)
        if not isinstance(module_name, str):
            raise ParseError(f"Invalid argument type: {type(module_name).__name__}, expected str",
                             get_node(module_name) or argument_list)

        module = importlib.import_module(module_name)

        def check_symbol(symbol):
            if not hasattr(module, symbol):
                raise ParseError(f"Module {module_name} lacks requested symbol {symbol}",
                                 get_node(module_name) or argument_list)

        if isinstance(symbols, str):
            check_symbol(symbols)
            self.variables[alias or symbols] = getattr(module, symbols)

        elif symbols:
            if alias and len(symbols) > 1:
                raise ParseError("Cannot combine alias with more than one explicitly imported symbol",
                                 get_node(alias) or argument_list)

            for symbol in symbols:
                check_symbol(symbol)
                self.variables[symbol] = getattr(module, symbol)

        else:
            self.variables[alias or module_name] = module

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
            args, kwargs = self.variables.parse_argument_list(argument_list)

            assert name_node.text, "Invalid text node"
            name = name_node.text.decode()
            self.settings.compilers[varname].add(name, (args, kwargs))
            return True

        self.add_variable(prefix, self.variables.parse(name_node)
                          if argument_list is None else self.variables.parse_call(name_node, argument_list))

        return True


class Lambda:
    def __init__(self, name: str, node: Node, scope: VariableScope):
        self.name = name
        self.__name__ = name

        *args, fnc = list(self.parse_nested_comma(node))
        self.arg_names = args

        self.scope = scope.copy()
        self.scope.add(self.name, self)

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

        function_scope = VariableScope(self.scope)

        for name, value in zip(self.arg_names, args):
            assert name.text is not None, "Invalid text node"
            arg_name: str = name.text.decode()
            function_scope.add(arg_name, self.scope.evaluate(value))

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


class Test(ParseContext):
    def __init__(self, node: dict[str, Node], diagnostics: Diagnostics, variables: Optional[VariableScope], settings: Optional[SettingScope]):
        super().__init__(diagnostics, variables, settings)
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
            if (var := self.variables.get(ident)) is None:
                continue

            if ident in BUILTINS:
                # don't mark builtins as used
                continue
            
            if callable(var):
                # don't mark functions as used
                continue

            self.used.add(self.variables.parse(ident_node))

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
        # TODO get used variables from parent scope (file scope)
        for var in self.used:
            variable = self.variables.evaluate(var)
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


class TranslationUnit(ParseContext):
    def __init__(self, source: str | bytes, source_path: Optional[Path] = None, logger: Optional[Diagnostics] = None):
        self.raw_source: bytes = source.encode("utf-8") if isinstance(source, str) else source
        self.source_path = source_path
        self.log: Diagnostics = logger or Diagnostics(self.raw_source.decode("utf-8"), source_path)
        # builtins shall be treated as globals
        super().__init__(self.log, VariableScope(data=BUILTINS), None)

        self.tests: list[Test] = []
        # TODO
        self.skip_list: list[Range | Test] = []

    @staticmethod
    def from_file(path: Path):
        assert path.exists()
        return TranslationUnit.from_source(path.read_bytes(), path)

    @staticmethod
    def from_source(source: str | bytes, path: Optional[Path] = None):
        logger = Diagnostics(source if isinstance(source, str) else source.decode("utf-8"), path)
        try:
            tu = TranslationUnit(source, path, logger=logger)
            tu.parse(tu.tree)
            return tu

        except ParseError as exc:
            logger.emit_error(exc.message, exc.node)
            raise SystemExit(1) from exc

        except UndefinedError as exc:
            if similar := exc.scope.nearest_match(exc.name, exc.is_function):
                # add suggestions
                start_column = exc.node.range.start_point.column
                decorated_name = decorated(similar, fg=Fore.GREEN, style=Style.BRIGHT)
                logger.emit_error(f"{exc.message}; did you mean `{decorated_name}`?", exc.node)
                stderr.write(f"{' '*5}| {' '*start_column}{decorated_name}\n")
            else:
                # print only the error if there are no close matches
                logger.emit_error(exc.message, exc.node)

            raise SystemExit(1) from exc

    def parse(self, tree_root: Node):
        self.skip_list = []
        for _, node in QUERY["test"].matches(tree_root):
            if "kind" in node and "code" in node:
                # test
                test = Test(node, self.log, self.variables, self.settings)
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
    def tree(self):
        parser = Parser(CPP)
        return parser.parse(self.raw_source).root_node

    @cached_property
    def lines(self):
        return self.raw_source.decode("utf-8").splitlines()

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
            lines.append(f"    Configurations: {list(test.settings.effective_configurations())}")
            for compiler_name, compiler in test.settings.compilers.items():
                lines.append(f"    {compiler_name}")
                lines.append(f"      {compiler.assertions}")

        return "\n".join(lines)


def parse(source_path: Path):
    source = TranslationUnit.from_file(source_path)
    return source.source, source.tests


def print_tree(source: Path, query=None):
    parser = Parser(CPP)
    tree_obj = parser.parse(source.read_bytes())
    if query is None:
        print(tree_obj.root_node)
    else:
        matches = QUERY[query].matches(tree_obj.root_node)
        print(f"{len(matches)} matches:")
        pprint(matches)
