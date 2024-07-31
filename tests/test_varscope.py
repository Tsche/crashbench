from typing import Iterable
from crashbench.parser import VariableScope
import tree_sitter_cpp
from tree_sitter import Language, Parser
import sys
CPP = Language(tree_sitter_cpp.language())
CPP_PARSER = Parser(CPP)

def parse(data: str):
    return CPP_PARSER.parse(data.encode())

def get_node_for(expression: str):
    tree = parse(f"auto _ = {expression};")
    node_query = CPP.query("""
(translation_unit 
    (declaration type: 
        (placeholder_type_specifier (auto)) 
        declarator: 
            (init_declarator 
                declarator: (identifier) 
                value: (_) @value)
    )
)""")
    captures = node_query.captures(tree.root_node)
    assert len(captures) == 1
    node, capture_name = captures[0]
    assert capture_name == "value"
    return node

parser = VariableScope()

def test_parse_numerals():
    assert parser.parse(get_node_for("0")) == 0
    assert parser.parse(get_node_for("1")) == 1
    assert parser.parse(get_node_for("-1")) == -1
    assert parser.parse(get_node_for("1.1")) == 1.1
    # assert parser.parse(get_node_for("100'000")) == 100000


def test_parse_bool():
    true =  parser.parse(get_node_for("true"))
    false = parser.parse(get_node_for("false"))
    assert true == True
    assert false == False
    assert hasattr(true, "node")
    assert hasattr(false, "node")

def test_parse_none():
    nullptr = parser.parse(get_node_for("nullptr"))
    assert nullptr == None
    assert hasattr(nullptr, "node")

def test_parse_string():
    empty = parser.parse(get_node_for('""'))
    regular = parser.parse(get_node_for('"foo"'))
    concatenated = parser.parse(get_node_for('"foo""bar"'))

    assert empty == ""
    assert regular == "foo"
    assert concatenated == "foobar"

    assert hasattr(empty, "node")
    assert hasattr(regular, "node")
    # assert hasattr(concatenated, "node")
    # assert isinstance(concatenated.node, list)
    # assert len(concatenated.node) == 2

if __name__ == "__main__":
    test_parse_numerals()
    test_parse_bool()
    test_parse_none()
    test_parse_string()