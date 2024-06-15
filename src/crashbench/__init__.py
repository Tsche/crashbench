from .cli import main

__all__ = ["main"]


# from pprint import pprint

# from pathlib import Path
# import tree_sitter_cpp
# from tree_sitter import Language, Parser

# CPP_LANGUAGE = Language(tree_sitter_cpp.language())


# def mainf():
#     parser = Parser(CPP_LANGUAGE)
#     path = Path.cwd() / "example" / "benchmark.cpp"
#     query_path = Path.cwd() / "query2.lisp"
#     source_code = path.read_bytes()
#     tree = parser.parse(source_code)

#     print(str(tree.root_node))
#     print()
#     # query = CPP_LANGUAGE.query("(attributed_statement) @attr")
#     #     query = CPP_LANGUAGE.query("""\
#     # (function_definition
#     #     (attribute_declaration)+ @attr) @fnc
#     # """)

#     query = CPP_LANGUAGE.query(query_path.read_text())
#     pprint(query.matches(tree.root_node))
#     for idx, captures in query.matches(tree.root_node):
#         for key, value in captures.items():
#             print(f"{key} = {value.text.decode()}")
#         typename = captures["type"].text.decode()
#         name = captures["name"].text.decode()
#         if typename == "using" and name == "namespace":
#             # tree sitter failed on attributed using-declaration
#             assert captures["decl"].has_error, "Couldn't recover from erroneous parse."
            
#             next_sibling = captures["name"].next_sibling
#             assert next_sibling.is_error, "Next sibling is expected to be an error but wasn't"
#             assert next_sibling.child_count == 1, "No identifier child found"
#             ident = next_sibling.children[0]
#             print(ident.text.decode())
#             print(next_sibling)


#         # print(captures["decl"].text.decode())
#         # print(f"=> {captures['type'].text.decode()} {captures['ident'].text.decode()} = {captures['attr'].text.decode()[2:-2]}")
#         # print()
#         # for capture in captures["attr"]:
#         #     print(capture.text.decode('utf-8'))
#         # print(captures["fnc"].text.decode())
